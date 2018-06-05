import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 1 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # target get updates slowly and is used to
        # eval is used to predict the next step
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval_net', trainable=True)
            a_ = self._build_a(self.S_, scope='target_net', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval_net', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target_net', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        with tf.variable_scope('q_target'):
            q_target = self.R + GAMMA * q_
            tf.summary.histogram('q_target', q_target)

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        with tf.variable_scope('td_error'):
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            tf.summary.scalar('td_error', td_error)

        with tf.variable_scope('ctrain'):
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
            #tf.summary.histogram('ctrain', self.ctrain)

        with tf.variable_scope('a_loss'):
            self.a_loss = - tf.reduce_mean(q)    # maximize the q
            tf.summary.scalar('loss', self.a_loss)

        with tf.variable_scope('atrain'):
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)
            #tf.summary.histogram('atrain', self.atrain)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # total learning step
        self.saver = tf.train.Saver()
        self.learn_step_counter = 0
        self.cost_his = []
        self.reward = []

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        # forward feed the observation and get q value for every actions
        actions_probs = self.sess.run(self.a, feed_dict={self.S: observation})
        action = np.random.choice(a=range(len(actions_probs.flatten())), p=actions_probs.flatten())

        return action


    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        # randomly sample the batch size amount of data to learn from
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        # dissect the sampled data into different categories
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        _, loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs})
        _, summary = self.sess.run([self.ctrain, self.merged_summary], {self.S: bs, self.a: ba, self.R: br,
                                                                       self.S_: bs_})
        self.cost_his.append(loss)
        self.writer.add_summary(summary, self.learn_step_counter)
        self.writer.flush()

        self.learn_step_counter += 1


    def store_transition(self, s, a, r, s_):
        self.reward.append(r)
        transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 6, activation=tf.nn.sigmoid, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)

            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 6
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            return tf.layers.dense(net, 6, trainable=trainable)  # Q(s,a)

    def plot_reward(self, path, save):
        plt.plot(np.arange(len(self.reward)), self.reward)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/reward.png')
        plt.close()

    def plot_cost(self, path, save):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/cost.png')
        plt.close()

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])