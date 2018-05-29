import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.cost_his = []
        self.reward = []
        self.memory_counter = 0

    def _build_net(self):
        # ------------------ all inputs ------------------------
        # None => can be any number
        # In most cases, None is used for fetch a changing amount of sample
        # Here None would be the batch size
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('eval_net'):
            # layer 1
            with tf.variable_scope('layer1'):
                l1 = self.add_layer(inputs=self.s, in_size=self.n_features, out_size=6, activation_function=tf.nn.relu, norm=True, )

            # layer 2
            with tf.variable_scope('layer2'):
                l2 = self.add_layer(inputs=l1, in_size=6, out_size=6, activation_function=tf.nn.relu, norm=True, )

            with tf.variable_scope('layer3'):
                l3 = self.add_layer(inputs=l2, in_size=6, out_size=6, activation_function=tf.nn.relu, norm=True, )

            # output layer
            with tf.variable_scope('output_layer'):
                self.q_eval = self.add_layer(inputs=l3, in_size=6, out_size=self.n_actions, activation_function=tf.nn.relu, norm=False, )

        with tf.variable_scope('target_net'):
            # layer 1
            with tf.variable_scope('layer1'):
                l1 = self.add_layer(inputs=self.s, in_size=self.n_features, out_size=6, activation_function=tf.nn.relu, norm=True, )

            # layer 2
            with tf.variable_scope('layer2'):
                l2 = self.add_layer(inputs=l1, in_size=6, out_size=6, activation_function=tf.nn.relu, norm=True, )

            with tf.variable_scope('layer3'):
                l3 = self.add_layer(inputs=l2, in_size=6, out_size=6, activation_function=tf.nn.relu, norm=True, )

            # output layer
            with tf.variable_scope('output_layer'):
                self.q_next = self.add_layer(inputs=l3, in_size=6, out_size=self.n_actions, activation_function=tf.nn.relu, norm=False, )


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
            # tf.summary.histogram('q_target', self.q_target)
        with tf.variable_scope('q_eval'):
            # tf.shape(self.a)[0] => gets the None/ gets the passed in batch size
            # self.q_eval => batch_size * n_actions, a_indices => batch_size * 2(1 represents the number of batch, and the other represents the number of action in the batch)
            # self.q_eval_wrt_a calculates the q_eval among all the sample batches' actions => size of batch size * 1
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            # tf.summary.histogram('q_eval', self.q_eval_wrt_a)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            with tf.control_dependencies(update_ops):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def add_layer(self, inputs, in_size, out_size, activation_function=None, norm=False):
        # weights and biases (bad initialization for this case)
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # just a default case over here
        z = Wx_plus_b

        # normalize fully connected product
        if norm:
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],  # the dimension you wanna normalize, here [0] for batch
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()

            z = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        # activation
        if activation_function is None:
            a = z
        else:
            a = activation_function(z)

        tf.summary.histogram('Wx_plus_b', Wx_plus_b)
        tf.summary.histogram('z', z)
        tf.summary.histogram('a', a)
        tf.summary.histogram('weights', Weights)
        tf.summary.histogram('biases', biases)

        return a

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        self.reward.append(r)
        # transform a and r into 1D array
        transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, disabled_actions):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

            # assign to 0 since 0 is the min of relu
            for i in range(0, len(disabled_actions)):
                actions_value[0][disabled_actions[i]] = 0 # may have to change

            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost, summary = self.sess.run(
            [self._train_op, self.loss, self.merged_summary],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)
        self.writer.add_summary(summary, self.learn_step_counter)
        self.writer.flush()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_reward(self, path, save):
        plt.plot(np.arange(len(self.reward)), self.reward)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/reward.png')
        plt.show()

    def plot_cost(self, path, save):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/cost.png')
        plt.show()

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])
