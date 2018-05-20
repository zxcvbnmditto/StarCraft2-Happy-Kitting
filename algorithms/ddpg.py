
import numpy as np
import tensorflow as tf

#### Global Variables
MAX_EPISODES = 200
MAX_EP_STEPS = 200

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

#### Actor
class Actor():
    # __init__
    '''
        Setup
        the actor network, actor target network
        soft_update
        train_op => gradient descent
        policy_grads => gradient of policy/action over target_params
    '''
    def __init__(self, sess, action_dim, learning_rate):
        self.sess = sess
        self.a_dim = action_dim
        self.lr = learning_rate

        with tf.variable_scope('Actor'):
            # action network
            self.a = build_actor(S, 'Actor/actor_network', trainable = True)
            # target action network
            self.target_a = build_actor(S_, 'Actor/target_network', trainable = False)

        # randomly initialize the parameters for neural network
        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/actor_network')
        self.target_a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/target_network')

        # slowly update target_params
        self.soft_update = [
            tf.assign(t, (1 - TAU) * t + TAU * e)
            for (t,e) in zip(self.target_a_params, self.a_params)
        ]

    def add_policy_gradient(self, q_grads):
        # gradient of policy over target params
        self.policy_gradient = tf.gradients(ys = self.a, xs = self.target_a_params, grad_ys = q_grads)

        # train_op
        self.train_actor = tf.train.GradientDescentOptimizer(self.lr).minimize(-self.policy_gradient, var_list = self.a_params)

    # _build_net
    # Build the neural network. 
    # 2 hidden layer, 30 nodes each
    def build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0, 0.1)
            init_b = tf.constant_initializer(0.1)
            node_count = 30

            with tf.variable_scope('l1'):
                l1 = tf.layers.dense(
                    s,
                    node_count,
                    activation = tf.nn.relu,
                    kernel_initializer= init_w,
                    bias_initializer = init_b,
                    name = 'l1',
                    trainable = trainable
                )
            with tf.variable_scope('l2'):
                l2 = tf.layers.dense (
                    l1,
                    node_count,
                    activation=tf.nn.relu,
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    name='l2',
                    trainable = trainable
                )

            with tf.variable_scope('output_layer'):
                output_layer = tf.layers.dense(
                    l2,
                    self.a_dim,
                    activation = tf.nn.relu,
                    kernel_initializer = init_w,
                    bias_initializer = init_b,
                    name = 'output_layer',
                    trainable = trainable
                )

        # output might have to fix later by rescaling the output actions
        return output_layer

    # learn
    def learn(self, s):
        self.sess.run(self.train_op, feed_dict = {S: s})
    
    # choose action
    def choose_action(self, s):
        self.sess.run(self.a, feed_dict = {S: s})

#### Critic



class Critic():
    # __init__
    '''
        Setup
        the critic network, critic target network
        soft_update
        target_q
        td_error
        loss
        train_op => gradient descent
        q_grads gradient of q over action
    '''

    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.a = a
        self.a_ = a_

        # setup critic network and critic target network
        with tf.variable_scope('Critic'):
            self.q = build_critic(S, self.a, 'Critic/critic_network', trainable = True)
            self.target_q = build_critic(S_, self.a_, 'Critic/critic_target', trainable = False)

        self.q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/critic_network')
        self.target_q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/critic_target')

        self.soft_update = [
            tf.assign(t, (1 - TAU) * t + TAU * e )
            for (t, e) in zip(self.target_q_params, self.q_params)
        ]

        self.target_q = R + self.gamma * self.target_q

        self.td_error = self.target_q - self.q

        self.loss = tf.reduce_mean(tf.square(self.td_error))

        self.train_critic = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list = self.q_params)

        self.q_gradient = tf.gradients(ys = self.q , xs = self.a)

    # _build_net
    # Build the neural network. 
    # 2 hidden layer, 30 nodes each
    def build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0, 0.1)
            init_b = tf.constant_initializer(0.1)
            node_count = 30

            with tf.variable_scope('l1'):
                w1_s = tf.get_variable('w1_s', [self.s_dim, node_count], initializer = init_w, trainable = trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, node_count], initializer = init_w, trainable = trainable)
                b1 = tf.get_variable('b1_a', [1, node_count], initializer = init_b, trainable = trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('l2'):
                l2 = tf.layers.dense(
                    l1,
                    node_count,
                    activation=tf.nn.relu,
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    name='l2',
                    trainable=trainable
                )

            with tf.variable_scope('output_layer'):
                output_layer = tf.layers.dense(
                    l2,
                    1,
                    activation = tf.nn.relu,
                    kernel_initializer = init_w,
                    bias_initializer = init_b,
                    name = 'output_layer',
                    trainable = trainable
                )

        return output_layer

    # learn
    def learn(self, s, a, r, s_):
        self.sess.run(self.train_critic, dict_feed={S: s, self.a: a, R: r, S_: s})

#### Memory

class Memory():
    # __init__
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.counter = 0

    # store_transitions
    def store_transitions(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # stack the information horizontally
        index = self.counter % self.capacity 
        self.data[index, :] = transition
        self.counter += 1

    # randomize_memory
    def randomized_memory(self, n):
        assert self.counter >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

#### initialization + sess
sess = tf.Session()

'''
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()
'''

'''
actor = Actor(sess, action_dim, LR_A)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)
'''

sess.run(tf.global_variables_initializer())

# have to add the shape
S = tf.placeholder('S', dtype = tf.float32, name = "S")
R = tf.placeholder('S', dtype = tf.float32, name = "R")
S_ = tf.placeholder('S', dtype = tf.float32, name = "S_")

tf.summary.FileWriter("logs/", sess.graph)


#### environment
'''
for play_count in MAX_EPISODES:
    env = env.get_current_state()
    
    
'''

