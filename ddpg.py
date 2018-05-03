
import numpy as np
import tensorflow as tf

#### Global Variables
TAU = 0.01


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
            tf.assign(t, (1 - TAU) * t + TAU * e )
            for (t,e) in zip(self.target_a_params, self.a_params)
        ]

        # gradient of policy over target params
        self.policy_gradient = tf.gradients(ys = self.a , xs = self.target_a_params, grad_ys = q_grads)

        # train_op
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(-self.policy_gradient, var_list = self.a_params)

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

    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):

    # _build_net
    # Build the neural network. 
    # 2 hidden layer, 30 nodes each
    def build_critic(self, s, a, scope, trainable):

    # learn
    def learn(self, s, a, r, s_):

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


#### environment