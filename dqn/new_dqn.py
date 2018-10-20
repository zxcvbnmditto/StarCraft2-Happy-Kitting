import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DQN(object):
    def __init__(self, FLAGS):
        self.n_actions = FLAGS.n_actions
        self.n_features = FLAGS.channel_size
        self.lr = FLAGS.lr
        self.decay_rate = FLAGS.gamma
        # self.replace_target_iter = replace_target_iter
        # self.memory_size = memory_size
        self.batch_size = FLAGS.batch_size
        self.use_double = True
        # Fix these
        # self.epsilon_max = e_greedy
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Placeholders
        # input State
        self.s = tf.placeholder(tf.float32,
                            [None, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size], name='s')
        # input Next State
        self.s_ = tf.placeholder(tf.float32,
                            [None, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size], name='s_')
        # input Reward
        self.r = tf.placeholder(tf.float32, [None], name='r')
        # input Action
        self.a = tf.placeholder(tf.int64, [None], name='a')
        self.training = tf.placeholder(tf.bool)

        self.build()

    def q_network(self, state, trainable):
        # 256 * 256 * 3
        outputs = tf.layers.conv2d(state, filters=8, kernel_size=6, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 128 * 128 * 8
        outputs = tf.layers.conv2d(outputs, filters=16, kernel_size=6, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 64 * 64 * 16
        outputs = tf.layers.conv2d(outputs, filters=32, kernel_size=5, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 32 * 32 * 32
        outputs = tf.layers.conv2d(outputs, filters=64, kernel_size=5, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 16 * 16 * 64
        outputs = tf.layers.conv2d(outputs, filters=128, kernel_size=5, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 8 * 8 * 128
        outputs = tf.layers.conv2d(outputs, filters=256, kernel_size=5, strides=2, padding='SAME', trainable=trainable)
        outputs = tf.layers.batch_normalization(outputs, momentum=0.9, trainable=trainable, training=self.training)
        outputs = tf.nn.relu(outputs)

        # 4 * 4 * 256
        outputs = tf.layers.flatten(outputs)

        dense_v = tf.layers.dense(outputs, units=256, trainable=trainable)
        dense_v = tf.layers.batch_normalization(dense_v, momentum=0.9, trainable=trainable, training=self.training)
        dense_v = tf.nn.relu(dense_v)

        dense_a = tf.layers.dense(outputs, units=256, trainable=trainable)
        dense_a = tf.layers.batch_normalization(dense_a, momentum=0.9, trainable=trainable, training=self.training)
        dense_a = tf.nn.relu(dense_a)

        # dueling
        vt = tf.layers.dense(dense_v, units=1, trainable=trainable)
        at = tf.layers.dense(dense_a, units=self.n_actions, trainable=trainable)
        q_value = vt + at

        coords_1 = tf.layers.dense(outputs, units=2, trainable=trainable)
        coords_1 = tf.nn.sigmoid(coords_1)
        coords_2 = tf.layers.dense(outputs, units=2, trainable=trainable)
        coords_2 = tf.nn.sigmoid(coords_2)

        return q_value, coords_1, coords_2

    def build(self):
        # Update every step
        with tf.variable_scope("training_network"):
            self.q_curr, self.coords_1, self.coords_2 = self.q_network(self.s, trainable=True)

        # Update every n steps
        with tf.variable_scope("target_network"):
            self.q_next, _, _ = self.q_network(self.s_, trainable=False)

        # print(self.a, self.q_curr)

        with tf.variable_scope("Loss"):
            # q value estimate from next state s_ with decay_rate
            with tf.variable_scope("q_target"):
                batch_indices = tf.reshape(tf.range(self.batch_size, dtype=tf.int64), [self.batch_size, 1])

                # Double DQN
                if self.use_double:
                    max_q_indices = tf.reshape(tf.argmax(self.q_curr, axis=1), [self.batch_size, 1])
                    target_indices = tf.concat([batch_indices, max_q_indices], axis=-1)
                    self.q_target = self.r + self.decay_rate * tf.gather_nd(params=self.q_next, indices=target_indices)
                else:
                    # Vanilla DQN
                    self.q_target = self.r + self.decay_rate * tf.reduce_max(self.q_next, axis=1)

            # q value estimate from current state s
            with tf.variable_scope("q_training"):
                a_indices = tf.reshape(self.a, [self.batch_size, 1])
                training_indices = tf.concat([batch_indices, a_indices], axis=-1)
                self.q_training = tf.gather_nd(params=self.q_curr, indices=training_indices)

            with tf.variable_scope('q_loss'):
                self.q_loss = tf.reduce_mean(tf.squared_difference(self.q_training, self.q_target))

        self.q_loss_sumop = tf.summary.scalar('q_loss', self.q_loss)

        training_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_network')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # copy update
        with tf.variable_scope("copy_op"):
            self.copy_op = [d_var.assign(t_var) for d_var, t_var in zip(target_vars, training_vars)]

        with tf.control_dependencies(update_ops):
            with tf.variable_scope("TrainerQ"):
                self.trainerQ = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9).minimize(
                    self.q_loss, var_list=training_vars)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def save_model(self, sess, path, count):
        self.saver.save(sess, path + '/dqn.ckpt', count)

    def load_model(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters........................')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..........................')
            sess.run(tf.global_variables_initializer())

        # ckpt = tf.train.get_checkpoint_state(path)
        # self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # return int(ckpt.model_checkpoint_path.split('-')[-1])
