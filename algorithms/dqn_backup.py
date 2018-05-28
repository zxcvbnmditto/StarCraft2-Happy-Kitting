w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        #with tf.variable_scope('eval_net'):
            # # hidden layer 1
            # with tf.variable_scope('layer1'):
            #     e_z1 = tf.layers.dense(self.s, 6, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='e_z1')
            #     w1 = tf.get_default_graph().get_tensor_by_name('eval_net/layer1/e_z1/kernel:0')
            #     b1 = tf.get_default_graph().get_tensor_by_name('eval_net/layer1/e_z1/bias:0')
            #     tf.summary.histogram('e_z1', e_z1)
            #     tf.summary.histogram('w1', w1)
            #     tf.summary.histogram('b1', b1)
            #
            #     e_bn1 = tf.layers.batch_normalization(e_z1, training=True)
            #     tf.summary.histogram('e_bn1', e_bn1)
            #
            #     e_a1 = tf.nn.relu(e_bn1)
            #     tf.summary.histogram('e_a1', e_a1)
            #
            # # hidden layer 2
            # with tf.variable_scope('layer2'):
            #     e_z2 = tf.layers.dense(e_a1, 6, activation=None, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e2')
            #     tf.summary.histogram('e_z2', e_z2)
            #
            #     e_bn2 = tf.layers.batch_normalization(e_z2, training=True)
            #     tf.summary.histogram('e_bn2', e_bn2)
            #
            #     e_a2 = tf.nn.relu(e_bn2)
            #     tf.summary.histogram('e_a2', e_a2)
            #
            # ### output layer
            # with tf.variable_scope('output_layer'):
            #     self.q_eval = tf.layers.dense(e_a2, self.n_actions, activation=tf.nn.relu, kernel_initializer=w_initializer,
            #                                   bias_initializer=b_initializer, name='q')
            #     tf.summary.histogram('q_eval', self.q_eval)


        # ------------------ build target_net ------------------

        #with tf.variable_scope('target_net'):
            # # hidden layer 1
            # with tf.variable_scope('layer1'):
            #     t_z1 = tf.layers.dense(self.s_, 6, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t1')
            #     tf.summary.histogram('t_z1', t_z1)
            #
            #     t_bn1 = tf.layers.batch_normalization(t_z1, training=True)
            #     tf.summary.histogram('t_bn1', t_bn1)
            #
            #     t_a1 = tf.nn.relu(t_bn1)
            #     tf.summary.histogram('t_a1', t_a1)
            #
            # # hidden layer 2
            # with tf.variable_scope('layer2'):
            #     t_z2 = tf.layers.dense(t_a1, 6, activation=None, kernel_initializer=w_initializer,
            #                            bias_initializer=b_initializer, name='t2')
            #     tf.summary.histogram('t_z2', t_z2)
            #
            #     t_bn2 = tf.layers.batch_normalization(t_z2, training=True)
            #     tf.summary.histogram('t_bn2', t_bn2)
            #
            #     t_a2 = tf.nn.relu(t_bn2)
            #     tf.summary.histogram('t_a2', t_a2)
            #
            # ### output layer
            # with tf.variable_scope('output_layer'):
            #     self.q_next = tf.layers.dense(t_a2, self.n_actions, activation=tf.nn.relu, kernel_initializer=w_initializer,
            #                                   bias_initializer=b_initializer, name='t3')
            #     tf.summary.histogram('q_next', self.q_next)