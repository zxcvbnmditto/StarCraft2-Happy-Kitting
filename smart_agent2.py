import random
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_HIT_POINTS_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_UNIT_DENSITY_AA = features.SCREEN_FEATURES.unit_density_aa.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index


_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK_UP = 'attackup'
ACTION_ATTACK_DOWN = 'attackdown'
ACTION_ATTACK_LEFT = 'attackleft'
ACTION_ATTACK_RIGHT = 'attackright'
ACTION_ATTACK_UP_LEFT = 'attackupleft'
ACTION_ATTACK_UP_RIGHT = 'attackupright'
ACTION_ATTACK_DOWN_LEFT = 'attackdownleft'
ACTION_ATTACK_DOWN_RIGHT = 'attackdownright'
ACTION_SELECT_UNIT_1 = 'selectunit1'
ACTION_SELECT_UNIT_2 = 'selectunit2'
ACTION_SELECT_UNIT_3 = 'selectunit3'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK_UP,
    ACTION_ATTACK_DOWN,
    ACTION_ATTACK_LEFT,
    ACTION_ATTACK_RIGHT,
    ACTION_ATTACK_UP_LEFT,
    ACTION_ATTACK_UP_RIGHT,
    ACTION_ATTACK_DOWN_LEFT,
    ACTION_ATTACK_DOWN_RIGHT,
    ACTION_SELECT_UNIT_1,
    ACTION_SELECT_UNIT_2,
    ACTION_SELECT_UNIT_3
]

KILL_UNIT_REWARD = 1
LOSS_UNIT_REWARD = -1


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
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

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.memory_counter = 0

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        # print(observation.shape)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
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

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.dqn = DeepQNetwork(
            len(smart_actions),
            25,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
        )

        self.previous_killed_unit_score = 0
        self.previous_lost_unit_score = 0

        self.previous_action = None
        self.previous_state = None


    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs,):
        super(SmartAgent, self).step(obs)

        #time.sleep(0.2)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['screen'][_UNIT_TYPE]
        units_count = obs.observation['multi_select'].shape[0]

        units = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        hp = [0, 0, 0, 0]

        for i in range(units_count):
            units[i] = (obs.observation['multi_select'][i])
            hp[i] = obs.observation['multi_select'][i][2]

        hp[3] = obs.observation['single_select'][0][2]

        unit_type = np.array(unit_type).flatten()
        units = np.array(units).flatten()
        hp = np.array(hp).flatten()

        current_state = np.hstack((units, hp))

        killed_unit_score = obs.observation['score_cumulative'][5]
        lost_unit_score = units_count

        # print(current_state, current_state.shape, self.steps)

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if lost_unit_score < self.previous_lost_unit_score:
                reward += LOSS_UNIT_REWARD

            # print(np.array(current_state), self.previous_state, self.previous_action, reward, self.steps)
            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.dqn.learn()
            # print(self.reward, self.steps)

        rl_action = self.dqn.choose_action(np.array(current_state))

        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_lost_unit_score = lost_unit_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return self.perform_action(obs, smart_action, player_x, player_y)


    def perform_action(self, obs, action, xloc, yloc):
        if action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif action == ACTION_SELECT_ARMY:
           if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif action == ACTION_SELECT_UNIT_1:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 1 and len(yloc) >= 1:
                    #print(1)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [0]])

        elif action == ACTION_SELECT_UNIT_2:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 2 and len(yloc) >= 2:
                    #print(2)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [1]])

        elif action == ACTION_SELECT_UNIT_3:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 3 and len(yloc) >= 3:
                    #print(3)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [2]])

        elif action == ACTION_ATTACK_UP:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [0, 36]])

        elif action == ACTION_ATTACK_DOWN:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [60, 36]])

        elif action == ACTION_ATTACK_LEFT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [32, 0]])

        elif action == ACTION_ATTACK_RIGHT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [32, 60]])

        elif action == ACTION_ATTACK_UP_LEFT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [0, 0]])

        elif action == ACTION_ATTACK_UP_RIGHT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [0, 60]])

        elif action == ACTION_ATTACK_DOWN_LEFT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [60, 0]])

        elif action == ACTION_ATTACK_DOWN_RIGHT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [60, 60]])

        return actions.FunctionCall(_NO_OP, [])
