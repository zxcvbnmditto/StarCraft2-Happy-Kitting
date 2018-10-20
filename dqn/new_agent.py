import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import time
import random
import matplotlib.pyplot as plt
from new_dqn import DQN
from memory import Memory
from pysc2.lib import actions

# Using Actions: Attack, Move, Select Rectangle
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

# return actions.FunctionCall(_SELECT_RECT, [_NOT_QUEUED, top_left_coord, bottom_right_coord])
# return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])  # x,y => col,row
# return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [x, y]])  # x,y => col,row

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_UNIT_ALLIANCE = 1
_UNIT_HEALTH = 2
_UNIT_SHIELD = 3
_UNIT_X = 12
_UNIT_Y = 13
_UNIT_IS_SELECTED = 17
_UNIT_COOLDOWN = 25

_NOT_QUEUED = [0]
_QUEUED = [1]

ATTACK_TARGET = 'attacktarget'
MOVE_UP = 'moveup'
MOVE_DOWN = 'movedown'
MOVE_LEFT = 'moveleft'
MOVE_RIGHT = 'moveright'
ACTION_SELECT_UNIT = 'selectunit'

smart_actions = [
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    ATTACK_TARGET
]

# Change this if using a different map
# Currently running HK2V1
DEFAULT_ENEMY_COUNT = 1
DEFAULT_PLAYER_COUNT = 2

ENEMY_MAX_HP = 150
PLAYER_MAX_HP = 60


class DQN_Agent(object):
    def __init__(self, FLAGS, sess):
        self.sess = sess
        self.max_steps = FLAGS.max_steps
        self.model = DQN(FLAGS)
        # Necessary
        self.obs_spec = None
        self.action_spec = None
        self.prev_action = None
        self.prev_state = None
        # Change this base on map
        self.prev_enemy_count = 1
        self.prev_player_count = 2
        self.channel_size = FLAGS.channel_size
        self.state_size = FLAGS.screen_resolution
        # Calculate the win
        self.win = 0
        self.memory = Memory(FLAGS)

    def step(self, obs, episode, step):
        # Get state
        rgb_screen = np.array(obs.observation['rgb_screen'])
        rgb_screen = np.transpose(rgb_screen, [1, 0, 2])

        if self.channel_size == 4:
            hp_ratio_screen = np.array(obs.observation['feature_screen']['unit_hit_points_ratio'])
            hp_ratio_screen = np.transpose(hp_ratio_screen, [1, 0])
            hp_ratio_screen = np.expand_dims(hp_ratio_screen, axis=2)
            features = np.concatenate((rgb_screen, hp_ratio_screen), axis=-1)
        elif self.channel_size == 3:
            features = rgb_screen
        else:
            print("Fail to create the correct state. Check channel_size")
            features = rgb_screen

        curr_state = (features / 127.5) - 1
        # print(features, features.shape)

        # Choose action
        # Later on dqn picks action

        # For now random pick one
        if episode < 100:
            if np.random.rand(1)[0] > 0.3:
                curr_action = np.random.randint(self.channel_size, size=1)
                coords1 = np.random.randint(self.state_size, size=2)
                coords2 = np.random.randint(self.state_size, size=2)
            else:
                curr_action, coords1, coords2 = self.choose_action(curr_state)
        elif 100 <= episode < 1000:
            if np.random.rand(1)[0] > 0.8:
                curr_action = np.random.randint(self.channel_size, size=1)
                coords1 = np.random.randint(self.state_size, size=2)
                coords2 = np.random.randint(self.state_size, size=2)
            else:
                curr_action, coords1, coords2 = self.choose_action(curr_state)
        else:
            if np.random.rand(1)[0] > 0.95:
                curr_action = np.random.randint(self.channel_size, size=1)
                coords1 = np.random.randint(self.state_size, size=2)
                coords2 = np.random.randint(self.state_size, size=2)
            else:
                curr_action, coords1, coords2 = self.choose_action(curr_state)

        next_action = None
        if curr_action == 0:
            # print("Select {} {}".format(coords1, coords2))
            next_action = actions.FunctionCall(_SELECT_RECT, [_NOT_QUEUED, coords1, coords2])
        elif curr_action == 1:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                # print("Move ")
                next_action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coords1])  # x,y => col,row
            else:
                next_action = actions.FunctionCall(_NO_OP, [])
        elif curr_action == 2:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                # print("Attack ")
                next_action = actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, coords1])  # x,y => col,row
            else:
                next_action = actions.FunctionCall(_NO_OP, [])

        # Get Reward
        reward = self.get_reward(obs, step)

        # record the transitions to memory and learn by DQN
        if self.prev_action is not None:
            # print(step, self.memory.temp_count, reward, self.prev_reward)
            # store prev, curr instead of curr, next bc we cannot get next directly
            self.memory.store_temp_transition(self.prev_state, self.prev_action, self.prev_reward, curr_state)

        self.prev_state = curr_state
        self.prev_action = curr_action
        self.prev_reward = reward

        return next_action

    def choose_action(self, obs):
        feed_dict = {self.model.s: np.expand_dims(obs, axis=0), self.model.training: False}
        q_curr, coords_1, coords_2 = self.sess.run(
            [self.model.q_curr, self.model.coords_1, self.model.coords_2], feed_dict=feed_dict)
        action = np.argmax(q_curr, axis=-1)

        coords_1 = (coords_1[0] * self.state_size).astype(int)
        coords_2 = (coords_2[0] * self.state_size).astype(int)

        return action, coords_1, coords_2

    def get_reward(self, obs, step):
        curr_enemy_unit_count = 0
        curr_player_unit_count = 0
        units_features = obs.observation['feature_units']

        for unit_features in units_features:
            if unit_features[_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                curr_enemy_unit_count += 1
            else:
                curr_player_unit_count += 1

        # reward: 1 if an enemy is killed, 0 for other
        enemy_killed_reward = self.prev_enemy_count - curr_enemy_unit_count
        # reward: 0 if no unit is killed, -x for x number of unit killed
        player_unit_saved_reward = curr_player_unit_count - self.prev_player_count
        reward = enemy_killed_reward + player_unit_saved_reward

        self.prev_enemy_count = curr_enemy_unit_count
        self.prev_player_count = curr_player_unit_count
        # print(reward)

        # Probably not bc step_mul might skip the timestep
        if curr_enemy_unit_count == 0 and curr_player_unit_count > 0:
            self.win += 1

        # Tie (game didn't end before max_steps) => reward = -1
        if (step == self.max_steps - 1 and curr_enemy_unit_count != 0):
            reward = -1

        return reward


    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        print("Reset Agent")
        # Need to reset the following vars to its initial states
        self.obs_spec = None
        self.action_spec = None
        self.prev_action = None
        self.prev_state = None
        self.prev_enemy_count = 1
        self.prev_player_count = 2



