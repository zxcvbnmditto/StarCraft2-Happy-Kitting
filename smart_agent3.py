import numpy as np
import tensorflow as tf
import math
from algorithms.dqn import DeepQNetwork

from pysc2.lib import actions
from pysc2.lib import features
from sklearn.cluster import KMeans

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index


_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_UNIT_ALLIANCE = 1
_UNIT_HEALTH = 2
_UNIT_X = 12
_UNIT_Y = 13
_UNIT_RADIUS = 15 # find range
_UNIT_HEALTH_RATIO = 7

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

KILL_UNIT_REWARD = 3
LOSS_UNIT_REWARD = -1

class SmartAgent(object):
    def __init__(self):
        # from the origin base.agent
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.dqn = DeepQNetwork(
            len(smart_actions),
            18, # one of the most important data that needs to be update # 17 or 7
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True
        )

        self.previous_killed_unit_score = 0
        self.previous_lost_unit_score = 0

        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        # from the origin base.agent
        self.counter += 1
        self.steps += 1

        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance = self.extract_features(obs)

        killed_unit_score = obs.observation['score_cumulative'][5]
        remaining_unit_count = obs.observation['player'][8]

        if self.previous_action is not None:
            reward = self.get_reward(obs, distance, player_hp, enemy_hp)

            print(reward, self.counter)
            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.dqn.learn()

        rl_action = self.dqn.choose_action(np.array(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_enemy_hp = enemy_hp

        return self.perform_action(obs, smart_action, player_loc)

    def get_reward(self, obs, distance, player_hp, enemy_hp):
        reward = 0

        # give reward if dealing damage on enemy
        for i in range(0, len(enemy_hp)):
            if self.previous_enemy_hp[i] > enemy_hp[i]:
                reward += 1

        if distance[0] > 9:
            reward -= 1

        if distance[1] > 9:
            reward -= 1

        if distance[2] > 9:
            reward -= 1

        if distance[0] <= 9:
            reward += 20

        if distance[1] <= 9:
            reward += 20

        if distance[2] <= 9:
            reward += 20

        return reward


    # extract all the desired features as inputs for the DQN
    def extract_features(self, obs):
        var = obs.observation['feature_units']
        # get units' location and distance
        enemy = [] # size 2
        player = [] # size 3

        # get health
        enemy_hp = [] # size 2
        player_hp = [] # size 3

        enemy_unit_count = 0
        player_unit_count = 0

        for i in range(0, var.shape[0]):
            if var[i][_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                enemy.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                enemy_hp.append(var[i][_UNIT_HEALTH])
                enemy_unit_count += 1
            else:
                player.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                player_hp.append(var[i][_UNIT_HEALTH])
                player_unit_count += 1

        # append if necessary
        for i in range(player_unit_count, 3):
            player.append((-1, -1))
            player_hp.append(0)

        for i in range(enemy_unit_count, 2):
            enemy.append((-1, -1))
            enemy_hp.append(0)

        # get distance
        min_distance = [100000, 100000, 100000]

        for i in range(0, player_unit_count):
            for j in range(0, enemy_unit_count):
                distance = int(math.sqrt((player[i][0] - enemy[j][0]) ** 2 + (
                        player[i][1] - enemy[j][1]) ** 2))

                if distance < min_distance[i]:
                    min_distance[i] = distance

        # flatten the array so that all features are a 1D array
        feature1 = np.array(enemy_hp).flatten() # enemy's hp
        feature2 = np.array(player_hp).flatten() # player's hp
        feature3 = np.array(enemy).flatten() # enemy's coordinates
        feature4 = np.array(player).flatten() # player's coordinates
        feature5 = np.array(min_distance).flatten() # distance

        # combine all features horizontally
        current_state = np.hstack((feature1, feature2, feature3, feature4, feature5))

        return current_state, feature1, feature2, enemy, player, feature5

        # make the desired action calculated by DQN
    def perform_action(self, obs, action, unit_locs):
        unit_count = obs.observation['player'][8]

        if action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif action == ACTION_SELECT_ARMY:
           if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif action == ACTION_SELECT_UNIT_1:
            if _SELECT_POINT in obs.observation['available_actions']:
                if unit_count >= 1:
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[0]])

        elif action == ACTION_SELECT_UNIT_2:
            if _SELECT_POINT in obs.observation['available_actions']:
                if unit_count >= 2:
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[1]])

        elif action == ACTION_SELECT_UNIT_3:
            if _SELECT_POINT in obs.observation['available_actions']:
                if unit_count >= 3:
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[2]])

        elif action == ACTION_ATTACK_UP:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [41, 0]])  # x,y => col,row

        elif action == ACTION_ATTACK_DOWN:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [41, 83]])

        elif action == ACTION_ATTACK_LEFT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [0, 30]])

        elif action == ACTION_ATTACK_RIGHT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [83, 30]])

        ###
        elif action == ACTION_ATTACK_UP:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [41, 0]])  # x,y => col,row

        elif action == ACTION_ATTACK_DOWN:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [41, 83]])

        elif action == ACTION_ATTACK_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 36]])

        elif action == ACTION_ATTACK_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 36]])

        # elif action == ACTION_ATTACK_UP_LEFT:
        #     if _MOVE_SCREEN in obs.observation["available_actions"]:
        #         return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 0]])
        #
        # elif action == ACTION_ATTACK_UP_RIGHT:
        #     if _MOVE_SCREEN in obs.observation["available_actions"]:
        #         return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 0]])
        #
        # elif action == ACTION_ATTACK_DOWN_LEFT:
        #     if _MOVE_SCREEN in obs.observation["available_actions"]:
        #         return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 83]])
        #
        # elif action == ACTION_ATTACK_DOWN_RIGHT:
        #     if _MOVE_SCREEN in obs.observation["available_actions"]:
        #         return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 83]])

        return actions.FunctionCall(_NO_OP, [])

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        self.counter = 0
        self.reward = 0



