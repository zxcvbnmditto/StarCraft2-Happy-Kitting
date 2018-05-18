import numpy as np
import math
import time
from algorithms.dqn import DeepQNetwork

from pysc2.lib import actions
from pysc2.lib import features

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
_UNIT_IS_SELECTED = 17

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
MOVE_UP = 'moveup'
MOVE_DOWN = 'movedown'
MOVE_LEFT = 'moveleft'
MOVE_RIGHT = 'moveright'
MOVE_UP_LEFT = 'moveupleft'
MOVE_DOWN_LEFT = 'movedownleft'
MOVE_UP_RIGHT = 'moveupright'
MOVE_DOWN_RIGHT = 'movedownright'
ACTION_SELECT_UNIT_1 = 'selectunit1'
ACTION_SELECT_UNIT_2 = 'selectunit2'
ATTACK_TARGET = 'attacktarget'

smart_actions = [
    ATTACK_TARGET,
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    ACTION_SELECT_UNIT_1,
    ACTION_SELECT_UNIT_2
]

DEFAULT_ENEMY_COUNT = 1
DEFAULT_PLAYER_COUNT = 2

KILL_UNIT_REWARD = 5
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
            11, # one of the most important data that needs to be update # 17 or 7
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=5000,
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

        # time.sleep(0.1)
        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = self.extract_features(obs)

        if self.counter == 1:
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, player_loc[0]])

        if self.previous_action is not None:
            reward = self.get_reward(obs, distance, player_hp, enemy_hp, player_count, enemy_count)

            # print(reward, self.counter)
            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.dqn.learn()

        rl_action = self.dqn.choose_action(np.array(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_enemy_hp = enemy_hp

        return self.perform_action(obs, smart_action, player_loc, enemy_loc, selected)

    def get_reward(self, obs, distance, player_hp, enemy_hp, player_count, enemy_count):
        reward = 0

        # give reward if dealing damage on enemy
        for i in range(0, enemy_count):
            if self.previous_enemy_hp[i] > enemy_hp[i]:
                reward += 1

        # reward increases if kills opponent's army
        kill_army_count = DEFAULT_ENEMY_COUNT - enemy_count
        reward += kill_army_count * KILL_UNIT_REWARD

        # reward decreases if loses army
        lost_army_count = DEFAULT_PLAYER_COUNT - player_count
        reward += lost_army_count * LOSS_UNIT_REWARD

        if distance[0] > 16:
            reward -= 1
        elif 16 >= distance[0] >= 9:
            reward += 2
        else:
            reward -= 1

        if distance[1] > 16:
            reward -= 1
        elif 16 >= distance[1] >= 9:
            reward += 2
        else:
            reward -= 1

        return reward

    # extract all the desired features as inputs for the DQN
    def extract_features(self, obs):
        var = obs.observation['feature_units']
        # get units' location and distance
        enemy, player = [], []

        # get health
        enemy_hp, player_hp = [], []

        # record the selected army
        is_selected = []

        # unit_count
        enemy_unit_count, player_unit_count = 0, 0

        for i in range(0, var.shape[0]):
            if var[i][_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                enemy.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                enemy_hp.append(var[i][_UNIT_HEALTH])
                enemy_unit_count += 1
            else:
                player.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                player_hp.append(var[i][_UNIT_HEALTH])
                is_selected.append(var[i][_UNIT_IS_SELECTED])
                player_unit_count += 1

        # append if necessary
        for i in range(player_unit_count, DEFAULT_PLAYER_COUNT):
            player.append((-1, -1))
            player_hp.append(0)
            is_selected.append(-1)

        for i in range(enemy_unit_count, DEFAULT_ENEMY_COUNT):
            enemy.append((-1, -1))
            enemy_hp.append(0)

        # get distance
        min_distance = [100000, 100000]

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

        return current_state, feature1, feature2, enemy, player, min_distance, is_selected, enemy_unit_count, player_unit_count

        # make the desired action calculated by DQN
    def perform_action(self, obs, action, unit_locs, enemy_locs, selected):
        unit_count = obs.observation['player'][8]

        index = -1

        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                index = i

        x = unit_locs[index][0]
        y = unit_locs[index][1]

        if action == ACTION_SELECT_UNIT_1:
            if _SELECT_POINT in obs.observation['available_actions']:
                if unit_count >= 1:
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[0]])

        elif action == ACTION_SELECT_UNIT_2:
            if _SELECT_POINT in obs.observation['available_actions']:
                if unit_count >= 2:
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[1]])

        #-----------------------
        elif action == ATTACK_TARGET:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_locs[0]])  # x,y => col,row
        # ------------------------

        elif action == MOVE_UP:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x
                y = y - 5

                if 0 > x:
                    x = 0
                elif x > 83:
                    x = 83

                if 0 > y:
                    y = 0
                elif y > 83:
                    y = 83

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])  # x,y => col,row

        elif action == MOVE_DOWN:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x
                y = y + 5

                if 0 > x:
                    x = 0
                elif x > 83:
                    x = 83

                if 0 > y:
                    y = 0
                elif y > 83:
                    y = 83

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        elif action == MOVE_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x - 5
                y = y

                if 0 > x:
                    x = 0
                elif x > 83:
                    x = 83

                if 0 > y:
                    y = 0
                elif y > 83:
                    y = 83

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        elif action == MOVE_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x + 5
                y = y

                if 0 > x:
                    x = 0
                elif x > 83:
                    x = 83

                if 0 > y:
                    y = 0
                elif y > 83:
                    y = 83

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        self.previous_action = 5
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[0]])

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        self.counter = 0
        self.reward = 0



