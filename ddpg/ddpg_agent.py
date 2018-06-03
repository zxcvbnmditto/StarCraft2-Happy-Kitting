import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from ddpg import DDPG
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_UNIT_ALLIANCE = 1
_UNIT_HEALTH = 2
_UNIT_SHIELD = 3
_UNIT_X = 12
_UNIT_Y = 13
_UNIT_IS_SELECTED = 17

_NOT_QUEUED = [0]
_QUEUED = [1]

ATTACK_TARGET = 'attacktarget'
MOVE_UP = 'moveup'
MOVE_DOWN = 'movedown'
MOVE_LEFT = 'moveleft'
MOVE_RIGHT = 'moveright'
ACTION_SELECT_UNIT = 'selectunit'

smart_actions = [
    ATTACK_TARGET,
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    ACTION_SELECT_UNIT
]

# Change this if using a different map
# Currently running HK2V1
DEFAULT_ENEMY_COUNT = 1
DEFAULT_PLAYER_COUNT = 2

ENEMY_MAX_HP = 150
PLAYER_MAX_HP = 60


class SmartAgent(object):
    def __init__(self):
        # from the origin base.agent
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.ddpg = DDPG(
            a_dim=len(smart_actions),
            s_dim=11, # one of the most important data that needs to be update manually
        )

        # self defined vars
        self.fighting = False
        self.win = 0
        self.player_hp = []
        self.enemy_hp = []
        self.previous_enemy_hp = []
        self.previous_player_hp = []

        self.previous_action = None
        self.previous_state = None

    def step(self, obs):

        # from the origin base.agent
        self.steps += 1
        self.reward += obs.reward

        self.counter += 1
        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = self.extract_features(obs)

        self.player_hp.append(sum(player_hp))
        self.enemy_hp.append(sum(enemy_hp))

        # scripted the few initial actions to increases the learning performance
        while not self.fighting:
            for i in range(0, player_count):
                if distance[i] < 20:
                    self.fighting = True
                    return actions.FunctionCall(_NO_OP, [])

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, enemy_loc[0]])

        # record the transitions to memory and learn by DQN
        if self.previous_action is not None:
            reward = self.get_reward(obs, distance, player_hp, enemy_hp, player_count, enemy_count)

            self.ddpg.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.ddpg.learn()

        # get the disabled actions and used it when choosing actions
        disabled_actions = self.get_disabled_actions(player_loc, selected)
        #disabled_actions = []
        rl_action = self.ddpg.choose_action(np.array(current_state), disabled_actions)
        smart_action = smart_actions[rl_action]

        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_enemy_hp = enemy_hp
        self.previous_player_hp = player_hp

        next_action = self.perform_action(obs, smart_action, player_loc, enemy_loc, selected, player_count, enemy_count, distance,
                            player_hp)

        return next_action

    def get_reward(self, obs, distance, player_hp, enemy_hp, player_count, enemy_count):
        reward = 0.

        # give reward by calculating opponents units lost hp
        for i in range(0, DEFAULT_ENEMY_COUNT):
            reward += ((ENEMY_MAX_HP - enemy_hp[i]) * 2)

        # give reward by remaining player units hp
        for i in range(0, DEFAULT_PLAYER_COUNT):
            reward += (player_hp[i])
            reward -= (distance[i] - 10) ** 2

        if reward < 0:
            reward = 0

        # get killed and lost unit reward from the map
        reward = int(reward)

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
                enemy_hp.append(var[i][_UNIT_HEALTH] +  + var[i][_UNIT_SHIELD])
                enemy_unit_count += 1
            else:
                player.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                player_hp.append(var[i][_UNIT_HEALTH])
                is_selected.append(var[i][_UNIT_IS_SELECTED])
                player_unit_count += 1

        # append if necessary so that maintains fixed length for current state
        for i in range(player_unit_count, DEFAULT_PLAYER_COUNT):
            player.append((-1, -1))
            player_hp.append(0)
            is_selected.append(-1)

        for i in range(enemy_unit_count, DEFAULT_ENEMY_COUNT):
            enemy.append((-1, -1))
            enemy_hp.append(0)

        # get distance
        min_distance = [100000 for x in range(DEFAULT_PLAYER_COUNT)]

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

        return current_state, enemy_hp, player_hp, enemy, player, min_distance, is_selected, enemy_unit_count, player_unit_count

    # make the desired action calculated by DQNLR_C
    def perform_action(self, obs, action, unit_locs, enemy_locs, selected, player_count, enemy_count, distance, player_hp):
        index = -1

        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                index = i

        x = unit_locs[index][0]
        y = unit_locs[index][1]

        if action == ATTACK_TARGET:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                if enemy_count >= 1:
                    return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_locs[0]])  # x,y => col,row

        elif action == MOVE_UP:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x
                y = y - 8

                if 3 > x:
                    x = 3
                elif x > 79:
                    x = 79

                if 3 > y:
                    y = 3
                elif y > 59:
                    y = 59

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])  # x,y => col,row

        elif action == MOVE_DOWN:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x
                y = y + 8

                if 3 > x:
                    x = 3
                elif x > 79:
                    x = 79

                if 3 > y:
                    y = 3
                elif y > 59:
                    y = 59

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        elif action == MOVE_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x - 8
                y = y

                if 3 > x:
                    x = 3
                elif x > 79:
                    x = 79

                if 3 > y:
                    y = 3
                elif y > 59:
                    y = 59

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        elif action == MOVE_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"] and index != -1:
                x = x + 8
                y = y

                if 3 > x:
                    x = 3
                elif x > 79:
                    x = 79

                if 3 > y:
                    y = 3
                elif y > 59:
                    y = 59

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

        # Default case => Select unit
        # select the unit that is closest to the enemy
        # if same distance, pick the one with lower hp
        # if same distance and hp, randomly select one
        closest_indices = []
        closest_index = distance.index(min(distance))

        for i in range(0, player_count):
            if distance[i] == distance[closest_index]:
                closest_indices.append(i)

        lowest_hp_indices = []
        lowest_hp_index = player_hp.index(min(player_hp))

        for i in range(0, player_count):
            if player_hp[i] == player_hp[lowest_hp_index]:
                lowest_hp_indices.append(i)

        common_indices = list(set(closest_indices).intersection(lowest_hp_indices))

        if len(common_indices) != 0:
            selected_index = random.choice(common_indices)
        elif len(closest_indices) != 0:
            selected_index = random.choice(closest_indices)
        else:
            selected_index = 0

        self.previous_action = 5
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[selected_index]])

    # get_disabled_actions filters the redundant actions from the action space
    def get_disabled_actions(self, player_loc, selected):
        disabled_actions = []

        index = -1

        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                index = i
                break

        x = player_loc[index][0]
        y = player_loc[index][1]

        # not selecting attack target if the previous actions is already attack target
        if self.previous_action == smart_actions.index(ATTACK_TARGET):
            disabled_actions.append(smart_actions.index(ATTACK_TARGET)) #0

        # not selecting a specific move action if the unit cannot move toward that direction (at the border)
        if y <= 7:
            disabled_actions.append(smart_actions.index(MOVE_UP)) #1

        if y >= 56:
            disabled_actions.append(smart_actions.index(MOVE_DOWN)) #2

        if x <= 7:
            disabled_actions.append(smart_actions.index(MOVE_LEFT)) #3

        if x >= 76:
            disabled_actions.append(smart_actions.index(MOVE_RIGHT)) #4

        # not selecting the same unit if the previous actions already attempts to select it
        if self.previous_action == smart_actions.index(ACTION_SELECT_UNIT):
            disabled_actions.append(smart_actions.index(ACTION_SELECT_UNIT)) #5

        return disabled_actions

    def plot_player_hp(self, path, save):
        plt.plot(np.arange(len(self.player_hp)), self.player_hp)
        plt.ylabel('player hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/player_hp.png')
        plt.show()

    def plot_enemy_hp(self, path, save):
        plt.plot(np.arange(len(self.enemy_hp)), self.enemy_hp)
        plt.ylabel('enemy hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/enemy_hp.png')
        plt.show()

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        # added instead of original
        self.fighting = False
        self.counter = 0



