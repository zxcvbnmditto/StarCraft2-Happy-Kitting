import matplotlib.pyplot as plt
import numpy as np
import math
import random
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

        # self defined vars
        self.player_hp = []
        self.enemy_hp = []
        self.my_reward = []
        self.fighting = False


    def step(self, obs):
        # from the origin base.agent
        self.steps += 1
        self.reward += obs.reward

        # time.sleep(0.1)
        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = self.extract_features(obs)
        reward = self.get_reward(obs, distance, player_hp, enemy_hp, player_count, enemy_count)

        self.player_hp.append(sum(player_hp))
        self.enemy_hp.append(sum(enemy_hp))
        self.my_reward.append(reward)

        return current_state, reward, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count

    def get_reward(self, obs, distance, player_hp, enemy_hp, player_count, enemy_count):
        reward = 0.

        # give reward by calculating opponents units lost hp
        for i in range(0, DEFAULT_ENEMY_COUNT):
            reward += ((ENEMY_MAX_HP - enemy_hp[i]) * 2)
            if enemy_hp[i] <= 5:
                reward += 180

        # give reward by remaining player units hp
        for i in range(0, DEFAULT_PLAYER_COUNT):
            reward += (player_hp[i])

            if player_hp[i] > 0:
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
                enemy_hp.append(var[i][_UNIT_HEALTH] + var[i][_UNIT_SHIELD ])
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

    # make the desired action calculated by DQN
    def perform_action(self, obs, action, unit_locs, enemy_locs, selected, player_count, enemy_count, distance, player_hp):
        index = -1

        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                index = i

        x = unit_locs[index][0]
        y = unit_locs[index][1]

        if action == 0:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                if enemy_count >= 1:
                    return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_locs[0]])  # x,y => col,row

        elif action == 1:
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

        elif action == 2:
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

        elif action == 3:
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

        elif action == 4:
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

        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, unit_locs[selected_index]])


    def plot_player_hp(self, path, save):
        plt.plot(np.arange(len(self.player_hp)), self.player_hp, 'b-')
        plt.ylabel('player hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/player_hp.png')
        plt.close()



    def plot_enemy_hp(self, path, save):
        plt.plot(np.arange(len(self.enemy_hp)), self.enemy_hp, 'r-')
        plt.ylabel('enemy hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/enemy_hp.png')
        plt.close()


    def plot_reward(self, path, save):
        plt.plot(np.arange(len(self.my_reward)), self.my_reward, '#33FF90')
        plt.ylabel('reward')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/reward.png')
        plt.close()



    def plot_all(self, path, save):
        plt.plot(np.arange(len(self.player_hp)), self.player_hp, 'b-', label="player_hp")
        plt.plot(np.arange(len(self.enemy_hp)), self.enemy_hp, 'r-', label="enemy_hp")
        plt.ylabel('hp')
        plt.xlabel('training steps')

        if save:
            plt.savefig(path + '/all.png')
        plt.close()

        self.my_reward = []
        self.enemy_hp = []
        self.player_hp = []

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        # added instead of original
        self.fighting = False





