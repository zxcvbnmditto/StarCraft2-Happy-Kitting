import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from dqn import DeepQNetwork
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
            10, # one of the most important data that needs to be update manually
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True
        )

        # self defined vars
        self.fighting = False
        self.player_hp = []
        self.enemy_hp = []
        self.previous_enemy_hp = []
        self.previous_player_hp = []
        self.leftover_enemy_hp = []
        self.count = 0

        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        # from the origin base.agent
        self.steps += 1
        self.reward += obs.reward

        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count, player_cooldown = self.extract_features(obs)

        self.player_hp.append(sum(player_hp))
        self.enemy_hp.append(sum(enemy_hp))

        # scripted the few initial actions to increases the learning performance
        while not self.fighting:
            for i in range(0, player_count):
                if distance[i] < 20:
                    self.fighting = True
                    # return actions.FunctionCall(_NO_OP, [])

            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_loc[0]])

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

        if selected[selected_index] == 0 or (selected[0] == 1 and selected[1] == 1):
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, player_loc[selected_index]])

        rl_action = self.dqn.choose_action(np.array(current_state))
        smart_action = smart_actions[rl_action]

        # record the transitions to memory and learn by DQN
        if self.previous_action is not None:
            reward = self.get_reward(obs, distance, player_hp, enemy_hp, player_count, enemy_count, rl_action, selected, player_loc,enemy_loc, player_cooldown)

            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))

        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_enemy_hp = enemy_hp
        self.previous_player_hp = player_hp

        next_action = self.perform_action(obs, smart_action, player_loc, enemy_loc, selected, player_count, enemy_count, distance,
                            player_hp)

        return next_action

    def get_reward(self, obs, distance, player_hp, enemy_hp, player_count, enemy_count, rl_action, selected, unit_locs, enemy_loc, player_cooldown):
        reward = 0.
        selected_index = -1

        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                selected_index = i

        x = unit_locs[selected_index][0]
        y = unit_locs[selected_index][1]

        if x <= 7 and rl_action == smart_actions.index(MOVE_LEFT):
            #print("left error")
            return -1

        if x >= 76 and rl_action == smart_actions.index(MOVE_RIGHT):
            #print("right error")
            return -1

        if y <= 7 and rl_action == smart_actions.index(MOVE_UP):
            #print("up error")
            return -1

        if y >= 56 and rl_action == smart_actions.index(MOVE_DOWN):
            #print("down error")
            return -1

        if player_cooldown[selected_index] != 0 and rl_action == smart_actions.index(ATTACK_TARGET):
            #print("cooldown error")
            return -1

        predicted_distance = []
        # up, down, left, right

        # up
        x_predict = x
        y_predict = y - 4

        if 3 > x_predict:
            x_predict = 3
        elif x_predict > 79:
            x_predict = 79

        if 3 > y_predict:
            y_predict = 3
        elif y_predict > 59:
            y_predict = 59
        predicted_distance.append(int(math.sqrt((x_predict - enemy_loc[0][0]) ** 2 + (
                y_predict - enemy_loc[0][1]) ** 2)))

        # down
        x_predict = x
        y_predict = y + 4

        if 3 > x_predict:
            x_predict = 3
        elif x_predict > 79:
            x_predict = 79

        if 3 > y_predict:
            y_predict = 3
        elif y_predict > 59:
            y_predict = 59
        predicted_distance.append(int(math.sqrt((x_predict - enemy_loc[0][0]) ** 2 + (
                y_predict - enemy_loc[0][1]) ** 2)))

        # left
        x_predict = x - 4
        y_predict = y

        if 3 > x_predict:
            x_predict = 3
        elif x_predict > 79:
            x_predict = 79

        if 3 > y_predict:
            y_predict = 3
        elif y_predict > 59:
            y_predict = 59
        predicted_distance.append(int(math.sqrt((x_predict - enemy_loc[0][0]) ** 2 + (
                y_predict - enemy_loc[0][1]) ** 2)))

        # right
        x_predict = x + 4
        y_predict = y

        if 3 > x_predict:
            x_predict = 3
        elif x_predict > 79:
            x_predict = 79

        if 3 > y_predict:
            y_predict = 3
        elif y_predict > 59:
            y_predict = 59
        predicted_distance.append(int(math.sqrt((x_predict - enemy_loc[0][0]) ** 2 + (
                y_predict - enemy_loc[0][1]) ** 2)))

        if rl_action == smart_actions.index(MOVE_UP) and distance[selected_index] > predicted_distance[0]:
            return -1

        if rl_action == smart_actions.index(MOVE_DOWN) and distance[selected_index] > predicted_distance[1]:
            return -1

        if rl_action == smart_actions.index(MOVE_LEFT) and distance[selected_index] > predicted_distance[2]:
            return -1

        if rl_action == smart_actions.index(MOVE_RIGHT) and distance[selected_index] > predicted_distance[3]:
            return -1

        if distance[selected_index] > 20:
            return -0.5
        elif distance[selected_index] < 9:
            # up left
            if x<=6 and y<=6:
                print("up left")
                if rl_action == smart_actions.index(MOVE_DOWN) or rl_action == smart_actions.index(MOVE_RIGHT):
                    reward += 0.3
                else:
                    return -1
            # up right
            if x>=78 and y<=6:
                print("up right")
                if rl_action == smart_actions.index(MOVE_DOWN) or rl_action == smart_actions.index(MOVE_LEFT):
                    reward += 0.3
                else:
                    return -1

            # down left
            if x<=6 and y>=58:
                print("down left")
                if rl_action == smart_actions.index(MOVE_UP) or rl_action == smart_actions.index(MOVE_RIGHT):
                    reward += 0.3
                else:
                    return -1

            # down right
            if x>=78 and y>=58:
                print("down right")
                if rl_action == smart_actions.index(MOVE_UP) or rl_action == smart_actions.index(MOVE_LEFT):
                    reward += 0.3
                else:
                    return -1

            return -1
        else:
            if rl_action == 0 and self.previous_action == 1:
                reward += -0.5

            if rl_action == 1 and self.previous_action == 0:
                reward += -0.5

            if rl_action == 3 and self.previous_action == 2:
                reward += -0.5

            if rl_action == 2 and self.previous_action == 3:
                reward += -0.5

            best_action = predicted_distance.index(max(predicted_distance))

            if distance[selected_index] > 15 and player_cooldown[selected_index] == 0:
                best_action = smart_actions.index(ATTACK_TARGET)

            if rl_action == best_action:
                reward += 0.5

        reward += (ENEMY_MAX_HP - sum(enemy_hp)) / ENEMY_MAX_HP * 0.5

        return reward

    # extract all the desired features as inputs for the DQN
    def extract_features(self, obs):
        var = obs.observation['feature_units']

        # get units' location and distance
        enemy, player = [], []

        # get health
        enemy_hp, player_hp, player_cooldown = [], [], []

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
                player_cooldown.append((var[i][_UNIT_COOLDOWN]))
                player_unit_count += 1

        # append if necessary so that maintains fixed length for current state
        for i in range(player_unit_count, DEFAULT_PLAYER_COUNT):
            player.append((-1, -1))
            player_hp.append(0)
            player_cooldown.append(0)
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

        # some new stuff to try
        player_units, enemy_units = [], []

        for i in range(0, var.shape[0]):
            if var[i][_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                unit = []
                unit.append(var[i][_UNIT_X])
                unit.append(var[i][_UNIT_Y])
                unit.append(var[i][_UNIT_HEALTH] + var[i][_UNIT_SHIELD])
                unit.append(var[i][_UNIT_COOLDOWN])

                enemy_units.append(unit)
            else:
                unit = []
                unit.append(var[i][_UNIT_X])
                unit.append(var[i][_UNIT_Y])
                unit.append(var[i][_UNIT_HEALTH])
                unit.append(var[i][_UNIT_COOLDOWN])
                unit.append(100000)  # default distance
                unit.append(var[i][_UNIT_IS_SELECTED])

                if var[i][_UNIT_IS_SELECTED] == 1:
                    player_units.append(unit)

                    if var[i][_UNIT_HEALTH] < 20:
                        self.count += 1

        # append if necessary so that maintains fixed length for current state
        for i in range(player_unit_count, 1):
            unit = [-1, -1, 0, 0, 100000, 0]
            player_units.append(unit)

        for i in range(enemy_unit_count, DEFAULT_ENEMY_COUNT):
            unit = [-1, -1, 0, 0]
            enemy_units.append(unit)

        for unit in player_units:
            for opponent in enemy_units:
                distance = int(math.sqrt((unit[0] - opponent[0]) ** 2 + (unit[1] - opponent[1]) ** 2))

                if distance < unit[4]:
                    unit[4] = distance

        # flatten the array so that all features are a 1D array
        feature1 = np.array(enemy_hp).flatten() # enemy's hp
        feature2 = np.array(player_hp).flatten() # player's hp
        feature3 = np.array(enemy).flatten() # enemy's coordinates
        feature4 = np.array(player).flatten() # player's coordinates
        feature5 = np.array(min_distance).flatten() # distance
        feature6 = np.array(player_cooldown).flatten()

        feature7 = np.array(player_units).flatten()
        feature8 = np.array(enemy_units).flatten()

        # combine all features horizontally
        #current_state = np.hstack((feature1, feature2, feature3, feature4, feature5, feature6))
        current_state = np.hstack((feature7, feature8))

        return current_state, enemy_hp, player_hp, enemy, player, min_distance, is_selected, enemy_unit_count, player_unit_count, player_cooldown

    # make the desired action calculated by DQN
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
                y = y - 4

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
                y = y + 4

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
                x = x - 4
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
                x = x + 4
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

        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])

    def plot_hp(self, path, save):
        plt.plot(np.arange(len(self.player_hp)), self.player_hp)
        plt.ylabel('player hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/player_hp.png')
        plt.close()

        plt.plot(np.arange(len(self.enemy_hp)), self.enemy_hp)
        plt.ylabel('enemy hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/enemy_hp.png')
        plt.close()

        plt.plot(np.arange(len(self.leftover_enemy_hp)), self.leftover_enemy_hp)
        plt.ylabel('enemy hp')
        plt.xlabel('Episodes')
        if save:
            plt.savefig(path + '/eval.png')
        plt.close()

        print("AVG ENEMY HP LEFT", sum(self.leftover_enemy_hp) / len(self.leftover_enemy_hp))
        print("Low hp controlled steps", self.count)


    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        # added instead of original
        self.fighting = False
        if self.episodes > 1:
            self.leftover_enemy_hp.append(sum(self.previous_enemy_hp))
            self.dqn.learn()




