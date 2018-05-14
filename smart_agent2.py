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

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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

KILL_UNIT_REWARD = 300
LOSS_UNIT_REWARD = 0

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
            17, # one of the most important data that needs to be update # 17 or 7
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
        self.reward += obs.reward

        # time.sleep(0.2)
        current_state, hp, player_units, enemy_units, distance = self.extract_features(obs)

        killed_unit_score = obs.observation['score_cumulative'][5]
        remaining_unit_count = obs.observation['multi_select'].shape[0]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if remaining_unit_count < 3:
                reward += LOSS_UNIT_REWARD

            # discourage idle
            if ((hp[0] == 45 and hp[1] == 45 and hp[2] == 45 and hp[3] == 0) or
               (hp[0] == 0 and hp[1] == 45 and hp[2] == 45 and hp[3] == 0) or
               (hp[0] == 45 and hp[1] == 0 and hp[2] == 45 and hp[3] == 0) or
               (hp[0] == 45 and hp[1] == 45 and hp[2] == 0 and hp[3] == 0) or
                (hp[0] == 0 and hp[1] == 0 and hp[2] == 45 and hp[3] == 0) or
                (hp[0] == 0 and hp[1] == 45 and hp[2] == 0 and hp[3] == 0) or
                (hp[0] == 45 and hp[1] == 0 and hp[2] == 0 and hp[3] == 0) or
               (hp[0] == 0 and hp[1] == 0 and hp[2] == 0 and hp[3] == 45) or
                (hp[0] == 0 and hp[1] == 0 and hp[2] == 0 and hp[3] == 0)
            ):
                reward = reward - 0.1 * self.counter

            # encourage to survive
            if 45 > hp[0] > 0:
                reward += hp[0] / 3 * self.counter

            if 45 > hp[1] > 0:
                reward += hp[1] / 3 * self.counter

            if 45 > hp[2] > 0:
                reward += hp[2] / 3 * self.counter

            if 45 > hp[3] > 0:
                reward += hp[3] * self.counter

            print(hp, reward, self.counter)
            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.dqn.learn()


        rl_action = self.dqn.choose_action(np.array(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return self.perform_action(obs, smart_action, player_units)

    # extract all the desired features as inputs for the DQN
    def extract_features(self, obs):

        # initialize default value to prevent shrinking
        units = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
        hp = [0, 0, 0, 0]

        units_count = obs.observation['multi_select'].shape[0]
        # assign the value if exist
        for i in range(units_count):
            units[i] = (obs.observation['multi_select'][i])
            hp[i] = obs.observation['multi_select'][i][2]

        # single select unit's hp
        hp[3] = obs.observation['single_select'][0][2]

        # get units' location and distance
        player_units, enemy_units, distance = self.get_distance_and_locs(obs)

        # flatten the array so that all features are a 1D array
        hp = np.array(hp).flatten()
        feature1 = np.array(player_units).flatten()
        feature2 = np.array(enemy_units).flatten()
        distance = np.array(distance).flatten()

        # combine all features horizontally
        current_state = np.hstack((hp, distance, feature1, feature2))

        return current_state, hp, player_units, enemy_units, distance

    def get_distance_and_locs(self, obs):
        # calculate player unit location
        player_unit_count = obs.observation['player'][8]
        player_unit_y, player_unit_x = (obs.observation['screen'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        player_units = []

        if player_unit_count > 0:

            for i in range(0, player_unit_count):
                player_units.append((player_unit_x[i], player_unit_y[i]))

            kmeans = KMeans(n_clusters=player_unit_count)
            kmeans.fit(player_units)

            player_units = []
            for i in range(0, player_unit_count):
                player_units.append((int(kmeans.cluster_centers_[i][0]), int(kmeans.cluster_centers_[i][1])))

        for i in range(player_unit_count, 3):
            player_units.append((int(-1), int(-1)))

        # calculate opponents unit location
        opponent_unit_y, opponent_unit_x = (obs.observation['screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        opponent_unit_count = math.ceil(opponent_unit_y.shape[0] / 12)
        opponent_units = []

        if opponent_unit_count > 0:

            for i in range(0, opponent_unit_count):
                opponent_units.append((opponent_unit_x[i], opponent_unit_y[i]))

            kmeans = KMeans(n_clusters=opponent_unit_count)
            kmeans.fit(opponent_units)

            opponent_units = []
            for i in range(0, opponent_unit_count):
                opponent_units.append((int(kmeans.cluster_centers_[i][0]), int(kmeans.cluster_centers_[i][1])))

        for i in range(opponent_unit_count, 2):
           opponent_units.append((int(-1), int(-1)))

        # calculate the min distance between each of player's unit to the opponent's
        min_distance = [1000000, 1000000, 1000000]

        for i in range(0, player_unit_count):
            for j in range(0, opponent_unit_count):
                distance = int(math.sqrt((player_units[i][0] - opponent_units[j][0]) ** 2 + (
                            player_units[i][1] - opponent_units[j][1]) ** 2))

                if distance < min_distance[i]:
                    min_distance[i] = distance

        return player_units, opponent_units, min_distance

        # make the desired action calculated by DQN
    def perform_action(self, obs, action, unit_locs):
        unit_count = obs.observation['player'][8]

        if action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        # elif action == ACTION_SELECT_ARMY:
        #   if _SELECT_ARMY in obs.observation['available_actions']:
        #        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

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
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [0, 41]])

        elif action == ACTION_ATTACK_RIGHT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [0, 83]])

        elif action == ACTION_ATTACK_UP_LEFT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [0, 0]])

        elif action == ACTION_ATTACK_UP_RIGHT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [83, 0]])

        elif action == ACTION_ATTACK_DOWN_LEFT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [0, 83]])

        elif action == ACTION_ATTACK_DOWN_RIGHT:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [83, 83]])

        return actions.FunctionCall(_NO_OP, [])

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        self.counter = 0



