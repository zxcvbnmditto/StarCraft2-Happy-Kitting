import numpy as np
import tensorflow as tf
from algorithms.dqn import DeepQNetwork

from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
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
            25, # one of the most important data that needs to be update
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
        self.steps += 1
        self.reward += obs.reward

        # time.sleep(0.2)
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        test_y, test_x = (obs.observation['screen'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        killed_unit_score = obs.observation['score_cumulative'][5]
        remaining_unit_score = obs.observation['multi_select'].shape[0]

        current_state, hp = self.extract_features(obs)

        print(self.reward, self.episodes, self.steps)

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if remaining_unit_score < 3:
                reward += LOSS_UNIT_REWARD

            # discourage idle
            if hp[0] is 45 and hp[1] is 45 and hp[2] is 45:
                reward -= 1

            # encourage to survive
            if 45 > hp[0] > 0:
                reward += 0.1

            if 45 > hp[1] > 0:
                reward += 0.1

            if 45 > hp[2] > 0:
                reward += 0.1

            self.dqn.store_transition(np.array(self.previous_state), self.previous_action, reward, np.array(current_state))
            self.dqn.learn()

        rl_action = self.dqn.choose_action(np.array(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return self.perform_action(obs, smart_action, player_x, player_y)

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

        # flatten the array so that all features are a 1D array
        units = np.array(units).flatten()
        hp = np.array(hp).flatten()

        # combine all features horizontally
        current_state = np.hstack((units, hp))

        return current_state, hp

    # make the desired action calculated by DQN
    def perform_action(self, obs, action, xloc, yloc):
        if action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif action == ACTION_SELECT_ARMY:
           if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif action == ACTION_SELECT_UNIT_1:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 1 and len(yloc) >= 1:
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [0]])

        elif action == ACTION_SELECT_UNIT_2:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 2 and len(yloc) >= 2:
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [1]])

        elif action == ACTION_SELECT_UNIT_3:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 3 and len(yloc) >= 3:
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [2]])

        elif action == ACTION_ATTACK_UP:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [41, 0]])# x,y => col,row

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


