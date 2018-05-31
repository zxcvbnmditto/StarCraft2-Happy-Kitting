#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading
import time

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import actions

import neat
import visualize
import numpy as np
import multiprocessing as mp

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Change the following 3 flags to run the program successfully
# agent, agent_file, map

# modify agent name here: "agent", "YourAgentFileName.YourAgentClassName", "Description"
flags.DEFINE_string("agent", "agent.SmartAgent",
                    "Which agent to run")

# modify executing file name here
flags.DEFINE_string("agent_file", "agent",
                    "Which file to run")

# edit map used here
flags.DEFINE_string("map", 'HK2V1', "Name of a map to use.")

# -----------------------------------------------------------------------------------------------
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

# edit steps limit to control training episodes.
flags.DEFINE_integer("max_agent_steps", 500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 2, "Game steps per agent step.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.mark_flag_as_required("map")

### currently doing
### 10 generations, 10 genomes, 10 episode each genome, 500 step(max) for each episode
### estimated step count = 10*10*10*500 = 500000

CONFIG = "./config"
EP_STEP = 400  # maximum episode steps
GENERATION_EP = 5  # evaluate each genome by average 10-episode rewards
GENERATION = 15
TRAINING = True  # training or testing
CONTINUE_TRAINING = True  # Train from scratch or from previous checkpoints
CHECKPOINT = 16  # test on this checkpoint


# -----------------------------------------------------------------------------------------------
def run_thread(agent_cls, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            feature_screen_size=FLAGS.screen_resolution,
            feature_minimap_size=FLAGS.minimap_resolution,
            visualize=visualize,
            use_feature_units=True
    ) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_cls()

        agent_name = FLAGS.agent_file

        # set the path to save the models and graphs
        # path = 'models/' + agent_name

        if TRAINING:
            run_loop([agent], env, CONTINUE_TRAINING)
        else:
            evaluation([agent], env)

        if FLAGS.save_replay:
            env.save_replay(agent_cls.__name__)


def run_loop(agents, env, continue_training):
    """A run loop to have agents and an environment interact."""
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    # restore from the model or not
    if continue_training:
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    else:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
        pop = neat.Population(config)

    # recode history
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(1))  # num represents the saving interval for generation

    def eval_genomes(genomes, config):
        # total estimated count 10 * 10 * 10 * 500
        global generation
        generation = generation + 1
        count = 0
        for genome_id, genome in genomes:

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            ep_r = []
            # loop count pop in each gen
            for ep in range(GENERATION_EP):  # run many episodes for the genome in case it's lucky
                print("Generation ", generation, "Genome_id ", genome_id, " episode ", ep, " count ", count)
                accumulative_r = 0.  # stage longer to get a greater episode reward
                timesteps = env.reset()

                for a in agents:
                    a.reset()

                for step in range(EP_STEP):
                    count = count + 1
                    # it finally works
                    current_state, reward, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = \
                    agents[0].step(timesteps[0])

                    if not agents[0].fighting:
                        for i in range(0, player_count):
                            if distance[i] < 20:
                                agents[0].fighting = True
                                action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                                break

                            action = actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], enemy_loc[0]])
                    else:
                        action_values = net.activate(current_state)
                        # action_index = np.argmax(action_values)

                        action_values = np.random.choice(a=range(6), p=softmax(action_values))

                        action = agents[0].perform_action(timesteps[0], action_values, player_loc, enemy_loc, selected,
                                                          player_count, enemy_count, distance, player_hp)
                        print(reward, accumulative_r)
                        accumulative_r += reward

                    if timesteps[0].last():
                        break
                    timesteps = env.step([action])

                ep_r.append(accumulative_r)

            genome.fitness = np.max(ep_r) / float(EP_STEP)  # depends on the minimum episode reward

    global generation
    generation = 0
    # call and run the NEAT algorithm
    pop.run(eval_genomes, GENERATION)  # train 10 generations

    # visualize training
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def evaluation(agents, env):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)

    def eval_genomes(genomes, config):
        # total estimated count 10 * 10 * 10 * 500
        count = 0
        for genome_id, genome in genomes:

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            ep_r = []
            # loop count pop in each gen
            for ep in range(GENERATION_EP):  # run many episodes for the genome in case it's lucky
                print("Genome_id ", genome_id, " episode ", ep, " count ", count)
                accumulative_r = 0.  # stage longer to get a greater episode reward
                timesteps = env.reset()

                for a in agents:
                    a.reset()

                for step in range(EP_STEP):
                    count = count + 1
                    # it finally works
                    current_state, reward, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = \
                    agents[0].step(timesteps[0])

                    action_values = net.activate(current_state)
                    action_index = np.argmax(action_values)
                    action = agents[0].perform_action(timesteps[0], action_index, player_loc, enemy_loc, selected,
                                                      player_count, enemy_count, distance, player_hp)
                    accumulative_r += reward
                    if timesteps[0].last():
                        break
                    timesteps = env.step([action])

                ep_r.append(accumulative_r)

            genome.fitness = np.max(ep_r) / float(EP_STEP)  # depends on the minimum episode reward

    winner = p.run(eval_genomes, 1)  # find the winner in restored population

    # show winner net
    node_names = {-1: 'enemy_hp', -2: 'player_hp_1', -3: 'player_hp_2', -4: 'enemy_x', -5: 'enemy_y', -6: 'player_x_1',
                  -7: 'player_y_1', -8: 'player_x_2', -9: 'player_y_2', -10: 'distance_1', -11: 'distance_2',
                  0: 'attack', 1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'select'}
    visualize.draw_net(p.config, winner, True, node_names=node_names)


def _main(unused_argv):
    """Run an agent."""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    # Map Name
    mapName = FLAGS.map

    globals()[mapName] = type(mapName, (maps.mini_games.MiniGame,), dict(filename=mapName))

    maps.get(FLAGS.map)  # Assert the map exists.

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread, args=(agent_cls, FLAGS.map, False))
        threads.append(t)
        t.start()

    run_thread(agent_cls, FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(_main)


if __name__ == "__main__":
    app.run(_main)
