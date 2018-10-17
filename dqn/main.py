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

from pysc2 import maps
from pysc2.env import available_actions_printer, sc2_env
from pysc2.lib import stopwatch
from dqn_agent import SmartAgent as MyAgentClass
from absl import app, flags
import os
import time

FLAGS = flags.FLAGS
# specifies the maximum step count for this run

flags.DEFINE_integer("max_steps", 500, "Total agent steps.")
flags.DEFINE_integer("max_episodes", 10, "Total agent steps.")
flags.DEFINE_integer("episode_per_checkpoint", 2, "Total agent steps.")

flags.DEFINE_string("map", 'HK2V1', "Name of a map to use.")

flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 2, "Game steps per agent step.")
flags.DEFINE_bool("visualize", True, "Whether to render with pygame.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("thread_count", 5, "How many instances to run in parallel.")

flags.DEFINE_float('gpu_fraction', 0.0, 'GPU fraction, 0.0 => allow growth')
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_string("logs_dir", '/media/james/D/starcraft2/HK2V1/dqn/logs', "Directory to save tensorboard")
flags.DEFINE_string("models_dir", '/media/james/D/starcraft2/HK2V1/dqn/models', "Directory to save models")
flags.DEFINE_string("graphs_dir", '/media/james/D/starcraft2/HK2V1/dqn/graphs', "Directory to save graphs")

flags.mark_flag_as_required("map")
# -----------------------------------------------------------------------------------------------

def main(unused_argv):
    print("Main----------------------")
    """Run an agent."""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    # Create Directory for storage
    if not os.path.isdir(FLAGS.models_dir):
        os.makedirs(FLAGS.models_dir)

    if not os.path.isdir(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    if not os.path.isdir(FLAGS.graphs_dir):
        os.makedirs(FLAGS.graphs_dir)

    # Assert the map exists
    globals()[FLAGS.map] = type(FLAGS.map, (maps.mini_games.MiniGame,), dict(filename=FLAGS.map))
    maps.get(FLAGS.map)

    # Define agent
    agent = MyAgentClass(FLAGS)
    # restore the model
    agent.dqn.load_model(FLAGS.models_dir)

    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=FLAGS.screen_resolution,
                    minimap=FLAGS.minimap_resolution,
                ),
                use_feature_units=True
            ),
            visualize=FLAGS.visualize,
    ) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        agent.setup(observation_spec, action_spec)

        for episode in range(FLAGS.max_episodes):
            start_time = time.time()
            step = -1

            # Don't modify timesteps
            timesteps = env.reset()
            agent.reset()

            try:
                while step < FLAGS.max_steps:
                    step += 1
                    actions = [agent.step(timestep) for timestep in timesteps]

                    if timesteps[0].last():
                        break

                    timesteps = env.step(actions)
            except KeyboardInterrupt:
                pass
            finally:
                print("Episode: {} Step {} Time {}".format(episode, step, time.time()-start_time))

            if episode % FLAGS.episode_per_checkpoint == 0:
                agent.dqn.save_model(FLAGS.models_dir, episode)

        # plot cost and reward
        agent.dqn.plot_cost(FLAGS.graphs_dir)
        agent.dqn.plot_reward(FLAGS.graphs_dir)
        agent.plot_hp(FLAGS.graphs_dir)

        if FLAGS.save_replay:
            env.save_replay(agent)

    if FLAGS.profile:
        print(stopwatch.sw)

if __name__ == "__main__":
    # Remeber to add this later
    # config = tf.ConfigProto()
    # if FLAGS.gpu_fraction > 0.0:
    #     print("GPU Fraction: {:.3f}".format(FLAGS.gpu_fraction))
    #     config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    # else:
    #     print("GPU Fraction: Allow Growth")
    #     config.gpu_options.allow_growth = True
    app.run(main)
