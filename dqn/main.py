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
from pysc2.lib.actions import ActionSpace

from new_agent import DQN_Agent as MyAgentClass
from absl import app, flags
import tensorflow as tf
import os
import time

FLAGS = flags.FLAGS
# specifies the maximum step count for this run

flags.DEFINE_integer("max_steps", 250, "Total agent steps.")
flags.DEFINE_integer("max_episodes", 50000, "Total agent steps.")
flags.DEFINE_integer("episode_per_checkpoint", 250, "Total agent steps.")

flags.DEFINE_string("map", 'HK2V1', "Name of a map to use.")

flags.DEFINE_integer("screen_resolution", 128, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("game_steps_per_learn", 30, "Game steps per Learn.")
flags.DEFINE_integer("episodes_per_copy", 50, "Episodes per Copy target params with training params.")

flags.DEFINE_integer("step_mul", 5, "Game steps per agent step.")
flags.DEFINE_float('gamma', 0.9, 'Decay rate')

flags.DEFINE_bool("visualize", True, "Whether to render with pygame.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")

flags.DEFINE_integer("memory_size", 10000, "Size of memory for experience replay")
flags.DEFINE_integer("batch_size", 64, "Size of batch")
flags.DEFINE_integer("channel_size", 3, "Size of features: 3 => pixels only, 4 => with hp")
flags.DEFINE_integer("n_actions", 3, "Number of actions")

flags.DEFINE_float('gpu_fraction', 0.0, 'GPU fraction, 0.0 => allow growth')
flags.DEFINE_float('lr', 0.001, 'Learning_rate')

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
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

    config = tf.ConfigProto()
    if FLAGS.gpu_fraction > 0.0:
        print("GPU Fraction: {:.3f}".format(FLAGS.gpu_fraction))
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    else:
        print("GPU Fraction: Allow Growth")
        config.gpu_options.allow_growth = True

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
    with tf.Session() as sess:
        agent = MyAgentClass(FLAGS, sess)

        # restore the model
        agent.model.load_model(sess, FLAGS.models_dir)
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, graph=sess.graph)

        _ = sess.run([agent.model.copy_op])
        with sc2_env.SC2Env(
                map_name=FLAGS.map,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    rgb_dimensions=sc2_env.Dimensions(
                        screen=FLAGS.screen_resolution,
                        minimap=FLAGS.minimap_resolution,
                    ),
                    feature_dimensions=sc2_env.Dimensions(
                        screen=FLAGS.screen_resolution,
                        minimap=FLAGS.minimap_resolution,
                    ),

                    # not too sure abt how to set the action_space value
                    action_space=ActionSpace.RGB,
                    use_feature_units=True,
                ),
                visualize=FLAGS.visualize,
        ) as env:
            env = available_actions_printer.AvailableActionsPrinter(env)

            observation_spec = env.observation_spec()
            action_spec = env.action_spec()
            agent.setup(observation_spec, action_spec)

            q_loss = 0.
            for episode in range(FLAGS.max_episodes):
                start_time = time.time()
                step = -1
                # Don't modify timesteps
                timesteps = env.reset()
                agent.reset()

                try:
                    while step < FLAGS.max_steps:
                        step += 1
                        actions = [agent.step(timestep, episode, step) for timestep in timesteps]

                        if timesteps[0].last():
                            break

                        if step % FLAGS.game_steps_per_learn == 0 and episode >= 10:
                            s, r, a, s_ = agent.memory.get_feed_dict_vars()
                            feed_dict = {
                                agent.model.s: s,
                                agent.model.r: r,
                                agent.model.a: a,
                                agent.model.s_: s_,
                                agent.model.training: True
                            }

                            _, q_loss, q_sumop = sess.run(
                                [agent.model.trainerQ, agent.model.q_loss, agent.model.q_loss_sumop], feed_dict=feed_dict)

                        timesteps = env.step(actions)
                except KeyboardInterrupt:
                    pass
                finally:
                    # Convert reward to advantage and store them from temp memory to main memory
                    agent.memory.update_memory()

                print("Episode: {} Step {} Q_Loss {} Time {}".format(episode, step, q_loss,
                                                                     time.time() - start_time))
                if episode > 10:
                    summary_writer.add_summary(q_sumop, episode)

                if episode % FLAGS.episodes_per_copy == 0:
                    _ = sess.run([agent.model.copy_op])

                if episode % FLAGS.episode_per_checkpoint == 0:
                    agent.model.save_model(sess, FLAGS.models_dir, episode)


            if FLAGS.save_replay:
                env.save_replay(agent)

    if FLAGS.profile:
        print(stopwatch.sw)

if __name__ == "__main__":
    # Remeber to add this later
    app.run(main)
