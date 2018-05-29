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
flags.DEFINE_integer("max_agent_steps", 25000, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 2, "Game steps per agent step.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.mark_flag_as_required("map")

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
    #path = 'models/' + agent_name

    # restore the model only if u have the previously trained a model
    #agent.dqn.load_model(path)

    # run the steps
    run_loop([agent], env, FLAGS.max_agent_steps)

    # save the model
    #agent.dqn.save_model(path, 1)

    # plot cost and reward
    save_pic = True
    #agent.dqn.plot_cost(path, save=save_pic)
    #agent.dqn.plot_reward(path, save=save_pic)
    #agent.plot_player_hp(path, save=save_pic)
    #agent.plot_enemy_hp(path, save=save_pic)

    if FLAGS.save_replay:
      env.save_replay(agent_cls.__name__)

def run_loop(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)
  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))

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