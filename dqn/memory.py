import numpy as np
import sys

class Memory(object):
    def __init__(self, FLAGS):
        print("Initialize Memory......................")
        # Use for storing entire history of transitions
        self.memory_size =FLAGS.memory_size
        self.prev_s = np.empty((self.memory_size, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size))
        self.r = np.empty((self.memory_size))
        self.a = np.empty((self.memory_size))
        self.curr_s = np.empty((self.memory_size, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size))

        # Use for storing current episode's transitions
        self.temp_memory_size = FLAGS.max_steps
        self.temp_prev_s = np.zeros((self.temp_memory_size, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size))
        self.temp_r = np.zeros((self.temp_memory_size))
        self.temp_a = np.zeros((self.temp_memory_size))
        self.temp_curr_s = np.zeros((self.temp_memory_size, FLAGS.screen_resolution, FLAGS.screen_resolution, FLAGS.channel_size))

        self.action_size = FLAGS.n_actions
        self.batch_size = FLAGS.batch_size
        self.gamma = FLAGS.gamma
        # count for transitions in the temp memory
        self.temp_count = 0
        # count for transitions in the main memory
        self.count = 0

    def store_temp_transition(self, prev_s, a, r, curr_s):
        self.temp_prev_s[self.temp_count] = prev_s
        self.temp_a[self.temp_count] = a
        self.temp_r[self.temp_count] = r
        self.temp_curr_s[self.temp_count] = curr_s
        self.temp_count += 1

    def update_memory(self):
        # get the correct reward
        # advantages = np.zeros((self.temp_count))
        # r_reversed_sum = 0
        # for i in reversed(range(self.temp_count)):
        #     r_reversed_sum = self.temp_r[i] + self.gamma * r_reversed_sum
        #     advantages[i] = r_reversed_sum
        #
        # # normalize advantage
        # advantages -= np.mean(advantages)
        # advantages /= np.std(advantages)

        # store temp transitions into main memory
        for id in range(self.temp_count):
            if self.count >= self.memory_size:
                self.count = 0

            self.prev_s[self.count] = self.temp_prev_s[id]
            # self.r[self.count] = advantages[id]
            self.r[self.count] = self.temp_r[id]
            # one hot style for action
            # self.a[self.count] = np.zeros(self.action_size)
            self.a[self.count] = self.temp_a[id]
            self.curr_s[self.count] = self.temp_curr_s[id]
            self.count += 1

        # clean up temp memory
        self.clear_memory()

    def get_feed_dict_vars(self):
        sampled_ids = np.random.randint(self.count, size=self.batch_size)
        return self.prev_s[sampled_ids], self.r[sampled_ids], self.a[sampled_ids], self.curr_s[sampled_ids]

    def clear_memory(self):
        # print("Cleaning Memory .......................")
        self.temp_prev_s = np.zeros((self.temp_prev_s.shape))
        self.temp_r = np.zeros((self.temp_r.shape))
        self.temp_a = np.zeros((self.temp_a.shape))
        self.temp_curr_s = np.zeros((self.temp_curr_s.shape))
        self.temp_count = 0
