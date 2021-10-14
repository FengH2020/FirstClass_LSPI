#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/13 17:34
# @Author  : Little Feng
# @Email   : LittleFeng@email.com
# @File    : Data_sample.py
# @Software: PyCharm
import numpy as np
from random import random


class DataSample(object):
    def __init__(self, init_state, init_action, reward_location, failure_pro):
        self.init_state = init_state
        self.init_action = init_action
        self.reward_location = reward_location
        self.failure_pro = failure_pro

    def sample(self):

        if random() < self.failure_pro:
            action_failure = True
        else:
            action_failure = False

        next_state = -1
        # move left
        if (self.init_action == 0 and not action_failure)\
                or (self.init_action == 1 and action_failure):
            next_state = max(0, self.init_state-1)
        # move right
        if (self.init_action == 0 and action_failure)\
                or (self.init_action == 1 and not action_failure):
            next_state = min(3, self.init_state + 1)

        # calculating the reward
        if next_state == -1:
            raise ValueError('state is out of range')
        elif next_state in self.reward_location:
            reward = 1
        else:
            reward = 0

        sample_vec = np.array([self.init_state, self.init_action, reward, next_state])
        return sample_vec
