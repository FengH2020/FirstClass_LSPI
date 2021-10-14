#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from Data_sample import DataSample
from lstdq import LSTDQ
from basisfunc import basis_phi

if __name__ == '__main__':
    N = np.power(10, 3)  # the size of the sample set

    epsilon = 10 ** -6  # the threshold of the distance between the two successive weights of the policy
    maxnum_iter = 10**2   # the maximum number of iterations
    num_state = int(4)        # the number of states
    num_action = int(2)       # the number of actions
    reward_location = [1, 2]  # assign the locations of endowing reward
    failure_pro = 0.1  # the probability of failing to implement an action

    init_state = random.randint(0, num_state-1)  # initialize the state
    init_policy = np.zeros(num_state, dtype=int)     # initialize the deterministic policy
    for i in range(num_state):
        init_policy[i] = random.randint(0, num_action-1)

    sample_mat = np.zeros((N, 4), dtype=int)  # the matrix consisting of all samples
    sample = np.zeros(4, dtype=int)
    for i in range(N):     # obtaining the sample set
        init_action = init_policy[init_state]
        sample = DataSample(init_state, init_action, reward_location, failure_pro).sample()
        sample_mat[i, ...] = sample
        init_state = sample[3]

    degree_ = 2
    basis_dim = (degree_+1) * num_action
    discount_factor = 0.9
    distance_ = float('inf')
    num_iter = 0
    err_ = 0.08
    epsilon = 10**-4  # the threshold of the distance between the two successive weights of the policy
    init_weight = LSTDQ(sample_mat, degree_, num_action, discount_factor, init_policy)
    while distance_ > epsilon and num_iter <= maxnum_iter:
        print("the initial policy is {}, \n the weight is \n {}".format(init_policy, init_weight))
        num_iter += 1
        for i in range(num_state):
            Q_s_max = float('-inf')
            max_action = -1
            for j in range(num_action):
                temp = np.dot(basis_phi(i, j, degree_, num_action), init_weight)
                if temp[0, 0] > Q_s_max:
                    Q_s_max = temp[0, 0]
                    max_action = j

            if random.random() < err_:
                init_policy[i] = random.randint(0, num_action-1)
            else:
                init_policy[i] = max_action
        # next_weight = LSTDQ(np.random.permutation(sample_mat), degree_, num_action, discount_factor, init_policy)

        init_state = random.randint(0, num_state - 1)  # initialize the state
        sample = np.zeros(4, dtype=int)
        for i in range(N):  # obtaining the sample set
            init_action = init_policy[init_state]
            sample = DataSample(init_state, init_action, reward_location, failure_pro).sample()
            sample_mat[i, ...] = sample
            init_state = sample[3]

        next_weight = LSTDQ(sample_mat, degree_, num_action, discount_factor, init_policy)
        print("update policy is {}, \n update weight is \n {}".format(init_policy, next_weight))
        distance_ = max(np.abs(np.subtract(init_weight, next_weight)))
        print("distance is {}  ".format(distance_))
        init_weight = next_weight
    print(num_iter)





