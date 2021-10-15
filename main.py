#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from Data_sample import DataSample
from lstdq import LSTDQ
from basisfunc import basis_phi

import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = np.power(10, 2)  # the size of the sample set

    maxnum_iter = 10**2   # the maximum number of iterations
    num_state = int(4)        # the number of states
    num_action = int(2)       # the number of actions
    reward_location = [1, 2]  # assign the locations of endowing reward
    failure_pro = 0.1  # the probability of failing to implement an action

    init_state = random.randint(0, num_state-1)  # initialize the state
    init_policy = np.zeros(num_state, dtype=int)     # initialize the deterministic policy
    next_policy = np.zeros(num_state, dtype=int)     # the improved deterministic policy

    num_episode = 8
    sample_mat = np.zeros((N*num_episode, 4), dtype=int)  # the matrix consisting of all samples
    for k_ in range(8):
        sample = np.zeros(4, dtype=int)
        for l_ in range(num_state):
            init_policy[l_] = random.randint(0, num_action - 1)
        for i in range(N):     # obtaining the sample set
            init_action = init_policy[init_state]
            sample = DataSample(init_state, init_action, reward_location, failure_pro).sample()
            sample_mat[i+k_*N, ...] = sample
            init_state = sample[3]

    degree_ = 2
    basis_dim = (degree_+1) * num_action
    discount_factor = 0.9
    distance_ = float('inf')
    num_iter = 0
    explor_ = 0.05
    epsilon = 10**-5  # the threshold of the distance between the two successive weights of the policy
    init_weight = LSTDQ(sample_mat, degree_, num_action, discount_factor, init_policy)
    fig, ax = plt.subplots()
    global line1
    global line2
    while distance_ > epsilon and num_iter <= maxnum_iter:
        for xj in range(num_state):
            if init_policy[xj] == 0:
                line1, = ax.plot(num_iter+1, xj, '<b', ms=10)
            else:
                line2, = ax.plot(num_iter+1, xj, '>r', ms=10)
        print("the policy is {}, \n the weight is \n {}".format(init_policy, init_weight))
        num_iter += 1
        for i in range(num_state):
            Q_s_max = float('-inf')
            max_action = -1
            for j in range(num_action):
                temp = np.dot(basis_phi(i, j, degree_, num_action), init_weight)
                if temp[0, 0] > Q_s_max:
                    Q_s_max = temp[0, 0]
                    max_action = j
            if random.random() < explor_:
                next_policy[i] = random.randint(0, num_action-1)
            else:
                next_policy[i] = max_action
        # next_weight = LSTDQ(np.random.permutation(sample_mat), degree_, num_action, discount_factor, init_policy)
        # if max(np.abs(np.subtract(init_policy, next_policy))) == 0:
        #     continue
        # else:
        next_weight = LSTDQ(sample_mat, degree_, num_action, discount_factor, next_policy)

        distance_ = max(np.abs(np.subtract(init_weight, next_weight)))
        print("distance is {}  ".format(distance_))
        init_weight = next_weight
        init_policy = next_policy
    ax.set_xlabel('Time step of learning')
    ax.set_ylabel('Different states')
    ax.set_title("Seeking optimal policy by LSPI")
    plt.grid(axis='x', linestyle=':')
    plt.legend([line1, line2], ['Left', 'Right'], loc=0)
    x_major_locator = plt.MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    y_major_locator = plt.MultipleLocator(1)
    # 把y轴的刻度间隔设置为1，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为1的倍数
    plt.show()
    print(num_iter)



