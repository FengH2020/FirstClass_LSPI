#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 12:58
# @Author  : Little Feng
# @Email   : LittleFeng@email.com
# @File    : basisfunc.py
# @Software: PyCharm (sample_mat, rank_k, basis_phi, discount_gamma, policy_):

import numpy as np


def basis_phi(state, action, degree_, num_action):
    basis_dim = (degree_ + 1) * num_action
    phi_ = np.zeros((1, basis_dim), dtype=int)
    for i in range(degree_+1):
        phi_[0, (i + action * (degree_ + 1))] = np.power(state, i)

    return phi_
