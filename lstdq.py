#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/13 21:58
# @Author  : Little Feng
# @Email   : LittleFeng@email.com
# @File    : lstdq.py
# @Software: PyCharm

from copy import copy
import numpy as np
from basisfunc import basis_phi


def LSTDQ(sample_mat, degree_, num_action, discount_factor, init_policy):
    n = np.shape(sample_mat)[0]
    # phi_s_a = basis_phi(sample_mat[0, 0], sample_mat[0, 1], basis_dim)
    k = (degree_+1) * num_action
    a_mat = np.zeros((k, k))
    np.fill_diagonal(a_mat, 0.00001)
    b_vec = np.zeros((k, 1))

    for i in range(n):
        a_mat += np.dot(basis_phi(sample_mat[i, 0], sample_mat[i, 1], degree_, num_action).T,
                        np.subtract(
                            basis_phi(sample_mat[i, 0], sample_mat[i, 1], degree_, num_action),
                            discount_factor *
                            basis_phi(sample_mat[i, 3], init_policy[sample_mat[i, 3]], degree_, num_action)
                        ))
        b_vec += sample_mat[i, 2] * basis_phi(sample_mat[i, 0], sample_mat[i, 1], degree_, num_action).T

    a_rank = np.linalg.matrix_rank(a_mat)
    if a_rank == k:
        w = np.linalg.solve(a_mat, b_vec)
        return w
    else:
        w = np.linalg.lstsq(a_mat, b_vec)
        raise ValueError("A matrix {} is not full rank; its lstsq solution is {}".format(a_mat, w))
