#!/usr/bin/env python3
"""Function that concatenates two numpy arrays along a given axis."""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    return np.concatenate((mat1, mat2), axis=axis)
