#!/usr/bin/env python3
""" Norm Constants"""

import numpy as np


def normalization_constants(X):
    """ Norm Constants

    Args:
        X (_type_): _description_
    Returns:
        _type_: _description_
    """

    return X.mean(axis=0), X.std(axis=0)
