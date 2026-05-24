#!/usr/bin/env python3
"""
Module that determines the definiteness of a matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Determines definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): square matrix

    Returns:
        str or None: type of definiteness or None
    """

    # Type check
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Valid matrix check
    if (
        matrix.size == 0
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        return None

    # Must be symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    eps = 1e-10

    all_positive = np.all(eigenvalues > eps)
    all_non_negative = np.all(eigenvalues >= -eps)
    all_negative = np.all(eigenvalues < -eps)
    all_non_positive = np.all(eigenvalues <= eps)

    # Classification rules
    if all_positive:
        return "Positive definite"

    if all_non_negative and np.any(np.abs(eigenvalues) <= eps):
        return "Positive semi-definite"

    if all_negative:
        return "Negative definite"

    if all_non_positive and np.any(np.abs(eigenvalues) <= eps):
        return "Negative semi-definite"

    return "Indefinite"
