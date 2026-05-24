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

    # Check type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check valid matrix
    if matrix.size == 0 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Ensure symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Numerical tolerance
    eps = 1e-10

    positive = np.all(eigenvalues > eps)
    non_negative = np.all(eigenvalues >= -eps)
    negative = np.all(eigenvalues < -eps)
    non_positive = np.all(eigenvalues <= eps)

    # Classification
    if positive:
        return "Positive definite"

    if non_negative and np.any(np.abs(eigenvalues) <= eps):
        return "Positive semi-definite"

    if negative:
        return "Negative definite"

    if non_positive and np.any(np.abs(eigenvalues) <= eps):
        return "Negative semi-definite"

    return "Indefinite"
