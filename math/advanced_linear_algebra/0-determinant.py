#!/usr/bin/env python3
"""
Module that computes the determinant of a square matrix.
"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Args:
        matrix (list of list of int/float): input matrix

    Returns:
        int/float: determinant value

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square
    """

    if (
	not isinstance(matrix, list)
	or not all(isinstance(row, list) for row in matrix)
	):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        minor = [
            [matrix[i][j] for j in range(n) if j != col]
            for i in range(1, n)
        ]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det
