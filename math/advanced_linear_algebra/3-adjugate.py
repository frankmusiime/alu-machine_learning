#!/usr/bin/env python3
"""
Module that computes the adjugate matrix of a square matrix.
"""


def determinant(matrix):
    """
    Helper function to compute determinant of a square matrix.
    """

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    det = 0

    for col in range(n):
        minor = [
            [matrix[i][j] for j in range(n) if j != col]
            for i in range(1, n)
        ]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det


def cofactor(matrix):
    """
    Computes the cofactor matrix of a square matrix.
    """

    n = len(matrix)
    cofactor_matrix = []

    for i in range(n):
        row = []

        for j in range(n):
            sub_matrix = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]

            sign = (-1) ** (i + j)
            row.append(sign * determinant(sub_matrix))

        cofactor_matrix.append(row)

    return cofactor_matrix


def adjugate(matrix):
    """
    Computes the adjugate matrix of a square matrix.

    Args:
        matrix (list of list of int/float): input matrix

    Returns:
        list of list: adjugate matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is empty or not square
    """

    if (
        not isinstance(matrix, list)
        or not all(isinstance(row, list) for row in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)

    # transpose cofactor matrix
    adj = [
        [cof[j][i] for j in range(n)]
        for i in range(n)
    ]

    return adj
