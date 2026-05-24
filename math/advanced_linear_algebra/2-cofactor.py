#!/usr/bin/env python3
"""
Module that computes the cofactor matrix of a square matrix.
"""


def determinant(matrix):
    """
    Computes determinant of a square matrix.
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
        sub = [
            [matrix[i][j] for j in range(n) if j != col]
            for i in range(1, n)
        ]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub)

    return det


def cofactor(matrix):
    """
    Computes the cofactor matrix of a square matrix.

    Args:
        matrix (list of list): input matrix

    Returns:
        list of list: cofactor matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is empty or not square
    """

    # Type check
    if (
        not isinstance(matrix, list)
        or not all(isinstance(row, list) for row in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    # Empty / invalid square check
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 case
    if n == 1:
        return [[1]]

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
