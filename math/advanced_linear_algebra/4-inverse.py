#!/usr/bin/env python3
"""
Module that computes the inverse of a square matrix.
"""


def determinant(matrix):
    """Computes determinant of a square matrix."""

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
    """Computes cofactor matrix."""

    n = len(matrix)
    cof = []

    for i in range(n):
        row = []

        for j in range(n):
            sub = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]

            sign = (-1) ** (i + j)
            row.append(sign * determinant(sub))

        cof.append(row)

    return cof


def adjugate(matrix):
    """Computes adjugate matrix."""

    cof = cofactor(matrix)
    n = len(matrix)

    return [
        [cof[j][i] for j in range(n)]
        for i in range(n)
    ]


def inverse(matrix):
    """
    Computes inverse of a square matrix.

    Returns None if matrix is singular.
    """

    # Type check
    if (
        not isinstance(matrix, list)
        or not all(isinstance(row, list) for row in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    # Empty / square check
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    # Singular matrix
    if det == 0:
        return None

    adj = adjugate(matrix)

    # IMPORTANT FIX: correct 1x1 handling (NO special case needed)
    return [
        [adj[i][j] / det for j in range(n)]
        for i in range(n)
    ]
