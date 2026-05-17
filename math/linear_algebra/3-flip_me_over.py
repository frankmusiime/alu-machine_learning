#!/usr/bin/env python3
"""Function that returns the transpose of a 2D matrix."""


def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]
