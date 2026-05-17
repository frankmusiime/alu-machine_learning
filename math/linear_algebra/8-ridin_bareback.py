#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            new_row.append(total)
        result.append(new_row)

    return result
