import math

def matrix_mult(A, B):
    if len(A[0]) != len(B):
        raise ValueError("The number of columns of A must be equal to the number of rows of B.")
    for i in range(len(A)):
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += A[i][k] * B[k][j]
            if i == 0:
                result_row = [sum]
            else:
                result_row.append(sum)
        if i == 0:
            result = [result_row]
        else:
            result.append(result_row)
    return result

def add_matrix(A, B):
    if len(A) != len(B):
        raise ValueError("Vectors must be of the same length.")
    return [A[i] + B[i] for i in range(len(A))]

def transpose_matrix(A):
    if not A:
        return []
    transposed = []
    for i in range(len(A[0])):
        transposed_row = []
        for j in range(len(A)):
            transposed_row.append(A[j][i])
        transposed.append(transposed_row)
    return transposed

def dot_product(A, B):
    if len(A) != len(B):
        raise ValueError("Vectors must be of the same length.")
    return sum(A[i] * B[i] for i in range(len(A)))

def scalar_mult(scalar, A):
    return [[scalar * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def elementwise_mult(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must be of the same dimensions.")
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

