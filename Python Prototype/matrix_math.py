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