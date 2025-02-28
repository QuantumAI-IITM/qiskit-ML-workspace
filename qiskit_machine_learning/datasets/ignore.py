import numpy as np
import functools as ft


def _n_z(i: int, n: int):

    z = np.diag([1, -1])
    i_1 = np.eye(2**i)
    i_2 = np.eye(2**(n-i-1))

    result = np.kron(i_1,z)
    result = np.kron(result, i_2)

    return result

# print(_n_z(2,4) - ft.reduce(np.kron, [np.eye(2)] * 2 + [np.diag([1,-1])] + [np.eye(2)] * (4 - 2 - 1)))