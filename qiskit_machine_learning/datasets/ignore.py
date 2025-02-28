import numpy as np
import functools as ft


def _n_hadamard(n: int):
    
    base = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = 1
    expo = n

    while expo>0:
        if expo%2==1:
            result = np.kron(result, base)
        base = np.kron(base, base)
        expo //= 2

    return result

def _i_z(i: int, n: int):

    z = np.diag([1, -1])
    i_1 = np.eye(2**i)
    i_2 = np.eye(2**(n-i-1))

    result = np.kron(i_1,z)
    result = np.kron(result, i_2)

    return result
    
def _n_z(h_n: np.ndarray):
    res = np.diag(h_n)
    res = np.sign(res)
    res = np.diag(res)
    return res

# print(_n_z(2,4) - ft.reduce(np.kron, [np.eye(2)] * 2 + [np.diag([1,-1])] + [np.eye(2)] * (4 - 2 - 1)))

h_n = _n_hadamard(3)
z_n = _i_z(0,3)*_i_z(1,3)*_i_z(2,3)

print(z_n - _n_z(h_n))