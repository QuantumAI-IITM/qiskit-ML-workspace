import numpy as np
import functools as ft
import itertools as it

from scipy.stats.qmc import Sobol

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

# print(z_n - _n_z(h_n))

def _modified_LHC(n:int, n_samples:int, n_div:int):

    samples = np.empty((n_samples,n),dtype = float)
    bin_size = 2*np.pi/n_div
    n_passes = (n_samples+n_div-1)//n_div

    all_bins = np.tile(np.arange(n_div),n_passes)

    for dim in range(n):
        np.random.shuffle(all_bins)
        chosen_bins = all_bins[:n_samples]
        offsets = np.random.random(n_samples)
        samples[:, dim] = (chosen_bins+offsets)*bin_size

    return samples


# print(_modified_LHC(3, 10, 5))

def _sobol_sampling(n, n_samples):
    sampler = Sobol(d=n, scramble=True)
    p = 2*np.pi*sampler.random(n_samples)
    return p

# print(_sobol_sampling(3,10))

x_vecs = _modified_LHC(3, 10, 5)


n = 3
# Single qubit Z gates
z_diags = np.array([np.diag(_i_z(i,n)).reshape((1,-1)) for i in range(n)])

# Precompute Pairwise ZZ block diagonals
zz_diags = {}
for (i, j) in it.combinations(range(n), 2):
    zz_diags[(i, j)] = z_diags[i] * z_diags[j]

def _phi_i(x_vecs: np.ndarray, i: int):
    return x_vecs[:,i].reshape((-1,1))

def _phi_ij(x_vecs: np.ndarray, i: int, j: int):
    return ((np.pi - x_vecs[:,i])*(np.pi - x_vecs[:,j])).reshape((-1,1))


pre_exp = np.zeros((10,8))

ind_pairs = zz_diags.keys()

dims = 8
# First Order Terms
for i in range(n):
    pre_exp += _phi_i(x_vecs, i)*z_diags[i]
# Second Order Terms 
for (i,j) in ind_pairs:
    pre_exp += _phi_ij(x_vecs, i, j)*zz_diags[(i,j)]

# Since pre_exp is purely diagonal, exp(A) = diag(exp(Aii))
post_exp = np.exp(1j * pre_exp)

Uphi = np.zeros((10, dims, dims), dtype = post_exp.dtype)
cols = range(dims)
Uphi[:,cols, cols] = post_exp[:, cols]

# print(Uphi)

# V change of basis: Eigenbasis of a random hermitian will be a random unitary
A = np.array(np.random.random((dims, )) 
            + 1j * np.random.random((dims, dims)))
Herm = A.conj().T @ A 
eigvals, eigvecs = np.linalg.eig(Herm)
idx = eigvals.argsort()[::-1]
V = eigvecs[:, idx]

# Observable for labelling boundary
O = V.conj().T @ z_n @ V
print(O.shape)

dims = 2**n
psi_0 = np.ones(dims) / np.sqrt(dims)
Psi = (Uphi @ h_n @ Uphi @ psi_0).reshape((-1, dims, 1))
Psi_dag = np.transpose(Psi.conj(), (0, 2, 1))
exp_val = np.real(Psi_dag @ O @ Psi)


print(exp_val.shape)