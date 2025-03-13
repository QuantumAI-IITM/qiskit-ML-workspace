# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""

Ad Hoc Dataset

"""
from __future__ import annotations
from functools import reduce
import itertools as it
from typing import Tuple, Dict, List
import numpy as np
from sklearn import preprocessing
from qiskit.utils import optionals
from ..utils import algorithm_globals
from scipy.stats.qmc import Sobol
import warnings

# pylint: disable=too-many-positional-arguments
def ad_hoc_data(
    training_size: int,
    test_size: int,
    n: int,
    gap: int = 0,
    plot_data: bool = False,
    one_hot: bool = True,
    include_sample_total: bool = False,
    entanglement: str = "full",
    sampling_method: str = "grid",
    divisions: int = 0,
    labelling_method: str = "expectation",
    reps: int = 1
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    r"""
    Generates a dataset that can be fully separated with
    :class:`~qiskit.circuit.library.ZZFeatureMap` according to the procedure
    outlined in [1]. 

    To construct the dataset, we first sample uniformly distributed vectors 
    :math:`\vec{x} \in (0, 2\pi]^{n}` and apply the feature map:

    .. math::
        |\Phi(\vec{x})\rangle = U_{{\Phi} (\vec{x})} H^{\otimes n} 
        U_{{\Phi} (\vec{x})} H^{\otimes n} |0^{\otimes n} \rangle

    where:

    .. math::
        U_{{\Phi} (\vec{x})} = \exp \left( i \sum_{S \subseteq [n] } 
        \phi_S(\vec{x}) \prod_{i \in S} Z_i \right)

    .. math::
        \begin{cases}
        \phi_{\{i, j\}} = (\pi - x_i)(\pi - x_j) \\
        \phi_{\{i\}} = x_i
        \end{cases}

    The second-order terms included in the above summation are decided by the 
    entanglement configuration passed as an argument to the function. 
    (See Args for more information). 

    We then attribute labels to the vectors according to the observable O below:

    .. math::
        O = V^\dagger \prod_i Z_i V

    Following this, the labelling method passed as an argument determines whether
    the label is based on a simple measurement done on the resulting quantum state
    or based on expectation as shown below:

    .. math::
        m(\vec{x}) = \begin{cases}
        1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle > \Delta \\
        -1 & \langle \Phi(\vec{x}) | V^\dagger \prod_i Z_i V | \Phi(\vec{x}) \rangle < -\Delta
        \end{cases}

    where :math:`\Delta` is the separation gap, and
    :math:`V` is a random unitary matrix.

    The method used for the uniform sampling of :math:`\vec{x}` is decided by the 
    sampling method argument given. (See Args for more information)

    **References:**

    [1] Havlíček V, Córcoles AD, Temme K, Harrow AW, Kandala A, Chow JM,
    Gambetta JM. Supervised learning with quantum-enhanced feature
    spaces. Nature. 2019 Mar;567(7747):209-12.
    `arXiv:1804.11326 <https://arxiv.org/abs/1804.11326>`_

    Args:
        training_size (int): Number of training samples.
        test_size (int): Number of testing samples.
        n (int): Number of qubits (dimension of the feature space).
        gap (int, optional): If `labelling_method="expectation"`, this defines the separation gap (Δ).
        plot_data (bool, optional): Whether to plot the data. Disabled if `n > 3`.
        one_hot (bool, optional): If `True`, returns labels in one-hot format.
        include_sample_total (bool, optional): If `True`, returns the total number
            of accepted samples along with training and testing samples.
        entanglement (str, optional): 
            - `"linear"`: Includes all terms of the form :math:`Z_{i}Z_{i+1}`.
            - `"circular"`: Includes `linear` terms and additionally :math:`Z_{n-1}Z_{0}`.
            - `"full"`: Includes all possible pairs :math:`Z_iZ_j`.
        sampling_method (str, optional): 
            - `"grid"`: Generates a uniform grid and selects :math:`\vec{x}` from the grid.
              (Only supported for `n <= 3`).
            - `"hypercube"`: Uses a variant of the Latin hypercube to generate 1D stratified samples.
            - `"sobol"`: Uses Sobol sequences to generate uniformly distributed samples.
        divisions (int, optional): If the `"hypercube"` method is used for sampling, `divisions` 
            must be defined. This determines the number of divisions each 1D stratification makes.
            Recommended to set this close to `training_size`.
        labelling_method (str, optional): Determines how labels are assigned.
            - `"expectation"`: Labels the datapoints based on the expectation value discussed above.
            - `"measurement"`: Labels the datapoints based on a simple measurement performed on the states.
        reps (int, optional): If `"measurement"` is used as the `labelling_method`, setting `reps > 1` 
            will result in repeated measurements of the same datapoint in the final dataset. The resulting
            dataset will be of sizes reps*training_size and reps*test_size

    Returns:
        Tuple: A tuple containing:
            - Training features (`np.ndarray`)
            - Training labels (`np.ndarray`)
            - Testing features (`np.ndarray`)
            - Testing labels (`np.ndarray`)
            - (Optional) Total accepted samples (`int`), if `include_sample_total` is True.
    """

    # Errors
    if training_size < 0: raise ValueError("Training size can't be less than 0")
    if test_size < 0: raise ValueError("Test size can't be less than 0")
    if n < 0: raise ValueError("Number of qubits can't be less than 0")
    if gap < 0 and labelling_method == "expectation": raise ValueError("Gap can't be less than 0")
    if entanglement not in {"linear", "circular", "full"}: raise ValueError("Invalid entanglement type. Must be 'linear', 'circular', or 'full'.")
    if sampling_method not in {"grid", "hypercube", "sobol"}: raise ValueError("Invalid sampling method. Must be 'grid', 'hypercube', or 'sobol'.")
    if divisions == 0 and sampling_method == "hypercube": raise ValueError("Divisions must be set for 'hypercube' sampling.")
    if labelling_method not in {"expectation", "measurement"}: raise ValueError("Invalid labelling method. Must be 'expectation' or 'measurement'.")
    if n > 3 and sampling_method == "grid": raise ValueError("Grid sampling is unsupported for n > 3.")

    # Warnings
    if n > 3 and plot_data:
        warnings.warn("Plotting for n > 3 is unsupported. Disabling plot_data.", UserWarning)
        plot_data = False

    # Initial State
    dims = 2**n
    psi_0 = np.ones(dims) / np.sqrt(dims)

    # n-qubit Hadamard
    h_n = _n_hadamard(n)

    # Single qubit Z gates
    z_diags = np.array([np.diag(_i_z(i,n)).reshape((1,-1)) for i in range(n)])

    # Precompute ZZ Entanglements
    zz_diags = {}
    if entanglement=="full":
        for (i, j) in it.combinations(range(n), 2):
            zz_diags[(i, j)] = z_diags[i] * z_diags[j] 
    else:
        for i in range(n-1):
            zz_diags[(i,i+1)] = z_diags[i] * z_diags[i+1]
        if entanglement=="circular":
            zz_diags[(n-1,0)] = z_diags[n-1] * z_diags[0]

    # n-qubit Z gate: notice that h_n[0,:] has the same elements as diagonal of z_n
    z_n = _n_z(h_n)

    # V change of basis: Eigenbasis of a random hermitian will be a random unitary
    V = _random_unitary(dims)

    # Observable for labelling boundary
    O = V.conj().T @ z_n @ V

    n_samples = training_size+test_size

    # Labelling Methods
    if labelling_method=="expectation":
        lab_fn = lambda x: _exp_label(x,gap)
    else:
        lab_fn = lamdba x: _measure(x)

    # Sampling Methods
    if sampling_method == "grid":
        features, labels = _grid_sampling(n, n_samples, z_diags, zz_diags, O, h_n, lab_fn)
    else:
        if sampling_method == "hypercube":
            samp_fn = lambda a,b: _modified_LHC(a,b,divisions)
        else:
            samp_fn = lambda a,b: _sobol_sampling(a,b)

        features, labels = _loop_sampling(n, n_samples, z_diags, zz_diags, O, h_n, lab_fn, samp_fn)

    if plot_data: _plot_ad_hoc_data(features, labels, training_size)

    if one_hot:
        labels = _onehot_labels(labels)

    res = [
        features[:training_size],
        labels[:training_size],
        features[training_size:],
        labels[training_size:],
    ]
    if include_sample_total:
        res.append(cur)

    return tuple(res)


@optionals.HAS_MATPLOTLIB.require_in_call
def _plot_ad_hoc_data(x_total: np.ndarray, y_total: np.ndarray, training_size: int) -> None:
    """Plot the ad hoc dataset.

    Args:
        x_total (np.ndarray): The dataset features.
        y_total (np.ndarray): The dataset labels.
        training_size (int): Number of training samples to plot.
    """
    import matplotlib.pyplot as plt

    n = x_total.shape[1]
    fig = plt.figure()
    projection = "3d" if n == 3 else None
    ax1 = fig.add_subplot(1, 1, 1, projection=projection)
    for k in range(0, 2):
        ax1.scatter(*x_total[y_total == k][:training_size].T)
    ax1.set_title("Ad-hoc Data")
    plt.show()

def _onehot_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels to one-hot encoded format.

    Args:
        labels (np.ndarray): Array of labels.

    Returns:
        np.ndarray: One-hot encoded labels.
    """
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False)
    labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))
    return labels_one_hot

def _n_hadamard(n: int) -> np.ndarray:
    """Generate an n-qubit Hadamard matrix.

    Args:
        n (int): Number of qubits.

    Returns:
        np.ndarray: The n-qubit Hadamard matrix.
    """
    base = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = 1
    expo = n

    while expo > 0:
        if expo % 2 == 1:
            result = np.kron(result, base)
        base = np.kron(base, base)
        expo //= 2

    return result

def _i_z(i: int, n: int) -> np.ndarray:
    """Create the i-th single-qubit Z gate in an n-qubit system.

    Args:
        i (int): Index of the qubit.
        n (int): Total number of qubits.

    Returns:
        np.ndarray: The Z gate acting on the i-th qubit.
    """
    z = np.diag([1, -1])
    i_1 = np.eye(2**i)
    i_2 = np.eye(2 ** (n - i - 1))

    result = np.kron(i_1, z)
    result = np.kron(result, i_2)

    return result

def _n_z(h_n: np.ndarray) -> np.ndarray:
    """Generate an n-qubit Z gate from the n-qubit Hadamard matrix.

    Args:
        h_n (np.ndarray): n-qubit Hadamard matrix.

    Returns:
        np.ndarray: The n-qubit Z gate.
    """
    res = np.diag(h_n)
    res = np.sign(res)
    res = np.diag(res)
    return res

def _modified_LHC(n: int, n_samples: int, n_div: int) -> np.ndarray:
    """Generate samples using modified Latin Hypercube Sampling.

    Args:
        n (int): Dimensionality of the data.
        n_samples (int): Number of samples to generate.
        n_div (int): Number of divisions for stratified sampling.

    Returns:
        np.ndarray: Generated samples.
    """
    samples = np.empty((n_samples, n), dtype=float)
    bin_size = 2 * np.pi / n_div
    n_passes = (n_samples + n_div - 1) // n_div

    all_bins = np.tile(np.arange(n_div), n_passes)

    for dim in range(n):
        algorithm_globals.random.shuffle(all_bins)
        chosen_bins = all_bins[:n_samples]
        offsets = algorithm_globals.random.random(n_samples)
        samples[:, dim] = (chosen_bins + offsets) * bin_size

    return samples

def _sobol_sampling(n: int, n_samples: int) -> np.ndarray:
    """Generate samples using Sobol sequence sampling.

    Args:
        n (int): Dimensionality of the data.
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated samples scaled to the interval [0, 2π].
    """
    sampler = Sobol(d=n, scramble=True)
    p = 2 * np.pi * sampler.random(n_samples)
    return p

def _phi_i(x_vecs: np.ndarray, i: int) -> np.ndarray:
    """Compute the φ_i term for a given dimension.

    Args:
        x_vecs (np.ndarray): Input sample vectors.
        i (int): Dimension index.

    Returns:
        np.ndarray: Computed φ_i values.
    """
    return x_vecs[:, i].reshape((-1, 1))

def _phi_ij(x_vecs: np.ndarray, i: int, j: int) -> np.ndarray:
    """Compute the φ_ij term for given dimensions.

    Args:
        x_vecs (np.ndarray): Input sample vectors.
        i (int): First dimension index.
        j (int): Second dimension index.

    Returns:
        np.ndarray: Computed φ_ij values.
    """
    return ((np.pi - x_vecs[:, i]) * (np.pi - x_vecs[:, j])).reshape((-1, 1))

def _random_unitary(dims):
    A = np.array(algorithm_globals.random.random((dims, dims)) 
                + 1j * algorithm_globals.random.random((dims, dims)))
    Herm = A.conj().T @ A 
    eigvals, eigvecs = np.linalg.eig(Herm)
    idx = eigvals.argsort()[::-1]
    V = eigvecs[:, idx]
    return V

def _loop_sampling(n, n_samples, z_diags, zz_diags, O, h_n, lab_fn, samp_fn):
    
    features = np.empty((n_samples, n), dtype=float)
    labels = np.empty(n_samples, dtype=int)
    dims = 2**n
    cur = 0

    while n_samples > 0:
        # Stratified Sampling for x vector
        x_vecs = samp_fn(n, n_samples)    

        # Seperable ZZFeaturemap: exp(sum j phi Zi + sum j phi Zi Zj)
        ind_pairs = zz_diags.keys()
        pre_exp = np.zeros((n_samples, dims))

        # First Order Terms
        for i in range(n):
            pre_exp += _phi_i(x_vecs, i)*z_diags[i]
        # Second Order Terms 
        for (i,j) in ind_pairs:
            pre_exp += _phi_ij(x_vecs, i, j)*zz_diags[(i,j)]
        
        # Since pre_exp is purely diagonal, exp(A) = diag(exp(Aii))
        post_exp = np.exp(1j * pre_exp)
        Uphi = np.zeros((n_samples, dims, dims), dtype = post_exp.dtype)
        cols = range(dims)
        Uphi[:,cols, cols] = post_exp[:, cols]

        Psi = (Uphi @ h_n @ Uphi @ psi_0).reshape((-1, dims, 1))

        # Labelling
        raw_labels = lab_fn(Psi)
        indx = np.abs(raw_labels)>0
        count = np.sum(indx)
        features[cur:cur+count] = x_vecs[indx]
        labels[cur:cur+count] = np.sign(raw_labels[indx])

        n_samples -= count
        cur += count

def _exp_label(Psi):
    Psi_dag = np.transpose(Psi.conj(), (0, 2, 1))
    exp_val = np.real(Psi_dag @ O @ Psi).flatten()
    labels = (np.abs(exp_val)>gap)*(np.sign(exp_val))
    return labels