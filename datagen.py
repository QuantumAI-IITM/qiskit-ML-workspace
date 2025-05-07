import numpy as np
import os
from typing import List, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class HardwareEfficientDatasetGenerator:
    def __init__(self, base_path: str = "Hardware_Efficient"):
        self.base_path = base_path
        self.supported_qubits = [3, 4, 8]
        self.supported_depths_34 = list(range(1, 7))
        self.supported_depths_8 = [5, 6]
        self.supported_ce_34 = [0.05, 0.15, 0.25, 0.35]
        self.supported_ce_8_6 = [0.10, 0.25]
        self.supported_ce_8_5 = [0.15, 0.40, 0.45]

    def validate_params(self, qubits: int, depth: int, goal_ce: float) -> bool:
        # Validate inputs
        if qubits not in self.supported_qubits:
            raise ValueError(f"Unsupported qubit count: {qubits}. Choose from {self.supported_qubits}")
        
        if (qubits == 8):
            if depth not in self.supported_depths_8:
                raise ValueError(f"Unsupported depth: {depth}. Choose from {self.supported_depths_8}")
            if (depth == 6):
                if goal_ce not in self.supported_ce_8_6:
                    raise ValueError(f"Unsupported CE value: {goal_ce}. Choose from {self.supported_ce_8_6}")
            else:
                if goal_ce not in self.supported_ce_8_5:
                    raise ValueError(f"Unsupported CE value: {goal_ce}. Choose from {self.supported_ce_8_5}")
        else:
            if depth not in self.supported_depths_34:
                raise ValueError(f"Unsupported depth: {depth}. Choose from {self.supported_depths_34}")
            if goal_ce not in self.supported_ce_34:
                raise ValueError(f"Unsupported CE value: {goal_ce}. Choose from {self.supported_ce_34}")
        
        return True

    def _construct_filepath(self, qubits: int, depth: int, goal_ce: float) -> str:
        """
        Construct the full file path based on parameters and directory structure.
        
        Args:
            qubits: Number of qubits (3 or 4)
            depth: Circuit depth (1-6)
            goal_ce: Target concentratable entanglement (0.05, 0.15, 0.25, 0.35, 0.5)
            
        Returns:
            Full path to .npy weights file
        """         

        self.validate_params(qubits, depth, goal_ce)

        # Validate goal_ce format
        ce_str = f"{int(goal_ce * 100):02d}"
        
        # Construct file path
        qubit_dir = f"{qubits}_Qubits"
        depth_dir = f"Depth_{depth}"
        filename = f"hwe_{qubits}q_ps_{ce_str}_{depth}_weights.npy"
        
        return os.path.join(self.base_path, qubit_dir, depth_dir, filename)

    def load_weights(self, qubits: int, depth: int, goal_ce: float) -> np.ndarray:
        """
        Load weights from the repository.
        
        Args:
            qubits: Number of qubits
            input_type: Type of input states
            goal_ce: Goal concentratable entanglement (0-1)
            depth: Circuit depth
            
        Returns:
            Weights as numpy array
        """
        file_path = self._construct_filepath(qubits, depth, goal_ce)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Weights file not found at: {file_path}")
            
        return np.load(file_path)

    def hardware_efficient_ansatz(self, qc, params, qubits, depth):
        """Adds hardware-efficient ansatz to a QuantumCircuit"""
        param_idx = 0
        for d in range(depth):
            # Single-qubit rotations
            for q in range(qubits):
                qc.rx(params[param_idx], q)
                qc.ry(params[param_idx+1], q)
                qc.rz(params[param_idx+2], q)
                param_idx += 3
            
            # Entangling layer
            for q in range(qubits - 1):
                qc.cx(q, q+1)
            if qubits > 1:
                qc.cx(qubits-1, 0)

    def generate_states(self, qubits: int, depth: int, goal_ce: float, 
                       num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        weights = self.load_weights(qubits, depth, goal_ce)
        simulator = Aer.get_backend('aer_simulator')

        # Validate parameter count
        expected_params = depth * qubits * 3
        if len(weights.flatten()) != expected_params:
            raise ValueError(f"Parameter mismatch: {len(weights.flatten())} vs {expected_params}")
        
        input_states = []
        output_states = []
        
        for _ in range(num_samples):
            input_state = np.random.randint(0, 2**qubits)
            qc = QuantumCircuit(qubits)
            
            # Initialize state
            for q in range(qubits):
                if (input_state >> q) & 1:
                    qc.x(q)
            
            # Create parameterized circuit
            params = ParameterVector('θ', depth * qubits * 3)
            self.hardware_efficient_ansatz(qc, params, qubits, depth)
            
            # Bind loaded weights
            bound_qc = qc.assign_parameters(weights.flatten())
            
            # Simulate
            bound_qc.save_statevector()
            result = simulator.run(bound_qc).result()
            statevector = result.get_statevector()
            
            input_states.append(input_state)
            output_states.append(statevector.data)
        
        return np.array(input_states), np.array(output_states)

    def save_dataset(self, input_states: np.ndarray, output_states: np.ndarray, 
                    save_dir: str, filename_prefix: str):
        """
        Save generated dataset to files.
        
        Args:
            input_states: Array of input computational basis states
            output_states: Array of output quantum states
            save_dir: Directory to save files
            filename_prefix: Prefix for output files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save input states
        np.save(os.path.join(save_dir, f"{filename_prefix}_inputs.npy"), input_states)
        
        # Save output states
        np.save(os.path.join(save_dir, f"{filename_prefix}_outputs.npy"), output_states)

    def create_classification_dataset(self, 
                                    entanglement_levels: List[float] = [0.15, 0.35, 0.45],
                                    depths: int | list[int] = 3,
                                    num_samples: int = 3000,
                                    qubits: int = 4,
                                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a classified dataset of quantum states with different entanglement levels.
        
        Args:
            entanglement_levels: List of target CE values for different classes
            depths: Single depth for all classes or list of depths per class
            num_samples: Total number of samples in the dataset
            qubits: Number of qubits in generated states
            random_state: Seed for reproducible shuffling
            
        Returns:
            X: Array of quantum states (num_samples, 2**qubits)
            y: Array of class labels (num_samples,)
            
        Raises:
            ValueError: For invalid input combinations
        """
        # Input validation
        if isinstance(depths, int):
            depths = [depths] * len(entanglement_levels)
        elif len(depths) != len(entanglement_levels):
            raise ValueError("Length of depths must match entanglement_levels")
            
        for ce_level, depth in zip(entanglement_levels, depths):
            self.validate_params(qubits, depth, ce_level)

        # Calculate samples per class with remainder distribution
        samples_per_class, remainder = divmod(num_samples, len(entanglement_levels))
        class_samples = [samples_per_class + (1 if i < remainder else 0) 
                        for i in range(len(entanglement_levels))]

        # Generate states for each class
        X, y = [], []
        for class_idx, (ce_level, depth, n_samples) in enumerate(zip(entanglement_levels, 
                                                                depths, 
                                                                class_samples)):
            print(f"Generating class {class_idx+1}/{len(entanglement_levels)}: "
                f"CE={ce_level}, depth={depth}, samples={n_samples}")
            
            _, states = self.generate_states(
                qubits=qubits,
                depth=depth,
                goal_ce=ce_level,
                num_samples=n_samples
            )
            
            X.append(states)
            y.append(np.full(n_samples, class_idx))

        # Combine and shuffle
        X = np.concatenate(X)
        y = np.concatenate(y)
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(X))
        
        return X[indices], y[indices]

# Example usage in the __main__ block
if __name__ == "__main__":
    generator = HardwareEfficientDatasetGenerator(base_path="Hardware_Efficient")

    # Classification dataset example
    print("\nGenerating classification dataset:")
    qubits = 3
    X, y = generator.create_classification_dataset(
        entanglement_levels=[0.15, 0.35],
        depths=[3, 5],
        qubits=qubits,
        num_samples=1000
    )
    
    print("\nClassification dataset stats:")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"State shape: {X[0].shape}") # 2**qubits 
    print(f"First state:\n{X[0]}")
    print(f"Class label: {y[0]}")
    
    # Save classification dataset
    generator.save_dataset(
        input_states=X,  # Dummy inputs since we're using class labels
        output_states=y,
        save_dir="classification_datasets",
        filename_prefix=f"{qubits}q_ce_classification"
    )
