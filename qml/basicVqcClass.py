from qiskit_machine_learning.algorithms import VQC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.algorithms.classifiers import VQC
from aif360.metrics import ClassificationMetric
from aif360.metrics import Metric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, Minimizer
from typing import Callable
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import derive_num_qubits_feature_map_ansatz
from qiskit_machine_learning.utils.loss_functions import Loss

# Create basic class
class BasicClass(VQC):
    def num_qubits(self) -> int:
        print("This is the BasicClass VQC")
        return self.circuit.num_qubits

