# class that extends the  VQC class from qiskit_machine_learning.algorithms.classifiers
# """
from qiskit_machine_learning.algorithms import VQC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms import VQC
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


class FairnessVQC(VQC):
    def __init__(
            self,
            training_data: StandardDataset,
            testing_data: StandardDataset,
            fairness_metric: Metric,
            sampler: BaseSampler | None = None,
            feature_map: QuantumCircuit | None = None,
            ansatz: QuantumCircuit | None = None,
            loss: str | Loss = "cross_entropy",
            optimizer: Optimizer | Minimizer | None = None,
            callback: Callable[[np.ndarray, float], None] | None = None,
            *,
            num_qubits: int | None = None,
    ) -> None:

        super().__init__(sampler=sampler, feature_map=feature_map, ansatz=ansatz, loss=loss, optimizer=optimizer, callback=callback)

        self.training_data = training_data
        self.testing_data = testing_data
        self.fairness_metric = fairness_metric

    ## Methods
    # override the implementation of the objective function to include the fairness metric
    def _objective_function(self, weights: np.ndarray) -> float:
        # compute the loss
        loss = super()._objective_function(weights)

        # compute the fairness metric
        fairness = self._compute_fairness()

        return loss + fairness

    # maybe create your own instance objective function?
    # output to CSV? or dictionary? Then can plot this data later

    def _compute_fairness(self) -> int:
        return 10
