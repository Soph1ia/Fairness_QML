import qiskit
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap, ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit.circuit.library import EfficientSU2
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.metrics import accuracy_score
import os
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def load_data():
    """
    Method that loads in the data from aif360/Data/ folder
    returns
    DataFrame
    """
    data_df = pd.read_csv("../aif360/Data/fake_data.csv")
    data_df.head()
    return data_df


def preprocess_data(df):
    """
    Method to remap the data to numerical values and split the data into train and test
    """
    # remap gender to binary
    gender_mapping = {'male': 0, 'female': 1}

    # replace values in gender column
    df['gender'].replace(gender_mapping, inplace=True)

    # remap the car column to binary
    car_mapping = {'yes': 0, 'no': 1}

    # replace values in car column
    df['car'].replace(car_mapping, inplace=True)

    # spit the data into y and x 
    y = df['target']
    X = df.drop(columns=['target'])

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def callback_graph(weights, obj_func_eval):
    """
    method to get the objective funtion values and display them as a graph
    """
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


def create_vqc_model(X_train):
    """
    Method that creates a FeatureMap
    initialises a sampler and optimizer
    creates a VQC modul

    returns
    vqc
    """
    num_features = X_train.shape[1]

    zz_feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    zz_feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

    sampler = Sampler()

    optimizer_c = COBYLA(maxiter=100)  # optimizer

    my_vqc = VQC(
        sampler=sampler,
        feature_map=zz_feature_map,
        ansatz=EfficientSU2(num_qubits=num_features, reps=1),
        optimizer=optimizer_c,
        callback=callback_graph
    )

    # my_vqc.fit()

    return my_vqc


def train_vqc_model(vqc, X_train, y_train):
    # clear objective value history
    objective_func_vals = []

    start = time.time()
    vqc.fit(X_train, y_train.values.reshape(-1, 1))
    elapsed = time.time() - start

    print(f"Training time: {round(elapsed)} seconds")


def test_trained_model_on_fairness(vqc_model, entire dataset):
    """
    This method needs to return how fair a model is
    TODO implement this method
    """
    y_pred = vqc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Prediction accuracy: {accuracy}")

    # copy of dataset_all
    dataset_prediction = dataset_all.copy()
    dataset_prediction.labels = y_pred.reshape(-1, 1)

    # Classification metrics
    metric_prediction = ClassificationMetric(dataset_all,
    dataset_prediction,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

# main
if __name__ == "__main__":
    print("qiskit version is: ", qiskit.__version__)
    print(os.getcwd())

    fake_data_df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(fake_data_df)

    created_vqc = create_vqc_model(X_train)

    print(created_vqc.circuit.num_qubits)
    train_vqc_model(created_vqc, X_train, y_train)
