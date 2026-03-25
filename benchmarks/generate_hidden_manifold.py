# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate datasets for the HIDDEN MANIFOLD and HIDDEN MANIFOLD DIFF benchmarks."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
current_path = pathlib.Path(__file__).parent.absolute()
print(f"Current path: {current_path}")

np.random.seed(3)
os.makedirs(current_path / "hidden_manifold", exist_ok=True)

def neural_net(x, W, v):
    """Transforms inputs via a single-layer neural network.
    Args:
        x (ndarray): input of shape (manifold_dimension,)
        W (ndarray): input-to-hidden weight matrix of shape (manifold_dimension, manifold_dimension)
        v (ndarray): hidden-to-output weight matrix of shape (manifold_dimension,)
    """
    return np.dot(v, np.tanh(W @ x) / np.sqrt(W.shape[0]))


def nonlinearity(X, biases):
    """Element-wise nonlinearity.

    Args:
        X (ndarray): inputs of shape (n_samples, n_features)
        biases (ndarray): biases of shape (n_features,)
    """
    return np.tanh(X - biases)


def generate_hidden_manifold_model(n_samples, n_features, manifold_dimension):
    """Data generation procedure for the 'hidden manifold model'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        manifold_dimension (int): dimension of hidden maniforls
    """

    # feature matrix F controls the embedding of the manifold
    F = np.random.normal(size=(manifold_dimension, n_features))

    # Gaussian matrix samples original inputs from the lower-dimensional manifold
    C = np.random.normal(size=(n_samples, manifold_dimension), loc=0, scale=1)

    # embed data, adding an element-wise nonlinearity
    biases = 2 * np.random.uniform(size=(n_features,)) - 1
    X = nonlinearity(C @ F / np.sqrt(manifold_dimension), biases)

    # define labels via a neural network
    W = np.random.normal(size=(manifold_dimension, manifold_dimension))
    v = np.random.normal(size=(manifold_dimension,))
    y = np.array([neural_net(c, W, v) for c in C])

    # post-process the labels to get balanced classes
    y = y - np.median(y)
    y = np.array([-1 if y_ < 0 else 1 for y_ in y])
    assert len(X[y == 1]) == n_samples // 2
    assert len(X[y == -1]) == n_samples // 2

    return X, y

manifold_dimension = 6
n_samples = 400

for n_features in [16, 32, 48, 64]:
    X, y = generate_hidden_manifold_model(n_samples, n_features, manifold_dimension)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    # name_train = f"{current_path}/hidden_manifold/hidden_manifold-{manifold_dimension}manifold-{n_features}d_train.csv"
    name_train = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_val = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_val.csv"
    data_val = np.c_[X_val, y_val]
    np.savetxt(name_val, data_val, delimiter=",")

    name_test = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")



