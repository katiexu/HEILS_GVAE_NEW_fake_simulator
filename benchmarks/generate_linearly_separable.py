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

"""Generate datasets for the  LINEARLY SEPARABLE benchmark."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
print(f"Current path: {current_path}")
os.makedirs(current_path / "hidden_manifold", exist_ok=True)

np.random.seed(42)

os.makedirs("linearly_separable", exist_ok=True)

def generate_linearly_separable(n_samples, n_features, margin):
    """Data generation procedure for 'linearly separable'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        margin (float): width between hyperplane and closest samples
    """

    w_true = np.ones(n_features)

    # hack: sample more data than we need randomly from a hypercube
    X = 2 * np.random.rand(2 * n_samples, n_features) - 1

    # only retain data outside a margin
    X = [x for x in X if np.abs(np.dot(x, w_true)) > margin]
    X = X[:n_samples]

    y = [np.dot(x, w_true) for x in X]
    y = [-1 if y_ > 0 else 1 for y_ in y]
    return X, y

n_samples = 400

for n_features in [16, 32, 48, 64, 80]:
    margin = 0.02 * n_features

    X, y = generate_linearly_separable(n_samples, n_features, margin)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    name_train = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_train.csv"
    data_train = np.c_[X_train, y_train]
    np.savetxt(name_train, data_train, delimiter=",")

    name_val = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_val.csv"
    data_val = np.c_[X_val, y_val]
    np.savetxt(name_val, data_val, delimiter=",")

    name_test = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_test.csv"
    data_test = np.c_[X_test, y_test]
    np.savetxt(name_test, data_test, delimiter=",")

    print(f"Generated linearly separable data with {n_features} features.")