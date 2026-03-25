from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf

from FusionModel import single_enta_to_design
from GVAE_translator import GVAE_translator, get_gate_and_adj_matrix
from schemes import Scheme
from Arguments import Arguments

DATA_ROOT = Path(__file__).resolve().parent / "data"
RANDOM_CIRCUITS_FILE = DATA_ROOT / "random_circuits_mnist_4_10000.json"


def load_random_circuits(path: Path = RANDOM_CIRCUITS_FILE) -> Iterable[Any]:
    """Load all circuit specifications from the random MNIST dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Random circuits file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_circuit_spec(spec: dict[str, Any]) -> Tuple[List[List[int]], List[List[int]], List[List[int]], Tuple[int, int]]:
    single_rows = sorted(spec["single"], key=lambda row: row[0])
    enta_rows = sorted(spec["enta"], key=lambda row: row[0])

    if not single_rows:
        raise ValueError("Circuit specification contains no single-qubit layers")

    tail_width = len(single_rows[0]) - 1
    if tail_width % 2 != 0:
        raise ValueError("Unexpected single layer encoding width")

    n_qubits = tail_width // 2
    n_layers = len(single_rows)

    data_uploading: List[List[int]] = []
    rotation: List[List[int]] = []
    entanglers: List[List[int]] = []

    for row in single_rows:
        if len(row) != 1 + 2 * n_qubits:
            raise ValueError("Inconsistent single layer encoding width")
        data_uploading.append([int(bit) for bit in row[1 : 1 + n_qubits]])
        rotation.append([int(bit) for bit in row[1 + n_qubits :]])

    for row in enta_rows:
        if len(row) != 1 + n_qubits:
            raise ValueError("Inconsistent entangling layer encoding width")
        entanglers.append([int(target) for target in row[1:]])

    arch_code = (n_qubits, n_layers)
    return data_uploading, rotation, entanglers, arch_code


def circuit_to_matrices(spec: dict[str, Any]) -> Tuple[Sequence[Any], List[List[int]], Any, Tuple[int, int]]:
    data_uploading, rotation, entanglers, arch_code = _parse_circuit_spec(spec)
    circuit_ops = GVAE_translator(data_uploading, rotation, entanglers, arch_code)
    circuit_list, gate_matrix, adj_matrix = get_gate_and_adj_matrix(circuit_ops, arch_code)
    return circuit_list, gate_matrix, adj_matrix, arch_code


def spec_to_design(spec: dict[str, Any]) -> Tuple[List[Tuple[str, List[int], int]], Tuple[int, int]]:
    """Convert a raw circuit specification into a TorchQuantum design list."""

    single_rows = sorted(spec["single"], key=lambda row: row[0])
    enta_rows = sorted(spec["enta"], key=lambda row: row[0])

    tail_width = len(single_rows[0]) - 1
    if tail_width % 2 != 0:
        raise ValueError("Unexpected single layer encoding width")

    n_qubits = tail_width // 2
    n_layers = len(single_rows)
    arch_code = (n_qubits, n_layers)
    design = single_enta_to_design(single_rows, enta_rows, arch_code)
    return design, arch_code


def add_virtual_start_node(adj_matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Augment the adjacency matrix with a virtual start node."""

    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    node_count = adj_matrix.shape[0]
    augmented = np.zeros((node_count + 1, node_count + 1), dtype=float)
    augmented[1:, 1:] = adj_matrix

    in_degrees = np.count_nonzero(adj_matrix, axis=0)
    entry_nodes = [idx for idx, degree in enumerate(in_degrees) if degree == 0]

    if entry_nodes:
        augmented[0, [idx + 1 for idx in entry_nodes]] = 1.0

    return augmented, entry_nodes


def add_virtual_terminals(adj_matrix: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """Attach virtual start/end nodes so DAG paths have explicit terminals."""

    augmented, entry_nodes = add_virtual_start_node(adj_matrix)
    # Pad once more to host the terminal sink node.
    augmented = np.pad(augmented, ((0, 1), (0, 1)), mode="constant")

    out_degrees = np.count_nonzero(adj_matrix, axis=1)
    exit_nodes = [idx for idx, degree in enumerate(out_degrees) if degree == 0]
    if not exit_nodes:
        exit_nodes = [adj_matrix.shape[0] - 1]

    for node_idx in exit_nodes:
        augmented[node_idx + 1, -1] = 1.0

    return augmented, entry_nodes, exit_nodes


def path_based_proxy(adj_matrix: np.ndarray) -> int:
    """Return the path-based proxy defined as the DAG path count."""

    augmented, _, _ = add_virtual_terminals(adj_matrix)
    edge_mask = augmented > 0
    path_counts = np.zeros(augmented.shape[0], dtype=np.int64)
    path_counts[0] = 1

    for node in range(augmented.shape[0]):
        successors = np.nonzero(edge_mask[node])[0]
        if successors.size == 0:
            continue
        path_counts[successors] += path_counts[node]

    return int(path_counts[-1])


def _apply_design_with_random_params(
    design: Sequence[Tuple[str, Sequence[int], int]],
    n_qubits: int,
    device: torch.device,
) -> torch.Tensor:
    """Simulate the circuit once with fresh random parameters and return the state."""

    qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=1, device=device)

    for gate_name, wires, _ in design:
        if gate_name == "U3":
            params = torch.rand(1, 3, device=device) * 2 * np.pi
            tqf.u3(qdev, wires=wires, params=params)
        elif gate_name == "C(U3)":
            params = torch.rand(1, 3, device=device) * 2 * np.pi
            tqf.cu3(qdev, wires=wires, params=params)
        else:
            # Treat data-uploading or identity-like operations as no-ops for expressibility.
            continue

    state = qdev.get_states_1d()[0]
    return state / state.norm(dim=-1, keepdim=True)


def expressibility_proxy(
    design: Sequence[Tuple[str, Sequence[int], int]],
    arch_code: Tuple[int, int],
    num_parameter_samples: int = 128,
    histogram_bins: int = 75,
    device: str | torch.device = "cpu",
) -> float:
    """Estimate expressibility via KL divergence between circuit and Haar fidelities."""

    n_qubits, _ = arch_code
    dimension = 2 ** n_qubits
    torch_device = torch.device(device)

    states = [
        _apply_design_with_random_params(design, n_qubits, torch_device)
        for _ in range(num_parameter_samples)
    ]
    states_tensor = torch.stack(states)

    overlaps = torch.matmul(states_tensor, states_tensor.conj().T)
    fidelities = torch.abs(overlaps) ** 2
    tri = torch.triu_indices(fidelities.shape[0], fidelities.shape[0], offset=1)
    sampled_fidelities = fidelities[tri[0], tri[1]].real.cpu().numpy()

    if sampled_fidelities.size == 0:
        return float("inf")

    bins = min(histogram_bins, max(10, sampled_fidelities.size // 10))
    hist, bin_edges = np.histogram(sampled_fidelities, bins=bins, range=(0.0, 1.0), density=False)
    if not hist.any():
        return float("inf")

    prob = hist.astype(np.float64)
    prob /= prob.sum()

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2
    haar_density = (dimension - 1) * np.maximum(1 - bin_centers, 1e-12) ** (dimension - 2)
    haar_prob = haar_density * bin_width
    haar_prob /= haar_prob.sum()

    eps = 1e-12
    mask = prob > 0
    divergence = np.sum(prob[mask] * (np.log(prob[mask] + eps) - np.log(haar_prob[mask] + eps)))
    return float(divergence)


if __name__ == "__main__":
    circuits = load_random_circuits()
    total = len(circuits)
    print(f"Loaded {total} circuits from {RANDOM_CIRCUITS_FILE}")

    # path_scored: List[Tuple[int, dict[str, Any], Tuple[int, int], int]] = []
    # for idx, circuit_spec in enumerate(circuits):
    #     _, _, adj, arch = circuit_to_matrices(circuit_spec)
    #     path_value = path_based_proxy(np.array(adj))
    #     path_scored.append((idx, circuit_spec, arch, path_value))

    # path_scored.sort(key=lambda item: item[3], reverse=True)
    # top_path = path_scored[: min(100, len(path_scored))]
    # print(f"Selected top {len(top_path)} circuits by path-based proxy.")

    # express_scored: List[Tuple[int, dict[str, Any], Tuple[int, int], int, float]] = []
    # for idx, circuit_spec, arch, path_value in top_path:
    #     design, design_arch = spec_to_design(circuit_spec)
    #     express_value = expressibility_proxy(design, design_arch)
    #     express_scored.append((idx, circuit_spec, arch, path_value, express_value))

    # express_scored.sort(key=lambda item: item[4])
    # top_express = express_scored[: min(30, len(express_scored))]
    # print(f"Selected top {len(top_express)} circuits by expressibility proxy.")
    # output_file = Path("data/top_express.json")
    # with output_file.open("w") as f:
    #     json.dump(top_express, f)
    # print(f"Saved top_express to {output_file}")

    with Path("data/top_express.json").open("r") as f:
        top_express = json.load(f)

    top_express_designs: List[Tuple[List[List[int]], List[List[int]], Tuple[int, int]]] = []

    for rank, (idx, circuit_spec, arch, path_value, express_value) in enumerate(top_express, start=1):
        single_rows = sorted(circuit_spec["single"], key=lambda row: row[0])
        enta_rows = sorted(circuit_spec["enta"], key=lambda row: row[0])
        top_express_designs.append((single_rows, enta_rows, arch))        
        print(
            f"Rank {rank:02d} | Circuit {idx}: qubits={arch[0]}, layers={arch[1]}, "
            f"path proxy={path_value}, expressibility={express_value:.4f}"
        )

    task = {
            'task': 'QML_Hidden_48d',
            'n_qubits': 12,
            'n_layers': 4,
            'fold': 3,
            'option': 'mix_reg',
            'regular': True,
            'num_processes': 2
            }    
    args = Arguments(**task)
    arch_code = [task['n_qubits'], task['n_layers']]

    output_csv = Path(f"results/{task['task']}_training_free.csv")
    with output_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["rank", "single", "enta", "test_accuracy"],
        )
        writer.writeheader()

    for rank, (single_rows, enta_rows, arch) in enumerate(top_express_designs, start=1):
        print(f"Evaluating design {rank}")
        design = single_enta_to_design(single_rows, enta_rows, arch_code, args.fold)
        _, report = Scheme(design, task, 'init', 30, verbs=False)
        test_acc = float(report.get('mae', 0.0))

        with output_csv.open("a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["rank", "single", "enta", "test_accuracy"],
            )
            writer.writerow(
                {
                    "rank": rank,
                    "single": single_rows,
                    "enta": enta_rows,
                    "test_accuracy": test_acc,
                }
            )

    print(f"Saved Scheme evaluations to {output_csv}")
        
