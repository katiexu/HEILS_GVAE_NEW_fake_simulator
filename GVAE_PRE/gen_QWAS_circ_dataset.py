import os
import sys

sys.path.insert(0, os.getcwd())
import json
import tqdm
import torch
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import var_config as vc

from tqdm import tqdm
from pennylane import CircuitGraph
from pennylane import numpy as pnp
from torch.nn import functional as F
import pickle
import random
from Arguments import Arguments


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

current_path = os.getcwd()

dev = qml.device("default.qubit", wires=vc.num_qubits)


# app=1: Fidelity Task; app=2: MAXCUT; app=3: VQE
@qml.qnode(dev)
def circuit_qnode(circuit_list, app=1, hamiltonian=None, edge=None):
    for params in list(circuit_list):
        if params == 'START':
            continue
        elif params[0] == 'Identity':
            qml.Identity(wires=params[1])
        elif params[0] == 'PauliX':
            qml.PauliX(wires=params[1])
        elif params[0] == 'PauliY':
            qml.PauliY(wires=params[1])
        elif params[0] == 'PauliZ':
            qml.PauliZ(wires=params[1])
        elif params[0] == 'Hadamard':
            qml.Hadamard(wires=params[1])
        elif params[0] == 'RX':
            param = pnp.array(params[2], requires_grad=True)
            qml.RX(param, wires=params[1])
        elif params[0] == 'RY':
            param = pnp.array(params[2], requires_grad=True)
            qml.RY(param, wires=params[1])
        elif params[0] == 'RZ':
            param = pnp.array(params[2], requires_grad=True)
            qml.RZ(param, wires=params[1])
        elif params[0] == 'CNOT':
            qml.CNOT(wires=[params[1], params[2]])
        elif params[0] == 'CZ':
            qml.CZ(wires=[params[1], params[2]])
        elif params[0] == 'U3':
            theta = pnp.array(params[2], requires_grad=True)
            phi = pnp.array(params[3], requires_grad=True)
            delta = pnp.array(params[4], requires_grad=True)
            qml.U3(theta, phi, delta, wires=params[1])
        elif params[0] == 'C(U3)':
            theta = pnp.array(params[2], requires_grad=True)
            phi = pnp.array(params[3], requires_grad=True)
            delta = pnp.array(params[4], requires_grad=True)
            qml.ctrl(qml.U3, control=params[1])(theta, phi, delta, wires=params[2])
        elif params[0] == 'SWAP':
            qml.SWAP(wires=[params[1], params[2]])
        elif params == 'END':
            break
        else:
            print(params)
            raise ValueError("There exists operations not in the allowed operation pool!")

    if app == 1:
        return qml.state()
    elif app == 2:
        if edge is None:
            return qml.sample()
        if hamiltonian != None:
            return qml.expval(hamiltonian)
        else:
            raise ValueError("Please pass a hamiltonian as an observation for QAOA_MAXCUT!")
    elif app == 3:
        if hamiltonian != None:
            return qml.expval(hamiltonian)
        else:
            raise ValueError("Please pass a hamiltonian as an observation for VQE!")
    else:
        print("Note: Currently, there are no correspoding appllications!")


def translator(self, selected_single, selected_enta):
    # Translate single-qubit gates into lists
    single_columns = {i: [] for i in range(vc.num_qubits)}
    for col_index in range(1, len(selected_single[0]), 2):
        for row_index in range(len(selected_single)):
            value1 = selected_single[row_index][col_index]
            value2 = selected_single[row_index][col_index + 1]
            combined = f"{value1}{value2}"
            if combined == '00':
                single_columns[(col_index - 1) / 2].append(('Identity', row_index))
            elif combined == '01':
                # angle = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                delta = np.random.uniform(0, 2 * np.pi)
                single_columns[(col_index - 1) / 2].append(('U3', row_index, theta, phi, delta))
            elif combined == '10':
                angle = np.random.uniform(0, 2 * np.pi)
                single_columns[(col_index - 1) / 2].append(('data', row_index, angle))
            elif combined == '11':
                angle = np.random.uniform(0, 2 * np.pi)
                single_columns[(col_index - 1) / 2].append(('data+U3', row_index, angle))
            else:
                pass

    # Translate entangled gates into lists
    enta_columns = {i: [] for i in range(len(selected_enta[0]) - 1)}
    for col_index in range(1, len(selected_enta[0])):
        for row_index in range(len(selected_enta)):
            control = row_index
            target = selected_enta[row_index][col_index] - 1
            if control == target:
                enta_columns[col_index - 1].append(('Identity', target))
            else:
                if col_index - 1 in enta_columns:
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    delta = np.random.uniform(0, 2 * np.pi)
                    enta_columns[col_index - 1].append(('C(U3)', control, target, theta, phi, delta))

    
    # Re-order gate lists and generate final design
    circuit_ops = []
    for layer in range(self.num_layers):
        circuit_ops.extend(single_columns[layer])
        circuit_ops.extend(enta_columns[layer])

    return circuit_ops

class CircuitManager:

    # class constructor
    def __init__(self, num_qubits, num_circuits, num_layers, allowed_gates):
        self.num_qubits = num_qubits
        self.num_circuits = num_circuits
        self.num_layers = num_layers
        self.allowed_gates = allowed_gates
        self.pbar = tqdm(range(self.num_circuits), desc="generated_num_circuits")

    # encode allowed gates in one-hot encoding
    def encode_gate_type(self):
        gate_dict = {}
        ops = self.allowed_gates.copy()
        ops.remove('C(U3)')
        # ops.insert(0, 'START')
        # ops.append('END')
        ops_len = len(ops)
        ops_index = torch.tensor(range(ops_len))
        type_onehot = F.one_hot(ops_index, num_classes=ops_len)
        for i in range(ops_len):
            gate_dict[ops[i]] = type_onehot[i]
        return gate_dict

    # Circuit generator function
    def generate_circuits(self):
        unique_circuits = []

        with open(args.file_single, 'rb') as file:
            search_space_single = pickle.load(file)
        with open(args.file_enta, 'rb') as file:
            search_space_enta = pickle.load(file)

        def filter_start_with(search_space, start_value):
            return [lst for lst in search_space if lst[0] == start_value]

        def generate_QWAS_circuits():
            selected_single = []
            selected_enta = []

            for start_value in range(1, self.num_qubits + 1):
                candidates_single = filter_start_with(search_space_single, start_value)
                selected_single.append(random.sample(candidates_single, 1)[0])

                candidates_enta = filter_start_with(search_space_enta, start_value)
                selected_enta.append(random.sample(candidates_enta, 1)[0])

            circuit_ops = translator(self, selected_single, selected_enta)

            return circuit_ops

        while len(unique_circuits) < self.num_circuits:
            circuit_ops = generate_QWAS_circuits()
            if circuit_ops == None:
                continue
            # if not set(circuit_ops).issubset(set(unique_circuits)):
            unique_circuits.append(tuple(circuit_ops))
            self.pbar.update(1)

        return unique_circuits



    # transform a circuit into a circuit graph
    def get_circuit_graph(self, circuit_list):
        circuit_qnode(circuit_list)
        tape = circuit_qnode.qtape
        ops = tape.operations
        obs = tape.observables
        return CircuitGraph(ops, obs, tape.wires)   


    def get_wires(self,op):
        if op[0] == 'C(U3)':
            return [op[1], op[2]]
        else:
            return [op[1]]

    def get_gate_and_adj_matrix(self, circuit_list):        
        
        n_qubits = self.num_qubits
        n_layers = self.num_layers
        interval = 2 * n_qubits
        gate_matrix = []
        op_list = []
        cl = list(circuit_list).copy()
        
        gate_dict = self.encode_gate_type()
        single_gate_type = len(gate_dict)
        
        for i in range(n_layers):
            cu3gate=[[0 for j in range(single_gate_type+n_qubits)] for i in range(n_qubits)]
        # op_list.append('START')
            for op in circuit_list[i*interval:i*interval+n_qubits]:
                op_qubits = [0] * n_qubits
                op_vector = gate_dict[op[0]].tolist() + op_qubits
                gate_matrix.append(op_vector)

            for op in circuit_list[i*interval+n_qubits:(i+1)*interval]:
                op_wires = self.get_wires(op)
                if len(op_wires) > 1:
                    i,j=op_wires
                    cu3gate[i][j+single_gate_type]=1
            gate_matrix.extend(cu3gate)
               
        op_len = len(circuit_list)
        adj_matrix = np.zeros((op_len, op_len), dtype=float)
        
        for index, op in enumerate(circuit_list):
            op_wires = self.get_wires(op)
            if not (index % interval >= n_qubits and len(op_wires) == 1):
                for wire_idx, wire in enumerate(op_wires):
                    for other_index, other_op in enumerate(circuit_list[index + 1:]):
                        other_index = index + 1 + other_index
                        other_wires = self.get_wires(other_op)
                        if not (other_index % interval >= n_qubits and len(other_wires) == 1):
                            if wire in other_wires:
                                # 根据 wire 在 op_wires 中的位置设置值
                                adj_matrix[index, other_index] = (wire_idx + 1) / 2  # 第0位 -> 1，第1位 -> 2
                                break
        pass       

        return cl, gate_matrix, adj_matrix



    @property
    def get_num_qubits(self):
        return self.num_qubits

    @property
    def get_num_circuits(self):
        return self.num_circuits

    @property
    def get_num_gates(self):
        return self.num_gates

    @property
    def get_max_depth(self):
        return self.max_depth


# dump circuit features in json file
def data_dumper(circuit_manager: CircuitManager, f_name: str = 'data.json'):
    """dump circuit DAG features."""
    circuit_features = []
    # file_path = os.path.join(current_path, f'circuit\\data\\{f_name}')
    file_path = os.path.join(current_path, f'data/{f_name}')
    for i in range(circuit_manager.get_num_circuits):
        op_list, gate_matrix, adj_matrix = circuit_manager.get_gate_and_adj_matrix(circuits[i])
        circuit_features.append({'op_list': op_list, 'gate_matrix': gate_matrix, 'adj_matrix': adj_matrix.tolist()})
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(circuit_features, file)


if __name__ == '__main__':

    task = {
    'task': 'MNIST_10',
    'option': 'explicit_reg',
    'n_qubits': 5,
    'n_layers': 4,
    'fold': 1
    }
    args = Arguments(**task)

    num_circuits = 1000
    num_qubits = args.n_qubits
    num_layers = args.n_layers
    allowed_gates = ['Identity', 'U3', 'data', 'data+U3', 'C(U3)']

    circuit_manager = CircuitManager(num_qubits, num_circuits, num_layers, allowed_gates)
    circuits = circuit_manager.generate_circuits()
    print("Number of unique circuits generated:", len(circuits))
    # print("The first curcuit list: ", circuits[0])
    op_list, gate_matrix, adj_matrix = circuit_manager.get_gate_and_adj_matrix(circuits[1])
    # print("The first curcuit info: ")
    # print("op_list: ", op_list)
    # print("gate_matrix\n", gate_matrix)
    # print("adj_matrx: \n", adj_matrix)
    # fig, ax = qml.draw_mpl(circuit_qnode)(circuits[0])
    # plt.show()
    data_dumper(circuit_manager, f_name=f"data_{task['n_qubits']}_qubits_1000.json")