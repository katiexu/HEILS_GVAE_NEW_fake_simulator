import torch
import numpy as np
from torch.nn import functional as F


allowed_gates = ['Identity', 'U3', 'data', 'data+U3']

def GVAE_translator(data_uploading, rot, enta, arch_code):

    single_list = []
    enta_list = []
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    for i in range(0, n_layers):
        single_item = []
        for j in range(0, n_qubits):
            d = int(data_uploading[i][j])
            r = int(rot[i][j])
            combination = f'{d}{r}'
            if combination == '00':
                single_item.append(('Identity', j))
            elif combination == '01':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('U3', j, angle))
            elif combination == '10':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('data', j, angle))
            elif combination == '11':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('data+U3', j, angle))
        single_list.append(single_item)

        enta_item = []
        for j, et in enumerate(enta[i]):
            if j == int(et) - 1:
                enta_item.append(('Identity', j))
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                delta = np.random.uniform(0, 2 * np.pi)
                enta_item.append(('C(U3)', j, int(et) - 1, theta, phi, delta))
        enta_list.append(enta_item)

    circuit_ops = []
    for layer in range(0, n_layers):
        circuit_ops.extend(single_list[layer])
        circuit_ops.extend(enta_list[layer])

    return circuit_ops

def generate_circuits(net, arch_code):
    data_uploading = []
    rot = []
    enta = []

    for i in range(0, len(net), 3):
        data_uploading.append(net[i])
        rot.append(net[i + 1])
        enta.append(net[i + 2])

    circuit_ops = GVAE_translator(data_uploading, rot, enta, arch_code)

    return circuit_ops

# encode allowed gates in one-hot encoding
def encode_gate_type():
    gate_dict = {}
    ops = allowed_gates    
    ops_len = len(ops)
    ops_index = torch.tensor(range(ops_len))
    type_onehot = F.one_hot(ops_index, num_classes=ops_len)
    for i in range(ops_len):
        gate_dict[ops[i]] = type_onehot[i]
    return gate_dict

def get_wires(op):
    if op[0] == 'C(U3)':
        return [op[1], op[2]]
    else:
        return [op[1]]  
    
def get_gate_and_adj_matrix(circuit_list, arch_code):

        n_qubits = arch_code[0]
        n_layers = arch_code[1]
        interval = 2 * n_qubits
        gate_matrix = []
        op_list = []
        cl = list(circuit_list).copy()
        
        gate_dict = encode_gate_type()
        single_gate_type = len(gate_dict)
        
        for i in range(n_layers):
            cu3gate=[[0 for j in range(single_gate_type+n_qubits)] for i in range(n_qubits)]
        # op_list.append('START')
            for op in circuit_list[i*interval:i*interval+n_qubits]:
                op_qubits = [0] * n_qubits
                op_vector = gate_dict[op[0]].tolist() + op_qubits
                gate_matrix.append(op_vector)

            for op in circuit_list[i*interval+n_qubits:(i+1)*interval]:
                op_wires = get_wires(op)
                if len(op_wires) > 1:
                    i,j=op_wires
                    cu3gate[i][j+single_gate_type]=1
            gate_matrix.extend(cu3gate)
               
        op_len = len(circuit_list)
        adj_matrix = np.zeros((op_len, op_len), dtype=float)
        
        for index, op in enumerate(circuit_list):
            op_wires = get_wires(op)
            if not (index % interval >= n_qubits and len(op_wires) == 1):
                for wire_idx, wire in enumerate(op_wires):
                    for other_index, other_op in enumerate(circuit_list[index + 1:]):
                        other_index = index + 1 + other_index
                        other_wires = get_wires(other_op)
                        if not (other_index % interval >= n_qubits and len(other_wires) == 1):
                            if wire in other_wires:
                                # 根据 wire 在 op_wires 中的位置设置值
                                adj_matrix[index, other_index] = (wire_idx + 1) / 2  # 第0位 -> 1，第1位 -> 2
                                break
        pass       

        return cl, gate_matrix, adj_matrix