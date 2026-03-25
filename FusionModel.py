import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from qiskit_aer import QasmSimulator, AerSimulator
from torchquantum.encoding import encoder_op_list_name_dict
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.primitives import Estimator
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeKolkata, FakeNairobi, FakeMontreal, FakeToronto
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import BackendEstimator


# PennyLane imports
import pennylane as qml
from tqdm import tqdm
from Arguments import Arguments  # Only for setting qml.device()
from datasets import MNISTDataLoaders


def gen_arch(change_code, base_code):  # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]
    arch_code = ([i for i in range(2, n_qubits + 1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code


def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:, 0] - 1
        change_code = change_code.reshape(-1, length)
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1, 0)
            j += 1
    return single_dict


def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # number of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits] - 1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits]) - 1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design


def single_enta_to_design(single, enta, arch_code, fold=1):
    """
    Generate a design list usable by QNET from single and enta codes

    Args:
        single: Single-qubit gate encoding, format: [[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                Each two bits of gate_config represent a layer: 00=Identity, 01=U3, 10=data, 11=data+U3
        enta: Two-qubit gate encoding, format: [[qubit, target_layer0, target_layer1, ...], ...]
              Each value represents the target qubit position in that layer
        arch_code_fold: [n_qubits, n_layers]

    Returns:
        design: List containing quantum circuit design info, each element is (gate_type, [wire_indices], layer)
    """
    design = []
    single = qubit_fold(single, 0, fold)
    enta = qubit_fold(enta, 1, fold)

    n_qubits, n_layers = arch_code

    # Process each layer
    for layer in range(n_layers):
        # First process single-qubit gates
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The config for each layer is at position: 1 + layer*2 and 1 + layer*2 + 1
            config_start_idx = 1 + layer * 2
            if config_start_idx + 1 < len(qubit_config):
                gate_config = f"{qubit_config[config_start_idx]}{qubit_config[config_start_idx + 1]}"

                if gate_config == '01':  # U3
                    design.append(('U3', [qubit], layer))
                elif gate_config == '10':  # data
                    design.append(('data', [qubit], layer))
                elif gate_config == '11':  # data+U3
                    design.append(('data', [qubit], layer))
                    design.append(('U3', [qubit], layer))
                # 00 (Identity) skip

        # Then process two-qubit gates
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The target qubit position in the list: 1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # Convert to 0-based index

                # If control and target qubits are different, add C(U3) gate
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # If same, skip (equivalent to Identity)

    return design


def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1, 0)
    single = np.ones((qubits, 2 * layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers + 1)], entangle, axis=1)
    return arch.transpose(1, 0)


def shift_ith_element_right(original_list, i):
    """
    对列表中每个item的第i个元素进行循环右移一位

    Args:
        original_list: 原始列表，如 [[3, 0, 5], [4, 3, 6], [5, 1, 7], [1, 2, 8]]
        i: 要循环右移的元素索引，如 i=1 表示第二个元素

    """
    ith_elements = [item[i] for item in original_list]
    # 循环右移一位：最后一个元素移到开头
    shifted_ith = [ith_elements[-1]] + ith_elements[:-1]
    result = [item[:i] + [shifted_ith[idx]] + item[i + 1:] for idx, item in enumerate(original_list)]
    return result


def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:
            if phase == 0:
                q = job[0]
                job_list += [[fold * (q - 1) + 1 + i] + job[1:] for i in range(0, fold)]
            else:
                job = [i - 1 for i in job]
                q = job[0]
                indices = [i for i, x in enumerate(job) if x < q]
                enta = [[fold * j + i + 1 for j in job] for i in range(0, fold)]
                for i in indices:
                    enta = shift_ith_element_right(enta, i)
                job_list += enta
    else:
        job_list = jobs
    return job_list


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(self.n_wires)]

        self.q_params_rot = nn.Parameter(
            pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(
            pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each CU3 gate needs 3 parameters

        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x):
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.u3(qdev, wires=self.design[i][1], params=params)
            elif self.design[i][0] == 'C(U3)':
                layer = self.design[i][2]
                control_qubit = self.design[i][1][0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.cu3(qdev, wires=self.design[i][1], params=params)
            else:  # data uploading: if self.design[i][0] == 'data'
                j = int(self.design[i][1][0])
                self.uploading[j](qdev, x[:, j])
        out = self.measure(qdev)
        if task_name.startswith('QML'):
            out = out[:, :2]  # only take the first two measurements for binary classification
        return out


class EstimatorQiskitLayer(nn.Module):

    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.n_layers = self.args.n_layers
        self.shots = self.args.shots
        self.SEED=arguments.qiskit_seed

        # Trainable parameters with identical structure to other layers
        self.q_params_rot = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)

        # Reuse original circuit construction logic to ensure consistent structure
        self.qc_template, self.data_params, self.u3_param_map, self.cu3_param_map = self._build_parametric_circuit()
        self.observables = self._prebuild_observables()

        # Initialize GenericBackendV2 and Estimator
        # self.backend = QasmSimulator(method='density_matrix')

        self._init_Noisemodel(arguments.name)
        self._init_estimator()

    def _init_Noisemodel(self,name):
        # 自动匹配硬件噪声模型
        if 'kolkata' in name:
            fake_backend = FakeKolkata()
        elif 'nairobi' in name:
            fake_backend = FakeNairobi()
        elif 'montreal' in name:
            fake_backend = FakeMontreal()
        elif 'toronto' in name:
            fake_backend = FakeToronto()
        else:
            fake_backend = FakeKolkata()

        # 从真实硬件 backend 提取噪声模型
        self.noise_model = NoiseModel.from_backend(fake_backend)
        print(f"✅ 已加载 Qiskit 0.46 官方噪声模型: {fake_backend.name()}")

    def _init_estimator(self):
        # self.backend = AerSimulator(
        #     shots=self.shots,  # 采样次数
        #     seed_simulator=self.SEED,  # 随机种子
        #     noise_model=self.noise_model  # 噪声模型（有就传，没有为None）
        # )
        self.backend = QasmSimulator(
            shots=self.shots,  # 采样次数
            seed_simulator=self.SEED,  # 随机种子
            noise_model=self.noise_model  # 噪声模型（有就传，没有为None）
        )
        self.estimator = BackendEstimator(
            backend=self.backend,
            skip_transpilation=False  # 必须开启 transpile
        )

    def _build_parametric_circuit(self):
        """Construct parametric quantum circuit with consistent structure"""
        qc = QuantumCircuit(self.n_wires)
        data_params = []
        u3_param_map = {}
        cu3_param_map = {}

        for j in range(self.n_wires):
            qubit_data_params = ParameterVector(f'data_q{j}', length=4)
            data_params.append(qubit_data_params)

        for i in tqdm(range(len(self.design)), desc="Building Circuit",leave=True):
            elem = self.design[i]
            if elem[0] == 'U3':
                layer = elem[2]
                qubit = elem[1][0]
                param_key = (layer, qubit)
                if param_key not in u3_param_map:
                    u3_params = ParameterVector(f'u3_l{layer}q{qubit}', length=3)
                    u3_param_map[param_key] = u3_params
                theta, phi, lam = u3_param_map[param_key]
                qc.u(theta, phi, lam, qubit)
            elif elem[0] == 'C(U3)':
                layer = elem[2]
                control_qubit = elem[1][0]
                target_qubit = elem[1][1]
                param_key = (layer, control_qubit)
                if param_key not in cu3_param_map:
                    cu3_params = ParameterVector(f'cu3_l{layer}cq{control_qubit}', length=3)
                    cu3_param_map[param_key] = cu3_params
                theta, phi, lam = cu3_param_map[param_key]
                qc.cu(theta, phi, lam, 0, control_qubit, target_qubit)
            else:
                j = int(elem[1][0])
                qc.ry(data_params[j][0], j)
                qc.rz(data_params[j][1], j)
                qc.rx(data_params[j][2], j)
                qc.ry(data_params[j][3], j)
        return qc, data_params, u3_param_map, cu3_param_map

    def _prebuild_observables(self):
        """Pre-build Pauli observables for expectation value calculation"""
        observables = []
        for q in range(self.n_wires):
            pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def _preprocess_x(self, x):
        """Preprocess input data following the original pipeline"""
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task
        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4, device=x.device)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)
        return x

    def create_pauli_observables(self, physical_qubit_indices):
        """
        Create Pauli-Z observables based on physical qubit mapping
        physical_qubit_indices = [0, 1, 3, 2] means:
            - Logical qubit 0 maps to physical qubit 0 -> 'ZIII'
            - Logical qubit 1 maps to physical qubit 1 -> 'IZII'
            - Logical qubit 2 maps to physical qubit 3 -> 'IIIZ'
            - Logical qubit 3 maps to physical qubit 2 -> 'IIZI'
        """
        observables = []

        # Create observable for each physical qubit
        for i, physical_qubit_idx in enumerate(physical_qubit_indices):
            pauli_str = (len(physical_qubit_indices) - 1 - physical_qubit_idx) * 'I' + 'Z' + physical_qubit_idx * 'I'
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)

        return observables

    def forward(self, x):
        """Forward pass: parameter binding, transpilation, and expectation value calculation via Estimator"""
        device = x.device
        x_pre = self._preprocess_x(x)
        bsz = x_pre.shape[0]

        # Parameter binding: assign parameters for each sample
        x_np = x_pre.detach().cpu().numpy()
        u3_np = self.q_params_rot.detach().cpu().numpy()
        cu3_np = self.q_params_enta.detach().cpu().numpy()

        batch_results = []
        for batch_idx in tqdm(range(bsz), desc="Running Estimator"):
            # Build parameter binding dictionary
            param_bind = {}
            for j in range(self.n_wires):
                param_bind[self.data_params[j]] = x_np[batch_idx, j]
            for (layer, q), params in self.u3_param_map.items():
                param_bind[params] = u3_np[layer, q]
            for (layer, cq), params in self.cu3_param_map.items():
                param_bind[params] = cu3_np[layer, cq]

            # Bind parameters and transpile circuit
            qc = self.qc_template.assign_parameters(param_bind)
            transpiled_qc = transpile(qc, backend=self.backend)

            # Calculate expectation values for all observables
            exp_vals = []
            physical_qubit_indices = [0,1,2,3]
            # for q in range(transpiled_qc.num_qubits):
            #     try:
            #         initial_layout = str(transpiled_qc.layout.initial_layout[q])
            #         index = int(initial_layout.split(', ')[-1].rstrip(')'))
            #         physical_qubit_indices.append(index)
            #     except (KeyError, IndexError, ValueError, AttributeError) as e:
            #         print(f"Warning: Could not extract mapping for physical qubit {q}: {e}")

            # Create Pauli-Z observables based on transpilation mapping (physical_qubit_indices)
            observables = self.create_pauli_observables(physical_qubit_indices)

            for obs in observables:
                job = self.estimator.run(transpiled_qc, obs)
                res = job.result()
                exp_vals.append(res.values[0])

            # Reverse order to match original code output
            # exp_vals = exp_vals[::-1]
            batch_results.append(exp_vals)

        # Convert results to PyTorch tensor
        output = torch.tensor(batch_results, dtype=torch.float32, device=device)
        return output


# -------------------------------------------------------------------------------------
dev = qml.device("lightning.qubit", wires=Arguments().n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(self, x):
    kernel_size = self.args.kernel
    task_name = self.args.task
    if not task_name.startswith('QML'):
        x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
        if kernel_size == 4:
            # x = x.view(bsz, 6, 6)
            # tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
            # x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            pass
        else:
            # x = x.view(bsz, 4, 4).transpose(1, 2)
            x = x.view(4, 4).transpose(0, 1)
    else:
        # x = x.view(bsz, self.n_wires, -1)
        pass

    for i in range(len(self.design)):
        if self.design[i][0] == 'U3':
            layer = self.design[i][2]
            qubit = self.design[i][1][0]
            phi = self.q_params_rot[layer, qubit, 0]
            theta = self.q_params_rot[layer, qubit, 1]
            omega = self.q_params_rot[layer, qubit, 2]
            qml.Rot(phi, theta, omega, wires=qubit)
        elif self.design[i][0] == 'C(U3)':
            layer = self.design[i][2]
            control_qubit = self.design[i][1][0]
            target_qubit = self.design[i][1][1]
            phi = self.q_params_enta[layer, control_qubit, 0]
            theta = self.q_params_enta[layer, control_qubit, 1]
            omega = self.q_params_enta[layer, control_qubit, 2]
            qml.CRot(phi, theta, omega, wires=[control_qubit, target_qubit])
        else:  # data uploading: if self.design[i][0] == 'data'
            j = int(self.design[i][1][0])
            qml.RY(x[:, j][0].detach(), wires=j)
            qml.RX(x[:, j][1].detach(), wires=j)
            qml.RZ(x[:, j][2].detach(), wires=j)
            qml.RY(x[:, j][3].detach(), wires=j)

    return [qml.expval(qml.PauliZ(i)) for i in range(self.args.n_qubits)]


class PennylaneLayer(nn.Module):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.u3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3),
                                      requires_grad=True)  # Each U3 gate needs 3 parameters
        self.cu3_params = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3),
                                       requires_grad=True)  # Each CU3 gate needs 3 parameters

    def forward(self, x):
        output_list = []
        for batch in range(x.size(0)):  # Use actual batch size
            x_batch = x[batch]
            output = quantum_net(self, x_batch)
            q_out = torch.stack([output[i] for i in range(len(output))]).float()
            output_list.append(q_out)
        outputs = torch.stack(output_list)

        return outputs


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        if arguments.backend == 'tq':
            print("\nRun with TorchQuantum backend.")
            self.QuantumLayer = TQLayer(self.args, self.design)
        elif arguments.backend == 'qi':
            print(f"\nRun with Qiskit quantum backend. noise = {arguments.noise} seed = {arguments.qiskit_seed} shots = {arguments.shots}, name = {arguments.name}")
            self.QuantumLayer = EstimatorQiskitLayer(self.args, self.design)
        else:  # PennyLane or others
            print("\nRun with PennyLane quantum backend or others.")
            self.QuantumLayer = PennylaneLayer(self.args, self.design)

    def forward(self, x_image, n_qubits, task_name):
        # exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        output = self.QuantumLayer(x_image)
        # output = F.log_softmax(output, dim=1)
        return output
