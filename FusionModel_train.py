import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
import numpy as np

# Qiskit imports
from math import pi
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator
# Qiskit核心导入
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import StatevectorSimulator  # 新版Aer模拟器
from qiskit_aer.backends import AerSimulator  # 备用高速模拟器
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# PennyLane imports
import pennylane as qml
from Arguments import Arguments  # Only for setting qml.device()
import os

os.environ['QISKIT_LOG_LEVEL'] = 'ERROR'
os.environ['QISKIT_ENABLE_DISPLAY'] = '0'


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


class TQLayer_old(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits

        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(10)]

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3))  # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3))  # each CU3 gate needs 3 parameters
        rot_trainable = True
        enta_trainable = True

        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):

                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                    self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer]))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x, n_qubits=4, task_name=None):
        bsz = x.shape[0]
        if task_name.startswith('QML'):
            x = x.view(bsz, n_qubits, -1)
        else:
            kernel_size = self.args.kernel
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        for layer in range(self.design['n_layers']):
            for j in range(self.n_wires):
                if self.design['qubit_{}'.format(j)][0][layer] != 0:
                    self.uploading[j](qdev, x[:, j])
                if self.design['qubit_{}'.format(j)][1][layer] == 0:
                    self.rots[j + layer * self.n_wires](qdev, wires=j)

            for j in range(self.n_wires):
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        out = self.measure(qdev)
        if task_name.startswith('QML'):
            out = out[:, :2]  # only take the first two measurements for binary classification

        return out


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


class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u3_params, cu3_params, qc_template, data_params, u3_param_map,
                cu3_param_map, observables, args, simulator):
        bsz, n_wires = x.shape[0], x.shape[1]
        # 批量参数绑定（保留优化）
        x_np = x.detach().cpu().numpy()
        u3_params_np = u3_params.detach().cpu().numpy()
        cu3_params_np = cu3_params.detach().cpu().numpy()

        bound_circuits = []
        for batch in range(bsz):
            param_binding = {}
            # 绑定数据参数
            for j in range(n_wires):
                param_binding[data_params[j][0]] = x_np[batch, j, 0]
                param_binding[data_params[j][1]] = x_np[batch, j, 1]
                param_binding[data_params[j][2]] = x_np[batch, j, 2]
                param_binding[data_params[j][3]] = x_np[batch, j, 3]
            # 绑定U3参数
            for (layer, qubit), params in u3_param_map.items():
                param_binding[params[0]] = u3_params_np[layer, qubit, 0]  # theta
                param_binding[params[1]] = u3_params_np[layer, qubit, 1]  # phi
                param_binding[params[2]] = u3_params_np[layer, qubit, 2]  # lam
            # 绑定CU3参数
            for (layer, cq), params in cu3_param_map.items():
                param_binding[params[0]] = cu3_params_np[layer, cq, 0]  # theta
                param_binding[params[1]] = cu3_params_np[layer, cq, 1]  # phi
                param_binding[params[2]] = cu3_params_np[layer, cq, 2]  # lam
            bound_circuits.append(qc_template.assign_parameters(param_binding))

        # Qiskit 1.0+ 批量执行（速度核心优化）
        result = simulator.run(bound_circuits, shots=1).result()

        # 批量处理结果（解析期望值，无采样噪声）
        quantum_outputs = []
        for i in range(bsz):
            statevector = Statevector(result.get_statevector(i))
            exp_vals = [statevector.expectation_value(obs).real for obs in observables]
            exp_vals = exp_vals[::-1]  # 对齐TQ输出顺序
            quantum_outputs.append(exp_vals)
        quantum_results = torch.tensor(quantum_outputs, dtype=torch.float32)

        # 保存反向传播数据
        ctx.save_for_backward(x, u3_params, cu3_params)
        ctx.qc_template = qc_template
        ctx.data_params = data_params
        ctx.u3_param_map = u3_param_map
        ctx.cu3_param_map = cu3_param_map
        ctx.observables = observables
        ctx.args = args
        ctx.simulator = simulator
        ctx.bsz = bsz
        ctx.n_wires = n_wires

        return quantum_results

    @staticmethod
    def backward(ctx, grad_output):
        # 加载前向保存的张量和配置
        x, u3_params, cu3_params = ctx.saved_tensors
        qc_template = ctx.qc_template
        data_params = ctx.data_params
        u3_param_map = ctx.u3_param_map
        cu3_param_map = ctx.cu3_param_map
        observables = ctx.observables
        args = ctx.args
        simulator = ctx.simulator
        bsz, n_wires = ctx.bsz, ctx.n_wires
        n_layers = u3_params.shape[0]

        eps = 1e-6
        # 初始化梯度（正确形状：[n_layers, n_wires, 3]）
        u3_grad = torch.zeros_like(u3_params)
        cu3_grad = torch.zeros_like(cu3_params)

        # ========== 修复1：按参数维度（theta/phi/lam）分别计算梯度 ==========
        # 处理U3参数梯度（layer → qubit → param维度）
        for layer in range(n_layers):
            for qubit in range(n_wires):
                # 跳过无U3参数的qubit（避免空参数扰动）
                if (layer, qubit) not in u3_param_map:
                    continue

                # 对U3的3个参数分别扰动
                for p_idx in range(3):  # 0:theta, 1:phi, 2:lam
                    orig_val = u3_params[layer, qubit, p_idx].clone()

                    # 正向扰动
                    u3_params[layer, qubit, p_idx] = orig_val + eps
                    out_plus = QuantumFunction.forward(
                        ctx, x, u3_params, cu3_params, qc_template, data_params,
                        u3_param_map, cu3_param_map, observables, args, simulator
                    )

                    # 反向扰动
                    u3_params[layer, qubit, p_idx] = orig_val - eps
                    out_minus = QuantumFunction.forward(
                        ctx, x, u3_params, cu3_params, qc_template, data_params,
                        u3_param_map, cu3_param_map, observables, args, simulator
                    )

                    # 中心有限差分计算梯度（正确链式法则）
                    grad = (out_plus - out_minus) / (2 * eps)  # [bsz, 4]
                    # 加权求和：grad_output [bsz,4] * grad [bsz,4] → 标量
                    u3_grad[layer, qubit, p_idx] = (grad_output * grad).sum()

                    # 恢复原始参数
                    u3_params[layer, qubit, p_idx] = orig_val

        # 处理CU3参数梯度（同U3逻辑，按维度分别计算）
        for layer in range(n_layers):
            for cq in range(n_wires):
                # 跳过无CU3参数的控制qubit
                if (layer, cq) not in cu3_param_map:
                    continue

                # 对CU3的3个参数分别扰动
                for p_idx in range(3):  # 0:theta, 1:phi, 2:lam
                    orig_val = cu3_params[layer, cq, p_idx].clone()

                    # 正向扰动
                    cu3_params[layer, cq, p_idx] = orig_val + eps
                    out_plus = QuantumFunction.forward(
                        ctx, x, u3_params, cu3_params, qc_template, data_params,
                        u3_param_map, cu3_param_map, observables, args, simulator
                    )

                    # 反向扰动
                    cu3_params[layer, cq, p_idx] = orig_val - eps
                    out_minus = QuantumFunction.forward(
                        ctx, x, u3_params, cu3_params, qc_template, data_params,
                        u3_param_map, cu3_param_map, observables, args, simulator
                    )

                    # 中心有限差分计算梯度
                    grad = (out_plus - out_minus) / (2 * eps)  # [bsz, 4]
                    cu3_grad[layer, cq, p_idx] = (grad_output * grad).sum()

                    # 恢复原始参数
                    cu3_params[layer, cq, p_idx] = orig_val

        # 返回梯度（x无梯度，其他为None）
        return None, u3_grad, cu3_grad, None, None, None, None, None, None, None


class QiskitLayer(nn.Module):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits  # 4量子比特
        self.n_layers = self.args.n_layers  # 4层

        # 初始化参数（单精度浮点加速）
        self.u3_params = nn.Parameter(
            pi * torch.rand(self.n_layers, self.n_wires, 3, dtype=torch.float32),
            requires_grad=True
        )
        self.cu3_params = nn.Parameter(
            pi * torch.rand(self.n_layers, self.n_wires, 3, dtype=torch.float32),
            requires_grad=True
        )

        # 电路/观测器构建（保留原逻辑）
        self.qc_template, self.data_params, self.u3_param_map, self.cu3_param_map = self._build_parametric_circuit()
        self.observables = self._prebuild_observables()

        # ========== 速度优化：初始化高性能Statevector模拟器 ==========
        self.simulator = StatevectorSimulator()

    def _build_parametric_circuit(self):
        qc = QuantumCircuit(self.n_wires)
        data_params = []
        u3_param_map = {}
        cu3_param_map = {}

        # 数据参数（每个qubit 4个参数）
        for j in range(self.n_wires):
            qubit_data_params = ParameterVector(f'data_q{j}', length=4)
            data_params.append(qubit_data_params)
            # 数据编码门（移到这里，避免design解析错误）
            qc.ry(qubit_data_params[0], j)
            qc.rz(qubit_data_params[1], j)
            qc.rx(qubit_data_params[2], j)
            qc.ry(qubit_data_params[3], j)

        # 量子层参数（U3/CU3）
        for i in range(len(self.design)):
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
                # 修复CU3门参数顺序（旧版错误：多了0参数）
                qc.cu(theta, phi, lam, 0, control_qubit, target_qubit)

        return qc, data_params, u3_param_map, cu3_param_map

    def _prebuild_observables(self):
        observables = []
        for q in range(self.n_wires):
            pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def _preprocess_x(self, x):
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task

        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size, stride=kernel_size)  # 明确stride，避免维度错误
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                # 预分配内存，减少cat开销
                x_padded = torch.empty(bsz, x.shape[1] * x.shape[2] + 4, device=x.device, dtype=x.dtype)
                x_padded[:, :-4] = x.view(bsz, -1)
                x_padded[:, -4:] = 0
                x = x_padded.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        # 确保输入维度匹配（n_wires × 4）
        if x.shape[-1] != 4:
            x = x[..., :4]  # 截断到4维

        return x

    def forward(self, x):
        device = x.device
        x_pre = self._preprocess_x(x)
        # 调用自定义可微函数
        quantum_results = QuantumFunction.apply(
            x_pre, self.u3_params, self.cu3_params,
            self.qc_template, self.data_params, self.u3_param_map,
            self.cu3_param_map, self.observables, self.args,
            self.simulator
        )
        quantum_results = quantum_results.to(device, non_blocking=True)

        # 任务适配（QML任务只取前2维）
        if self.args.task.startswith('QML'):
            quantum_results = quantum_results[:, :2]

        return quantum_results

    # 可选：添加梯度清零优化
    def zero_grad(self, set_to_none: bool = False):
        if set_to_none:
            self.u3_params.grad = None
            self.cu3_params.grad = None
        else:
            if self.u3_params.grad is not None:
                self.u3_params.grad.zero_()
            if self.cu3_params.grad is not None:
                self.cu3_params.grad.zero_()


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
            qml.RZ(x[:, j][1].detach(), wires=j)  # 修正：先RZ
            qml.RX(x[:, j][2].detach(), wires=j)  # 修正：后RX
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


class EstimatorQiskitLayer(nn.Module):
    SEED = 170

    def __init__(self, arguments, design, shots=1000):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.n_layers = self.args.n_layers
        self.shots = shots  # 固定为10000

        # 可训练参数
        self.u3_params = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)
        self.cu3_params = nn.Parameter(pi * torch.rand(self.n_layers, self.n_wires, 3), requires_grad=True)

        # 构建电路和观测器
        self.qc_template, self.data_params, self.u3_param_map, self.cu3_param_map = self._build_parametric_circuit()
        self.observables = self._prebuild_observables()

        # 预编译电路
        self.backend = GenericBackendV2(num_qubits=self.n_wires)
        self.transpiled_qc = transpile(self.qc_template, backend=self.backend, seed_transpiler=170)

        # 整理所有参数（顺序与电路中参数出现顺序一致）
        self.all_data_params = []
        for j in range(self.n_wires):
            self.all_data_params.extend(self.data_params[j])

        self.all_u3_params = []
        self.u3_keys = list(self.u3_param_map.keys())
        for (layer, qubit), params in self.u3_param_map.items():
            self.all_u3_params.extend(params)

        self.all_cu3_params = []
        self.cu3_keys = list(self.cu3_param_map.keys())
        for (layer, cq), params in self.cu3_param_map.items():
            self.all_cu3_params.extend(params)

        self.all_circuit_params = self.all_data_params + self.all_u3_params + self.all_cu3_params
        self.data_param_len = len(self.all_data_params)
        self.u3_param_len = len(self.all_u3_params)
        self.cu3_param_len = len(self.all_cu3_params)

        self.estimator = Estimator(
            backend_options={
                'method': 'statevector',
                'device': 'GPU' if arguments.device == 'cuda' else 'CPU',  # GPU/CPU自动适配
                'noise_model': None
            },
            run_options={
                'shots': self.shots,
                'seed': 170
            },
            transpile_options={
                'seed_transpiler': 170
            }
        )

        # 固定采样次数（shots=10000无需多次采样）
        self.n_repeats = 1

    def _build_parametric_circuit(self):
        """构建参数化量子电路（保留原有逻辑）"""
        qc = QuantumCircuit(self.n_wires)
        data_params = []
        u3_param_map = {}
        cu3_param_map = {}

        # 数据参数
        for j in range(self.n_wires):
            qubit_data_params = ParameterVector(f'data_q{j}', length=4)
            data_params.append(qubit_data_params)

        # 构建电路
        for i in range(len(self.design)):
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
        """预构建Pauli-Z观测器（保留原有逻辑）"""
        observables = []
        for q in range(self.n_wires):
            pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def _preprocess_x(self, x):
        """输入预处理（保留原有逻辑）"""
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

    def forward(self, x):
        """前向传播（无重复编译+批量参数绑定）"""
        return EstimatorQuantumFunction.apply(
            x,
            self.u3_params,
            self.cu3_params,
            self.transpiled_qc,
            self.all_circuit_params,
            self.observables,
            self.args,
            self.estimator,
            self.n_repeats,
            self.u3_keys,
            self.cu3_keys,
            self.data_param_len,
            self.u3_param_len,
            self.cu3_param_len
        )


class EstimatorQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u3_params, cu3_params, transpiled_qc, all_circuit_params, observables,
                args, estimator, n_repeats, u3_keys, cu3_keys, data_param_len, u3_param_len, cu3_param_len):
        # ========== 完整保存所有需要的上下文 ==========
        ctx.save_for_backward(x, u3_params, cu3_params)
        ctx.transpiled_qc = transpiled_qc
        ctx.all_circuit_params = all_circuit_params
        ctx.observables = observables
        ctx.args = args
        ctx.estimator = estimator
        ctx.n_repeats = n_repeats
        ctx.u3_keys = u3_keys
        ctx.cu3_keys = cu3_keys
        ctx.data_param_len = data_param_len
        ctx.u3_param_len = u3_param_len
        ctx.cu3_param_len = cu3_param_len

        # 关键标记
        ctx.is_qml_task = args.task.startswith('QML')
        ctx.n_qubits = transpiled_qc.num_qubits

        device = x.device
        x_pre = EstimatorQuantumFunction._preprocess_x(x, args)
        bsz = x_pre.shape[0]

        # 转换为numpy（仅一次）
        x_np = x_pre.detach().cpu().numpy()
        u3_np = u3_params.detach().cpu().numpy()
        cu3_np = cu3_params.detach().cpu().numpy()

        # 计算期望值（无重复编译+1电路1观测器）
        batch_results = []
        for batch_idx in range(bsz):
            # 批量整理参数值（长度严格匹配）
            data_vals = x_np[batch_idx].flatten()[:ctx.data_param_len]
            u3_vals = u3_np.flatten()[:ctx.u3_param_len]
            cu3_vals = cu3_np.flatten()[:ctx.cu3_param_len]
            param_values = np.concatenate([data_vals, u3_vals, cu3_vals])

            # 批量绑定参数（仅一次）
            bound_qc = ctx.transpiled_qc.assign_parameters(dict(zip(ctx.all_circuit_params, param_values)))

            # 计算所有观测器的期望值（1电路+1观测器，符合Qiskit要求）
            exp_vals = []
            for obs in ctx.observables:
                job = ctx.estimator.run([bound_qc], [obs])
                exp_vals.append(job.result().values[0])
            exp_vals = exp_vals[::-1]  # 保持原有顺序
            batch_results.append(exp_vals)

        # 输出处理
        output = torch.tensor(batch_results, dtype=torch.float32, device=device)
        if ctx.is_qml_task:
            output = output[:, :2]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, u3_params, cu3_params = ctx.saved_tensors
        # 加载上下文
        transpiled_qc = ctx.transpiled_qc
        observables = ctx.observables
        estimator = ctx.estimator
        n_qubits = ctx.n_qubits

        # 初始化梯度
        u3_grad = torch.zeros_like(u3_params)
        cu3_grad = torch.zeros_like(cu3_params)

        # ========== 解析梯度：参数偏移法则（U3参数） ==========
        shift = np.pi / 2  # 固定偏移，无需ε
        for layer in range(u3_params.shape[0]):
            for qubit in range(u3_params.shape[1]):
                for param_idx in range(3):  # U3的3个参数（θ,φ,λ）
                    # 构建偏移参数
                    u3_plus = u3_params.clone()
                    u3_plus[layer, qubit, param_idx] += shift

                    u3_minus = u3_params.clone()
                    u3_minus[layer, qubit, param_idx] -= shift

                    # 计算偏移后的期望值（仅需2次前向，无ε误差）
                    out_plus = EstimatorQuantumFunction.forward(
                        ctx, x, u3_plus, cu3_params, transpiled_qc, ctx.all_circuit_params,
                        observables, ctx.args, estimator, 1, ctx.u3_keys, ctx.cu3_keys,
                        ctx.data_param_len, ctx.u3_param_len, ctx.cu3_param_len
                    )
                    out_minus = EstimatorQuantumFunction.forward(
                        ctx, x, u3_minus, cu3_params, transpiled_qc, ctx.all_circuit_params,
                        observables, ctx.args, estimator, 1, ctx.u3_keys, ctx.cu3_keys,
                        ctx.data_param_len, ctx.u3_param_len, ctx.cu3_param_len
                    )

                    # 解析梯度（无ε分母，无数值误差）
                    grad = (out_plus - out_minus) / 2
                    # 链式法则：乘以损失函数的梯度
                    u3_grad[layer, qubit, param_idx] = (grad_output * grad).sum()

        # ========== 同理计算CU3参数的解析梯度 ==========
        for layer in range(cu3_params.shape[0]):
            for cq in range(cu3_params.shape[1]):
                for param_idx in range(3):
                    cu3_plus = cu3_params.clone()
                    cu3_plus[layer, cq, param_idx] += shift

                    cu3_minus = cu3_params.clone()
                    cu3_minus[layer, cq, param_idx] -= shift

                    out_plus = EstimatorQuantumFunction.forward(
                        ctx, x, u3_params, cu3_plus, transpiled_qc, ctx.all_circuit_params,
                        observables, ctx.args, estimator, 1, ctx.u3_keys, ctx.cu3_keys,
                        ctx.data_param_len, ctx.u3_param_len, ctx.cu3_param_len
                    )
                    out_minus = EstimatorQuantumFunction.forward(
                        ctx, x, u3_params, cu3_minus, transpiled_qc, ctx.all_circuit_params,
                        observables, ctx.args, estimator, 1, ctx.u3_keys, ctx.cu3_keys,
                        ctx.data_param_len, ctx.u3_param_len, ctx.cu3_param_len
                    )

                    grad = (out_plus - out_minus) / 2
                    cu3_grad[layer, cq, param_idx] = (grad_output * grad).sum()

        return None, u3_grad, cu3_grad, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _preprocess_x(x, args):
        """静态预处理（保留原有逻辑）"""
        bsz = x.shape[0]
        kernel_size = args.kernel
        task_name = args.task

        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4, device=x.device)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, args.n_qubits, -1)
        return x


# class EstimatorQiskitLayerV2(nn.Module):
#     SEED = 170
#
#     def __init__(self, arguments, design, shots=1024):
#         super().__init__()
#         self.args = arguments
#         self.design = design
#         self.n_wires = self.args.n_qubits
#         self.n_layers = self.args.n_layers
#         self.shots = shots
#
#         # 1. 构建参数化电路，确定输入/权重参数结构
#         self.qc_template, self.input_params, self.weight_param_list = self._build_parametric_circuit()
#         self.observables = self._prebuild_observables()
#
#         # 2. 计算输入/权重维度
#         self.n_inputs = len(self.input_params)
#         self.n_weights = len(self.weight_param_list)
#
#         # 3. 初始化Estimator和梯度计算器（适配旧版Qiskit/QML）
#         self.backend = GenericBackendV2(num_qubits=self.n_wires)
#         self._init_estimator_and_gradient()
#
#         # 4. 缓存编译后的电路和观测器（加速）
#         self._cache_circuit_and_observables()
#
#         # 5. 构建QNN和TorchConnector（核心：让TorchConnector管理权重）
#         self.qnn = self._build_qnn()
#         self.torch_qnn = TorchConnector(self.qnn)
#
#         # 6. 初始化自定义参数（与TorchConnector权重形状对齐）
#         self._init_custom_weights()
#
#         # 7. 注册钩子：将TorchConnector的梯度同步到自定义参数
#         self._register_gradient_hooks()
#
#     def _init_estimator_and_gradient(self):
#         """初始化Estimator和梯度计算器（适配旧版Qiskit/QML，移除所有不兼容参数）"""
#         # 适配旧版AER：只保留基础参数
#         self.estimator = Estimator(
#             backend_options={
#                 'method': 'statevector',
#                 'noise_model': None,
#                 'device': 'CPU',  # 明确指定CPU（单线程）
#                 'precision': 'single',  # 单精度浮点，速度更快
#                 'statevector_parallel_threshold': 0,  # 关闭并行（旧版支持）
#             },
#             run_options={
#                 'shots': self.shots,
#                 'seed': self.SEED,
#             },
#             transpile_options={
#                 'seed_transpiler': self.SEED,
#                 'optimization_level': 0,  # 关闭电路优化，减少编译时间
#             }
#         )
#
#         # 适配旧版QML：移除analytic和epsilon参数
#         self.gradient = ParamShiftEstimatorGradient(self.estimator)
#
#     def _build_parametric_circuit(self):
#         """构建参数化量子电路（明确区分输入/权重参数）"""
#         qc = QuantumCircuit(self.n_wires)
#         input_params = ParameterVector('input', length=self.n_wires * 4)  # 固定输入维度: n_qubits×4
#         weight_param_list = []  # 按顺序存储所有权重参数
#         weight_param_map = {}
#
#         # 数据编码层（使用输入参数）
#         param_idx = 0
#         for j in range(self.n_wires):
#             if param_idx + 4 <= len(input_params):
#                 qc.ry(input_params[param_idx], j)
#                 qc.rz(input_params[param_idx + 1], j)
#                 qc.rx(input_params[param_idx + 2], j)
#                 qc.ry(input_params[param_idx + 3], j)
#                 param_idx += 4
#
#         # 量子层（使用权重参数）
#         for i in range(len(self.design)):
#             elem = self.design[i]
#             if elem[0] == 'U3':
#                 layer = elem[2]
#                 qubit = elem[1][0]
#                 param_key = f'u3_l{layer}q{qubit}'
#                 if param_key not in weight_param_map:
#                     params = ParameterVector(param_key, length=3)
#                     weight_param_map[param_key] = params
#                     weight_param_list.extend(params)
#                 theta, phi, lam = weight_param_map[param_key]
#                 qc.u(theta, phi, lam, qubit)
#             elif elem[0] == 'C(U3)':
#                 layer = elem[2]
#                 control_qubit = elem[1][0]
#                 target_qubit = elem[1][1]
#                 param_key = f'cu3_l{layer}cq{control_qubit}'
#                 if param_key not in weight_param_map:
#                     params = ParameterVector(param_key, length=3)
#                     weight_param_map[param_key] = params
#                     weight_param_list.extend(params)
#                 theta, phi, lam = weight_param_map[param_key]
#                 qc.cu(theta, phi, lam, 0, control_qubit, target_qubit)
#
#         return qc, input_params, weight_param_list
#
#     def _cache_circuit_and_observables(self):
#         """缓存编译后的电路和简化的观测器（减少重复计算）"""
#         # 缓存编译后的电路
#         self._cached_circuit = transpile(
#             self.qc_template,
#             backend=self.backend,
#             optimization_level=0,
#             seed_transpiler=self.SEED
#         )
#         # 简化观测器，减少期望值计算开销
#         self._cached_observables = [obs.simplify() for obs in self.observables]
#
#     def _build_qnn(self):
#         """构建QNN（严格区分input/weight参数 + 缓存优化）"""
#         qnn = EstimatorQNN(
#             circuit=self._cached_circuit,
#             observables=self._cached_observables,
#             input_params=self.input_params,
#             weight_params=self.weight_param_list,
#             estimator=self.estimator,
#             gradient=self.gradient,
#             input_gradients=False
#         )
#         return qnn
#
#     def _init_custom_weights(self):
#         """初始化自定义权重参数（与TorchConnector权重对齐）"""
#         # 1. 初始化原始形状的参数（单精度浮点，减少内存）
#         self.u3_params = nn.Parameter(
#             torch.pi * torch.rand(self.n_layers, self.n_wires, 3, dtype=torch.float32),
#             requires_grad=True
#         )
#         self.cu3_params = nn.Parameter(
#             torch.pi * torch.rand(self.n_layers, self.n_wires, 3, dtype=torch.float32),
#             requires_grad=True
#         )
#
#         # 2. 展平并同步到TorchConnector
#         flat_weights = self._flatten_weights()
#         with torch.no_grad():
#             self.torch_qnn.weight.copy_(flat_weights)
#
#
#     def _flatten_weights(self):
#         """将u3_params/cu3_params展平为TorchConnector权重格式"""
#         weights_flat = []
#         # 严格按照design顺序展平
#         for i in range(len(self.design)):
#             elem = self.design[i]
#             if elem[0] == 'U3':
#                 layer = elem[2]
#                 qubit = elem[1][0]
#                 weights_flat.append(self.u3_params[layer, qubit])
#             elif elem[0] == 'C(U3)':
#                 layer = elem[2]
#                 control_qubit = elem[1][0]
#                 weights_flat.append(self.cu3_params[layer, control_qubit])
#         return torch.cat(weights_flat)
#
#     def _unflatten_weights(self, flat_weights):
#         """将TorchConnector的扁平权重恢复为u3_params/cu3_params形状"""
#         idx = 0
#         # 严格按照design顺序恢复
#         for i in range(len(self.design)):
#             elem = self.design[i]
#             if elem[0] == 'U3':
#                 layer = elem[2]
#                 qubit = elem[1][0]
#                 self.u3_params.data[layer, qubit] = flat_weights[idx:idx + 3]
#                 idx += 3
#             elif elem[0] == 'C(U3)':
#                 layer = elem[2]
#                 control_qubit = elem[1][0]
#                 self.cu3_params.data[layer, control_qubit] = flat_weights[idx:idx + 3]
#                 idx += 3
#
#     def _register_gradient_hooks(self):
#         """注册梯度钩子：同步TorchConnector梯度到自定义参数（优化版）"""
#
#         def grad_hook(grad):
#             """钩子函数：梯度反展平 + 内存优化"""
#             # 初始化梯度（in-place操作，减少内存分配）
#             if self.u3_params.grad is None:
#                 self.u3_params.grad = torch.zeros_like(self.u3_params, dtype=torch.float32)
#             else:
#                 self.u3_params.grad.zero_()
#
#             if self.cu3_params.grad is None:
#                 self.cu3_params.grad = torch.zeros_like(self.cu3_params, dtype=torch.float32)
#             else:
#                 self.cu3_params.grad.zero_()
#
#             # 将扁平梯度反展平到u3_params/cu3_params（in-place拷贝）
#             idx = 0
#             for i in range(len(self.design)):
#                 elem = self.design[i]
#                 if elem[0] == 'U3':
#                     layer = elem[2]
#                     qubit = elem[1][0]
#                     self.u3_params.grad[layer, qubit].copy_(grad[idx:idx + 3])
#                     idx += 3
#                 elif elem[0] == 'C(U3)':
#                     layer = elem[2]
#                     control_qubit = elem[1][0]
#                     self.cu3_params.grad[layer, control_qubit].copy_(grad[idx:idx + 3])
#                     idx += 3
#
#             # 梯度裁剪，避免异常值
#             grad.clamp_(-1.0, 1.0)
#             return grad
#
#         # 为TorchConnector的权重注册梯度钩子
#         self.torch_qnn.weight.register_hook(grad_hook)
#
#     def _prebuild_observables(self):
#         """预构建Pauli观测器"""
#         observables = []
#         for q in range(self.n_wires):
#             pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
#             observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
#             observables.append(observable)
#         return observables
#
#     def _preprocess_x(self, x):
#         """数据预处理（优化版：减少张量拷贝，加速）"""
#         bsz = x.shape[0]
#         kernel_size = self.args.kernel
#         task_name = self.args.task
#
#         if not task_name.startswith('QML'):
#             # 优化池化操作：明确指定stride，减少冗余计算
#             x = F.avg_pool2d(x, kernel_size, stride=kernel_size)
#
#             if kernel_size == 4:
#                 # 优化形状变换：预分配内存，减少中间张量
#                 x = x.view(bsz, -1)
#                 x_padded = torch.empty(bsz, x.shape[1] + 4, device=x.device, dtype=torch.float32)
#                 x_padded[:, :-4] = x
#                 x_padded[:, -4:] = 0
#                 x = x_padded.reshape(bsz, -1, 10).transpose(1, 2)
#             else:
#                 x = x.view(bsz, 4, 4).transpose(1, 2)
#         else:
#             x = x.view(bsz, self.n_wires, -1)
#
#         # 适配输入维度（优化版：预分配固定形状张量）
#         x_flat = x.reshape(bsz, -1)
#         target_dim = self.n_wires * 4
#         current_dim = x_flat.shape[1]
#
#         if current_dim != target_dim:
#             # 预分配内存，避免多次拼接
#             x_aligned = torch.empty(bsz, target_dim, device=x.device, dtype=torch.float32)
#             if current_dim > target_dim:
#                 x_aligned[:] = x_flat[:, :target_dim]
#             else:
#                 x_aligned[:, :current_dim] = x_flat
#                 x_aligned[:, current_dim:] = 0
#             x_flat = x_aligned
#
#         return x_flat
#
#     def forward(self, x):
#         """前向传播（优化版：批量处理+减少拷贝）"""
#         device = x.device
#         x_flat = self._preprocess_x(x)
#         batch_size = x_flat.shape[0]
#
#         # 强制批量计算，避免自动拆分
#         with torch.no_grad():
#             self.torch_qnn._neural_network.batch_size = batch_size
#
#         # 前向传播（核心：只传输入，权重由TorchConnector内部管理）
#         output_tensor = self.torch_qnn(x_flat)
#
#         # 调整形状并反转输出（in-place操作）
#         output_tensor = output_tensor.view(batch_size, self.n_wires).flip(dims=[1])
#
#         return output_tensor.to(device, non_blocking=True)
#
#     def zero_grad(self, set_to_none: bool = False):
#         """完整清零梯度（自定义参数+TorchConnector）"""
#         # 清零自定义参数梯度
#         if set_to_none:
#             self.u3_params.grad = None
#             self.cu3_params.grad = None
#         else:
#             if self.u3_params.grad is not None:
#                 self.u3_params.grad.zero_()
#             if self.cu3_params.grad is not None:
#                 self.cu3_params.grad.zero_()
#
#         # 清零TorchConnector梯度
#         if self.torch_qnn.weight.grad is not None:
#             if set_to_none:
#                 self.torch_qnn.weight.grad = None
#             else:
#                 self.torch_qnn.weight.grad.zero_()
#
#     def sync_weights(self):
#         """双向同步：自定义参数 ↔ TorchConnector权重"""
#         # 自定义参数 → TorchConnector
#         flat_weights = self._flatten_weights()
#         with torch.no_grad():
#             self.torch_qnn.weight.copy_(flat_weights)
#
#         # TorchConnector → 自定义参数（可选）
#         # self._unflatten_weights(self.torch_qnn.weight.data)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        if arguments.backend == 'tq':
            print("Run with TorchQuantum backend.")
            self.QuantumLayer = TQLayer(self.args, self.design)
        elif arguments.backend == 'qi':
            print("Run with Qiskit quantum backend.")
            self.QuantumLayer = QiskitLayer(self.args, self.design)
        elif arguments.backend == 'qiv2':
            print("Run with Qiskit quantum backend.")
            self.QuantumLayer = EstimatorQiskitLayer(self.args, self.design)
        else:  # PennyLane or others
            print("Run with PennyLane quantum backend or others.")
            self.QuantumLayer = PennylaneLayer(self.args, self.design)

    def forward(self, x_image, n_qubits, task_name):
        # exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        exp_val = self.QuantumLayer(x_image)
        output = F.log_softmax(exp_val, dim=1)
        return output

def test_full_consistency():
    # Fix random seeds for reproducibility
    import random
    device = torch.device("cuda")
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize configuration
    args = Arguments()
    single_code = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [2, 1, 1, 1, 1, 1, 1, 1, 1],
                   [3, 1, 1, 1, 1, 1, 1, 1, 1],
                   [4, 1, 1, 1, 1, 1, 1, 1, 1]]
    enta_code = [[1, 2, 2, 2, 2],
                 [2, 3, 3, 3, 3],
                 [3, 4, 4, 4, 4],
                 [4, 1, 1, 1, 1]]
    arch_code = [args.n_qubits, args.n_layers]  # [4,4]
    design = single_enta_to_design(single_code, enta_code, arch_code)

    # Create test input
    batch_size = 2  # Small batch size for faster testing
    test_input = torch.rand(batch_size, 1, 24, 24, requires_grad=True).to(device)  # 开启输入梯度

    threshold = 1e-5
    grad_threshold = 1e-4  # 梯度误差阈值
    est_grad_threshold = 5e-3  # Estimator梯度允许更大误差（采样噪声）

    # Initialize base models
    tq_layer = TQLayer(args, design).to(device).train()  # train模式开启梯度
    sv_layer = QiskitLayer(args, design).to(device).train()
    # Synchronize parameters to ensure identical initial values
    with torch.no_grad():
        sv_layer.u3_params.copy_(tq_layer.q_params_rot)
        sv_layer.cu3_params.copy_(tq_layer.q_params_enta)

    # ====================== Test 1: TQLayer vs StatevectorQiskitLayer (Output Consistency) ======================
    print("=" * 80)
    print("Test 1: TQLayer vs StatevectorQiskitLayer (Exact, No Sampling)")
    print("=" * 80)
    with torch.no_grad():
        tq_out = tq_layer(test_input)
        sv_out = sv_layer(test_input)

    # Calculate error metrics
    abs_diff_sv = torch.abs(tq_out - sv_out)
    max_diff_sv = abs_diff_sv.max().item()
    mean_diff_sv = abs_diff_sv.mean().item()

    print(f"TQ Output:\n{np.round(tq_out.cpu().numpy(), 6)}")
    print(f"Statevector Output:\n{np.round(sv_out.cpu().numpy(), 6)}")
    print(f"Max Absolute Error: {max_diff_sv:.8f}")
    print(f"Mean Absolute Error: {mean_diff_sv:.8f}")
    print(f"Output Consistent: {'✅ YES' if max_diff_sv < threshold else '❌ NO'}\n")

    # ====================== Test 2: TQLayer vs EstimatorQiskitLayer (Forward Output, Variable Shots) ======================
    test_shots_list = [1000, 10000, 100000]  # 测试不同采样次数
    print("=" * 80)
    print("Test 2: TQLayer vs EstimatorQiskitLayer (Forward Output, Variable Shots)")
    print("=" * 80)

    est_forward_results = {}  # 保存不同shots的前向结果，用于后续梯度对比
    for shots in test_shots_list:
        # Initialize Estimator layer and synchronize parameters
        est_layer = EstimatorQiskitLayer(args, design, shots=shots).to(device).eval()
        with torch.no_grad():
            est_layer.u3_params.copy_(tq_layer.q_params_rot)
            est_layer.cu3_params.copy_(tq_layer.q_params_enta)

        # Forward inference
        with torch.no_grad():
            est_out = est_layer(test_input)

        # Calculate error metrics
        abs_diff_est = torch.abs(tq_out - est_out)
        mean_diff_est = abs_diff_est.mean().item()
        max_diff_est = abs_diff_est.max().item()

        # 保存结果
        est_forward_results[shots] = est_out
        print(f"\nEstimator Shots = {shots}")
        print(f"Output:\n{np.round(est_out.cpu().numpy(), 6)}")
        print(f"Max Absolute Error vs TQLayer: {max_diff_est:.8f}")
        print(f"Mean Absolute Error vs TQLayer: {mean_diff_est:.8f}")
        print(f"Forward Consistent (relaxed): {'✅ YES' if mean_diff_est < 1e-2 else '❌ NO'}")

    # ====================== Test 3: Gradient Consistency (TQLayer vs StatevectorQiskitLayer) ======================
    print("\n" + "=" * 80)
    print("Test 3: Gradient Consistency (TQLayer vs StatevectorQiskitLayer)")
    print("=" * 80)

    # 重置模型梯度
    tq_layer.zero_grad()
    sv_layer.zero_grad()

    # 1. 计算TQLayer的梯度
    tq_out_train = tq_layer(test_input)
    tq_loss = tq_out_train.sum()  # 简单求和作为损失函数
    tq_loss.backward()  # 反向传播计算梯度

    # 提取TQLayer的梯度（保存为基准）
    tq_rot_grad = tq_layer.q_params_rot.grad.clone()
    tq_enta_grad = tq_layer.q_params_enta.grad.clone()

    # 2. 计算QiskitLayer的梯度
    sv_out_train = sv_layer(test_input)
    sv_loss = sv_out_train.sum()  # 相同的损失函数
    sv_loss.backward()  # 反向传播计算梯度

    # 提取QiskitLayer的梯度
    sv_u3_grad = sv_layer.u3_params.grad.clone()
    sv_cu3_grad = sv_layer.cu3_params.grad.clone()

    # 3. 对比梯度差异
    rot_grad_diff = torch.abs(tq_rot_grad - sv_u3_grad)
    rot_max_diff = rot_grad_diff.max().item()
    rot_mean_diff = rot_grad_diff.mean().item()

    enta_grad_diff = torch.abs(tq_enta_grad - sv_cu3_grad)
    enta_max_diff = enta_grad_diff.max().item()
    enta_mean_diff = enta_grad_diff.mean().item()

    # 打印梯度对比结果
    print(f"\n--- U3/q_params_rot Gradient Comparison ---")
    print(f"Max Absolute Gradient Diff: {rot_max_diff:.8f}")
    print(f"Mean Absolute Gradient Diff: {rot_mean_diff:.8f}")
    print(f"U3 Gradient Consistent: {'✅ YES' if rot_max_diff < grad_threshold else '❌ NO'}")

    print(f"\n--- CU3/q_params_enta Gradient Comparison ---")
    print(f"Max Absolute Gradient Diff: {enta_max_diff:.8f}")
    print(f"Mean Absolute Gradient Diff: {enta_mean_diff:.8f}")
    print(f"CU3 Gradient Consistent: {'✅ YES' if enta_max_diff < grad_threshold else '❌ NO'}")

    # 可选：打印梯度采样值
    print(f"\nSample TQLayer q_params_rot Gradient (layer 0, qubit 0):\n{np.round(tq_rot_grad[0, 0].cpu().numpy(), 6)}")
    print(f"Sample QiskitLayer u3_params Gradient (layer 0, qubit 0):\n{np.round(sv_u3_grad[0, 0].cpu().numpy(), 6)}")

    overall_grad_consistent = (rot_max_diff < grad_threshold) and (enta_max_diff < grad_threshold)
    print(f"\nOverall Gradient Consistent (Statevector): {'✅ YES' if overall_grad_consistent else '❌ NO'}")

    # ====================== Test 4: EstimatorQiskitLayer Gradient (Variable Shots) ======================
    print("\n" + "=" * 80)
    print("Test 4: EstimatorQiskitLayer Gradient Consistency (Variable Shots vs TQLayer)")
    print("=" * 80)

    for shots in test_shots_list:
        print(f"\n--- Estimator Shots = {shots} ---")
        # 初始化Estimator层并同步参数
        est_layer = EstimatorQiskitLayer(args, design, shots=shots).to(device).train()
        with torch.no_grad():
            est_layer.u3_params.copy_(tq_layer.q_params_rot)
            est_layer.cu3_params.copy_(tq_layer.q_params_enta)

        # 重置梯度
        est_layer.zero_grad()

        # 前向传播 + 反向传播计算梯度
        est_out_train = est_layer(test_input)
        est_loss = est_out_train.sum()  # 相同的损失函数
        est_loss.backward()

        # 提取Estimator层的梯度
        est_u3_grad = est_layer.u3_params.grad.clone()
        est_cu3_grad = est_layer.cu3_params.grad.clone()

        # 对比Estimator梯度与TQLayer梯度的差异
        est_rot_diff = torch.abs(tq_rot_grad - est_u3_grad)
        est_rot_max = est_rot_diff.max().item()
        est_rot_mean = est_rot_diff.mean().item()

        est_enta_diff = torch.abs(tq_enta_grad - est_cu3_grad)
        est_enta_max = est_enta_diff.max().item()
        est_enta_mean = est_enta_diff.mean().item()

        # 打印梯度对比结果
        print(f"\nU3 Gradient vs TQLayer:")
        print(f"  Max Diff: {est_rot_max:.8f} | Mean Diff: {est_rot_mean:.8f}")
        print(f"  Consistent: {'✅ YES' if est_rot_max < est_grad_threshold else '❌ NO'}")

        print(f"\nCU3 Gradient vs TQLayer:")
        print(f"  Max Diff: {est_enta_max:.8f} | Mean Diff: {est_enta_mean:.8f}")
        print(f"  Consistent: {'✅ YES' if est_enta_max < est_grad_threshold else '❌ NO'}")

        # 可选：打印梯度采样值
        print(f"\nSample Estimator u3_params Gradient (layer 0, qubit 0):")
        print(f"  {np.round(est_u3_grad[0, 0].cpu().numpy(), 6)}")

    # ====================== 最终总结 ======================
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(
        f"1. Statevector QiskitLayer vs TQLayer: Output {'✅' if max_diff_sv < threshold else '❌'}, Gradient {'✅' if overall_grad_consistent else '❌'}")
    print(
        f"2. Estimator QiskitLayer (shots={test_shots_list[-1]}) vs TQLayer: Forward error ~{mean_diff_est:.4f}, Gradient error ~{est_rot_mean:.4f} (U3) / {est_enta_mean:.4f} (CU3)")
    print(f"3. Estimator梯度阈值: {est_grad_threshold}, 采样次数越多，梯度越接近TQLayer")


if __name__ == "__main__":
    import torch
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")  # 屏蔽无关警告
    test_full_consistency()

