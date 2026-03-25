import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import List, Dict, Any, Tuple
import concurrent.futures
from functools import lru_cache, partial
import hashlib
import pickle


class EstimatorQiskitLayer(nn.Module):
    SEED = 170

    def __init__(self, arguments, design, shots=1024, use_statevector=True,
                 gradient_method='finite_difference', eps=1e-5):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        self.n_layers = self.args.n_layers
        self.shots = shots
        self.use_statevector = use_statevector
        self.gradient_method = gradient_method
        self.eps = eps

        # Trainable parameters
        self.u3_params = nn.Parameter(0.1 * torch.randn(self.n_layers, self.n_wires, 3))
        self.cu3_params = nn.Parameter(0.1 * torch.randn(self.n_layers, self.n_wires, 3))

        # Build template circuit
        self.qc_template, self.param_dict = self._build_parametric_circuit()

        # Pre-build observables
        self.observables = self._prebuild_observables()

        # Initialize backend and estimator
        self.backend = AerSimulator()
        self._init_estimator()

        # Transpile template circuit
        self.transpiled_template = transpile(
            self.qc_template,
            self.backend,
            optimization_level=3,
            seed_transpiler=self.SEED
        )

        # Cache for circuits
        self._circuit_cache = {}
        self._circuit_hash_dict = {}

    def _init_estimator(self):
        """Initialize Estimator with backend options"""
        backend_options = {
            'method': 'statevector' if self.use_statevector else 'density_matrix',
        }

        self.estimator = AerEstimator(
            backend_options=backend_options,
            run_options={
                'shots': self.shots if not self.use_statevector else None,
                'seed': self.SEED,
            },
            approximation=False
        )

    def _build_parametric_circuit(self):
        """Build parametric circuit"""
        qc = QuantumCircuit(self.n_wires)
        param_dict = {
            'data': [],
            'u3': {},
            'cu3': {}
        }

        # Create data parameters
        for j in range(self.n_wires):
            data_params = []
            for k in range(4):
                param = Parameter(f'data_{j}_{k}')
                data_params.append(param)
            param_dict['data'].append(data_params)

            # Add data encoding gates
            qc.ry(data_params[0], j)
            qc.rz(data_params[1], j)
            qc.rx(data_params[2], j)
            qc.ry(data_params[3], j)

        # Create U3 parameters
        for layer in range(self.n_layers):
            for q in range(self.n_wires):
                u3_params = []
                for k in range(3):
                    param = Parameter(f'u3_{layer}_{q}_{k}')
                    u3_params.append(param)
                param_dict['u3'][(layer, q)] = u3_params

        # Create CU3 parameters
        for layer in range(self.n_layers):
            for cq in range(self.n_wires):
                cu3_params = []
                for k in range(3):
                    param = Parameter(f'cu3_{layer}_{cq}_{k}')
                    cu3_params.append(param)
                param_dict['cu3'][(layer, cq)] = cu3_params

        # Build circuit from design
        for elem in self.design:
            if elem[0] == 'U3':
                layer, qubit = elem[2], elem[1][0]
                params = param_dict['u3'].get((layer, qubit))
                if params:
                    qc.u(params[0], params[1], params[2], qubit)

            elif elem[0] == 'C(U3)':
                layer, control, target = elem[2], elem[1][0], elem[1][1]
                params = param_dict['cu3'].get((layer, control))
                if params:
                    qc.cu(params[0], params[1], params[2], 0, control, target)

        return qc, param_dict

    def _prebuild_observables(self):
        """Pre-build Pauli observables"""
        observables = []
        for q in range(self.n_wires):
            pauli_str = 'I' * q + 'Z' + 'I' * (self.n_wires - q - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def _preprocess_x(self, x):
        """Preprocess input data"""
        bsz = x.shape[0]
        kernel_size = self.args.kernel
        task_name = self.args.task

        if not task_name.startswith('QML'):
            x = F.avg_pool2d(x, kernel_size)
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1),
                                 torch.zeros(bsz, 4, device=x.device, dtype=x.dtype)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1, 2)
            else:
                x = x.view(bsz, 4, 4).transpose(1, 2)
        else:
            x = x.view(bsz, self.n_wires, -1)

        return x

    def _get_circuit_hash(self, data_batch, u3_params, cu3_params):
        """Generate hash for circuit parameters"""
        # Create a hashable representation of parameters
        data_str = data_batch.tobytes()
        u3_str = u3_params.tobytes()
        cu3_str = cu3_params.tobytes()

        combined = data_str + u3_str + cu3_str
        return hashlib.md5(combined).hexdigest()

    def _create_circuit_batch(self, data_batch, u3_params, cu3_params, use_cache=True):
        """Create circuits for a batch of data with optional caching"""
        batch_size = data_batch.shape[0]
        circuits = []

        for i in range(batch_size):
            if use_cache:
                # Generate hash for this circuit
                circuit_hash = self._get_circuit_hash(
                    data_batch[i:i + 1],
                    u3_params,
                    cu3_params
                )

                if circuit_hash in self._circuit_cache:
                    circuits.append(self._circuit_cache[circuit_hash])
                    continue

            param_bind = {}

            # Bind data parameters
            for j in range(self.n_wires):
                for k in range(4):
                    param_name = f'data_{j}_{k}'
                    param_bind[param_name] = data_batch[i, j, k]

            # Bind U3 parameters
            for layer in range(self.n_layers):
                for q in range(self.n_wires):
                    for k in range(3):
                        param_name = f'u3_{layer}_{q}_{k}'
                        param_bind[param_name] = u3_params[layer, q, k]

            # Bind CU3 parameters
            for layer in range(self.n_layers):
                for cq in range(self.n_wires):
                    for k in range(3):
                        param_name = f'cu3_{layer}_{cq}_{k}'
                        param_bind[param_name] = cu3_params[layer, cq, k]

            # Create circuit
            circuit = self.transpiled_template.assign_parameters(param_bind)

            if use_cache:
                self._circuit_cache[circuit_hash] = circuit

            circuits.append(circuit)

        return circuits

    def _compute_expectation_batch(self, circuits, observables, parallel=False):
        """Compute expectation values for a batch of circuits"""
        batch_results = []

        if parallel:
            # Parallel computation
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_circuit = {}

                for circuit_idx, circuit in enumerate(circuits):
                    for obs_idx, obs in enumerate(self.observables):
                        future = executor.submit(
                            self._compute_expectation_single,
                            circuit, obs
                        )
                        future_to_circuit[future] = (circuit_idx, obs_idx)

                # Initialize results matrix
                results_matrix = [[0.0 for _ in range(len(observables))]
                                  for _ in range(len(circuits))]

                for future in concurrent.futures.as_completed(future_to_circuit):
                    circuit_idx, obs_idx = future_to_circuit[future]
                    try:
                        result = future.result()
                        results_matrix[circuit_idx][obs_idx] = result
                    except Exception as e:
                        print(f"Error computing expectation: {e}")
                        results_matrix[circuit_idx][obs_idx] = 0.0

                # Convert to batch results
                for circuit_idx in range(len(circuits)):
                    exp_vals = results_matrix[circuit_idx][::-1]  # Reverse order
                    batch_results.append(exp_vals)
        else:
            # Sequential computation
            for circuit in circuits:
                exp_vals = []
                for obs in observables:
                    job = self.estimator.run(circuit, obs)
                    result = job.result()
                    exp_vals.append(result.values[0])
                exp_vals = exp_vals[::-1]  # Reverse order
                batch_results.append(exp_vals)

        return batch_results

    def _compute_expectation_single(self, circuit, observable):
        """Compute expectation value for a single circuit and observable"""
        job = self.estimator.run(circuit, observable)
        return job.result().values[0]

    def forward(self, x):
        """Forward pass"""
        device = x.device
        x_pre = self._preprocess_x(x)
        bsz = x_pre.shape[0]

        # Convert to numpy
        with torch.no_grad():
            data_np = x_pre.cpu().numpy()
            u3_np = self.u3_params.detach().cpu().numpy()
            cu3_np = self.cu3_params.detach().cpu().numpy()

        # Create circuits
        circuits = self._create_circuit_batch(data_np, u3_np, cu3_np, use_cache=True)

        # Compute expectation values
        batch_results = self._compute_expectation_batch(circuits, self.observables, parallel=True)

        # Convert to tensor
        output = torch.tensor(batch_results, dtype=torch.float32, device=device)
        return output


# 简化的版本，不需要计算梯度
class SimpleEstimatorQiskitLayer(EstimatorQiskitLayer):
    """Simplified version without gradient computation"""

    def __init__(self, arguments, design, shots=1024, use_statevector=True):
        super().__init__(arguments, design, shots, use_statevector, 'none')

    def forward(self, x):
        """Simple forward pass without gradient tracking"""
        with torch.no_grad():
            device = x.device
            x_pre = self._preprocess_x(x)
            bsz = x_pre.shape[0]

            # Convert to numpy
            data_np = x_pre.cpu().numpy()
            u3_np = self.u3_params.detach().cpu().numpy()
            cu3_np = self.cu3_params.detach().cpu().numpy()

            # Create circuits
            circuits = self._create_circuit_batch(data_np, u3_np, cu3_np, use_cache=True)

            # Compute expectation values
            batch_results = []
            for circuit in circuits:
                exp_vals = []
                for obs in self.observables:
                    job = self.estimator.run(circuit, obs)
                    result = job.result()
                    exp_vals.append(result.values[0])
                exp_vals = exp_vals[::-1]  # Reverse order
                batch_results.append(exp_vals)

            # Convert to tensor
            output = torch.tensor(batch_results, dtype=torch.float32, device=device)

        return output


# 梯度计算版本
class GradEstimatorQiskitLayer(EstimatorQiskitLayer):
    """Version with gradient computation using finite difference"""

    def __init__(self, arguments, design, shots=1024, use_statevector=True, eps=1e-5):
        super().__init__(arguments, design, shots, use_statevector, 'finite_difference', eps)

    def forward(self, x):
        """Forward pass that tracks gradients"""
        return QuantumLayerFunction.apply(x, self.u3_params, self.cu3_params, self)


class QuantumLayerFunction(torch.autograd.Function):
    """Custom autograd function for quantum layer with gradient computation"""

    @staticmethod
    def forward(ctx, x, u3_params, cu3_params, layer_obj):
        # Save for backward
        ctx.save_for_backward(x, u3_params, cu3_params)
        ctx.layer_obj = layer_obj

        # Forward pass
        with torch.no_grad():
            device = x.device
            x_pre = layer_obj._preprocess_x(x)
            bsz = x_pre.shape[0]

            # Convert to numpy
            data_np = x_pre.cpu().numpy()
            u3_np = u3_params.detach().cpu().numpy()
            cu3_np = cu3_params.detach().cpu().numpy()

            # Create circuits
            circuits = layer_obj._create_circuit_batch(data_np, u3_np, cu3_np, use_cache=False)

            # Compute expectation values
            batch_results = []
            for circuit in circuits:
                exp_vals = []
                for obs in layer_obj.observables:
                    job = layer_obj.estimator.run(circuit, obs)
                    result = job.result()
                    exp_vals.append(result.values[0])
                exp_vals = exp_vals[::-1]  # Reverse order
                batch_results.append(exp_vals)

            # Convert to tensor
            output = torch.tensor(batch_results, dtype=torch.float32, device=device)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using finite difference"""
        x, u3_params, cu3_params = ctx.saved_tensors
        layer_obj = ctx.layer_obj
        device = x.device
        eps = layer_obj.eps

        # Initialize gradients
        grad_x = torch.zeros_like(x) if x.requires_grad else None
        grad_u3 = torch.zeros_like(u3_params) if u3_params.requires_grad else None
        grad_cu3 = torch.zeros_like(cu3_params) if cu3_params.requires_grad else None

        if not (x.requires_grad or u3_params.requires_grad or cu3_params.requires_grad):
            return grad_x, grad_u3, grad_cu3, None

        # Compute gradients for u3_params
        if u3_params.requires_grad:
            grad_u3 = QuantumLayerFunction._compute_param_gradient(
                x, u3_params, cu3_params, layer_obj, grad_output, 'u3', eps, device
            )

        # Compute gradients for cu3_params
        if cu3_params.requires_grad:
            grad_cu3 = QuantumLayerFunction._compute_param_gradient(
                x, u3_params, cu3_params, layer_obj, grad_output, 'cu3', eps, device
            )

        # Compute gradients for input x
        if x.requires_grad:
            grad_x = QuantumLayerFunction._compute_input_gradient(
                x, u3_params, cu3_params, layer_obj, grad_output, eps, device
            )

        return grad_x, grad_u3, grad_cu3, None

    @staticmethod
    def _compute_param_gradient(x, params1, params2, layer_obj, grad_output,
                                param_type, eps, device):
        """Compute gradient for parameters using finite difference"""
        if param_type == 'u3':
            params = params1
            param_shape = (layer_obj.n_layers, layer_obj.n_wires, 3)
        else:  # 'cu3'
            params = params2
            param_shape = (layer_obj.n_layers, layer_obj.n_wires, 3)

        gradient = torch.zeros_like(params)
        bsz = x.shape[0]

        # Precompute base output
        with torch.no_grad():
            x_pre = layer_obj._preprocess_x(x)
            data_np = x_pre.cpu().numpy()
            u3_np = params1.detach().cpu().numpy()
            cu3_np = params2.detach().cpu().numpy()

            base_circuits = layer_obj._create_circuit_batch(data_np, u3_np, cu3_np, use_cache=False)
            base_results = []
            for circuit in base_circuits:
                exp_vals = []
                for obs in layer_obj.observables:
                    job = layer_obj.estimator.run(circuit, obs)
                    result = job.result()
                    exp_vals.append(result.values[0])
                exp_vals = exp_vals[::-1]
                base_results.append(exp_vals)

            base_output = torch.tensor(base_results, device=device)

        # Compute gradient for each parameter
        for l in range(param_shape[0]):
            for w in range(param_shape[1]):
                for p in range(param_shape[2]):
                    # Create perturbed parameters
                    params_plus = params.detach().clone()
                    params_minus = params.detach().clone()

                    params_plus[l, w, p] += eps
                    params_minus[l, w, p] -= eps

                    # Compute output with perturbed parameters
                    with torch.no_grad():
                        if param_type == 'u3':
                            circuits_plus = layer_obj._create_circuit_batch(
                                data_np, params_plus.cpu().numpy(), cu3_np, use_cache=False
                            )
                            circuits_minus = layer_obj._create_circuit_batch(
                                data_np, params_minus.cpu().numpy(), cu3_np, use_cache=False
                            )
                        else:
                            circuits_plus = layer_obj._create_circuit_batch(
                                data_np, u3_np, params_plus.cpu().numpy(), use_cache=False
                            )
                            circuits_minus = layer_obj._create_circuit_batch(
                                data_np, u3_np, params_minus.cpu().numpy(), use_cache=False
                            )

                        # Compute outputs
                        output_plus = QuantumLayerFunction._compute_batch_output(
                            circuits_plus, layer_obj, device
                        )
                        output_minus = QuantumLayerFunction._compute_batch_output(
                            circuits_minus, layer_obj, device
                        )

                    # Finite difference gradient
                    fd_gradient = (output_plus - output_minus) / (2 * eps)

                    # Compute gradient
                    param_grad = torch.sum(fd_gradient * grad_output) / bsz
                    gradient[l, w, p] = param_grad

        return gradient

    @staticmethod
    def _compute_input_gradient(x, u3_params, cu3_params, layer_obj, grad_output, eps, device):
        """Compute gradient for input using finite difference"""
        gradient = torch.zeros_like(x)
        bsz = x.shape[0]

        # Precompute base output
        with torch.no_grad():
            x_pre = layer_obj._preprocess_x(x)
            data_np = x_pre.cpu().numpy()
            u3_np = u3_params.detach().cpu().numpy()
            cu3_np = cu3_params.detach().cpu().numpy()

            base_circuits = layer_obj._create_circuit_batch(data_np, u3_np, cu3_np, use_cache=False)
            base_output = QuantumLayerFunction._compute_batch_output(base_circuits, layer_obj, device)

        # Compute gradient for each input element
        x_flat = x.view(bsz, -1)
        grad_flat = gradient.view(bsz, -1)

        for i in range(bsz):
            for j in range(x_flat.shape[1]):
                # Perturb input
                x_plus = x_flat.clone()
                x_minus = x_flat.clone()

                x_plus[i, j] += eps
                x_minus[i, j] -= eps

                x_plus_reshaped = x_plus.view_as(x)
                x_minus_reshaped = x_minus.view_as(x)

                # Preprocess perturbed inputs
                x_pre_plus = layer_obj._preprocess_x(x_plus_reshaped)
                x_pre_minus = layer_obj._preprocess_x(x_minus_reshaped)

                data_plus = x_pre_plus.cpu().numpy()
                data_minus = x_pre_minus.cpu().numpy()

                # Compute outputs
                with torch.no_grad():
                    circuits_plus = layer_obj._create_circuit_batch(data_plus, u3_np, cu3_np, use_cache=False)
                    circuits_minus = layer_obj._create_circuit_batch(data_minus, u3_np, cu3_np, use_cache=False)

                    output_plus = QuantumLayerFunction._compute_batch_output(circuits_plus, layer_obj, device)
                    output_minus = QuantumLayerFunction._compute_batch_output(circuits_minus, layer_obj, device)

                # Finite difference gradient
                fd_gradient = (output_plus - output_minus) / (2 * eps)

                # Compute gradient
                input_grad = torch.sum(fd_gradient * grad_output) / bsz
                grad_flat[i, j] = input_grad

        return gradient

    @staticmethod
    def _compute_batch_output(circuits, layer_obj, device):
        """Compute output for a batch of circuits"""
        batch_results = []
        for circuit in circuits:
            exp_vals = []
            for obs in layer_obj.observables:
                job = layer_obj.estimator.run(circuit, obs)
                result = job.result()
                exp_vals.append(result.values[0])
            exp_vals = exp_vals[::-1]
            batch_results.append(exp_vals)

        return torch.tensor(batch_results, device=device)