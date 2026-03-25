import pickle
import numpy as np
import torch

from schemes import Scheme
from FusionModel import *
from Arguments import Arguments
import random
import json
from GVAE_PRE.gen_random_circuits import generate_random_circuits
from GVAE_PRE.utils import get_proj_mask, compute_scaling_factor
from configs import configs
from GVAE_TEST.GVAE_model_old import GVAE, generate_single_enta, is_valid_ops_adj, preprocessing, op_list_to_design
from GVAE_TEST.GVAE_translator import *


# Set global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def projection(arch_next, single, enta):
    # Define the projection logic here
    single = sorted(single, key=lambda x: x[0])
    enta = sorted(enta, key=lambda x: x[0])
    single = np.array(single)
    enta = np.array(enta)
    projected_archs = []
    for arch in arch_next:
        new_single = single * (arch[0]==-1) + arch[0] * (arch[0]!=-1)
        new_enta = enta * (arch[1]==-1) + arch[1] * (arch[1]!=-1)
        projected_archs.append([new_single.tolist(), new_enta.tolist()])
    return projected_archs

def arch_to_z(archs, arch_code_fold, encoder):
        # Convert arch matrix to latent space representations
        adj_list, op_list = [], []
        for net in archs:
            circuit_ops = generate_circuits(net, arch_code_fold)
            _, gate_matrix, adj_matrix = get_gate_and_adj_matrix(circuit_ops, arch_code_fold)
            ops = torch.tensor(gate_matrix, dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
            adj_list.append(adj)
            op_list.append(ops)

        adj = torch.cat(adj_list, dim=0)
        ops = torch.cat(op_list, dim=0)
        adj, ops, prep_reverse = preprocessing(adj, ops, **configs[4]['prep'])
        encoder.eval()
        mu, logvar = encoder(ops, adj)
        return mu, logvar    

def sample_normal(mu, logvar, step_size, arch_code_fold):
    """
    Sample from N(mu, exp(logvar)) using the reparameterization trick.   
    """
    std = torch.exp(logvar)
    eps = torch.randn_like(std)

    n_qubits, n_layers = arch_code_fold
    step_single, step_enta = step_size
    # Adjust the step size for single and enta
    step_size = [step_single] * n_qubits + [step_enta] * n_qubits
    step_size = [1] + step_size * n_layers + [1]  # start and end
    step_size = torch.Tensor(np.diag(step_size))
    # return mu + eps * std * step_size
    return mu + torch.matmul(step_size, eps)

def single_enta_to_design(single, enta, arch_code_fold):
    """
    从single和enta编码生成可以被QNET使用的design列表
    
    Args:
        single: 单量子门编码，格式为[[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                gate_config每两位表示一层：00=Identity, 01=U3, 10=data, 11=data+U3
        enta: 双量子门编码，格式为[[qubit, target_layer0, target_layer1, ...], ...]
              每一位表示在该层中的目标量子比特位置
        arch_code_fold: [n_qubits, n_layers]
    
    Returns:
        design: 包含量子电路设计信息的列表，每个元素为(gate_type, [wire_indices], layer)
    """
    design = []
    n_qubits, n_layers = arch_code_fold
    
    # 处理每一层
    for layer in range(n_layers):
        # 首先处理单量子门
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # 转换为0-based索引
            # 每层的配置在列表中的位置：1 + layer*2 和 1 + layer*2 + 1
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
                # 00 (Identity) 跳过
        
        # 然后处理双量子门
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # 转换为0-based索引
            # 目标量子比特位置在列表中的位置：1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # 转换为0-based索引
                
                # 如果控制量子比特和目标量子比特不同，则添加C(U3)门
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # 如果相同，则跳过（相当于Identity）
    
    return design

def Langevin_update(x, model, snr=10, step_size=0.01):
        
        x, logvar = arch_to_z([x], arch_code_fold, model.encoder)
        x_valid_list = []

        # Compute scaling factor c
        decoder = model.decoder
        decoder.eval()
        d = x.shape[2]  # Dimensionality
        c1 = compute_scaling_factor(x, decoder, snr[0], d)
        c2 = compute_scaling_factor(x, decoder, snr[1], d)
        
        # c2 = 0
        n_qubit = arch_code_fold[0]        
        # x_norm_per_sample = torch.norm(x, dim=2, keepdim=True)

        for i in range(1000):
            # noise = torch.randn_like(x)
            step_size = [c1, c2]
            x_new = sample_normal(x, logvar,step_size, arch_code_fold)
            x_new = decoder(x_new)
            if is_valid_ops_adj(x_new[0], x_new[1], n_qubit):
                single,enta,op_result = generate_single_enta(x_new[0], n_qubits)
                x_valid_list.append([single, enta, op_result])

        print('Number of valid circuits:', len(x_valid_list))
        return x_valid_list

def evaluate_langevin_neighborhood(arch, snr_values, task):
    results = {}
    weight = torch.load('init_weights/init_weight', weights_only=True)
    original_single, original_enta = arch['single'], arch['enta']
    for snr in snr_values:
        print("***"* 20)
        print(f"\033[31mEvaluating SNR={snr}\033[0m")
        Arch = cir_to_matrix(original_single, original_enta, arch_code, args.fold)
        arch_next = Langevin_update(Arch, GVAE_model, snr)        
        evaluate_number = task['eval_number']
        if len(arch_next) <= evaluate_number:
            number = len(arch_next)
        else:
            number = evaluate_number
        
        print(f"Found {len(arch_next)} architectures for SNR={snr}.")
            
        arch_next = random.sample(arch_next, number)  # Sample architectures for evaluation

        arch_next = projection(arch_next, original_single, original_enta)

        performances = []
        difference = []
        for arch in arch_next:
            single = arch[0]
            enta = arch[1]
            # print('single:', single, 'enta:', enta)

            # Evaluate using Scheme (set epochs as needed)
            design = single_enta_to_design(single, enta, arch_code_fold)
            # design = op_list_to_design(arch[2], arch_code_fold)

            model, report = Scheme(design, task, weight, epochs=task['eval_epochs'])
            performances.append(report['mae'])

            # diff = difference_between_archs(original_single, original_enta, single, enta)
            diff = [0, 0]  # Placeholder for difference calculation
            difference.append(diff)
            print('\033[33mDifference:\033[0m', diff)

        mean_perf = np.mean(performances) if performances else None
        std_perf = np.std(performances) if performances else None
        mean_diff = np.mean(difference, axis=0) if difference else None
        std_diff = np.std(difference, axis=0) if difference else None
        results[tuple(snr)] = {
            'mean_perf': mean_perf, 
            'std_perf': std_perf,
            'mean_diff': mean_diff, 
            'std_diff': std_diff
        }
    return results

def difference_between_archs(original_single, original_enta, decoded_single, decoded_enta):
    single_diff = sum(abs(np.array(a) - np.array(b)).sum() for a, b in zip(original_single, decoded_single))                    
    enta_diff = sum((abs(np.array(a) - np.array(b)) != 0).sum() for a, b in zip(original_enta, decoded_enta))

    return [single_diff, enta_diff]
                


if __name__ == "__main__":

    
    # Setup task and agent
    task = {
        'task': 'MNIST_4',
        'option': 'mix_reg',
        'regular': True,
        'n_qubits': 4,
        'n_layers': 4,
        'fold': 1,
        'eval_number': 10,
        'eval_epochs': 5,
        'snr_values': [[0.1, 0.1], [1, 1], [3, 3]]
    }

    # task = {
    #     'task': 'MNIST_10',
    #     'option': 'mix_reg',
    #     'regular': True,
    #     'n_qubits': 10,
    #     'n_layers': 4,
    #     'fold': 2
    # }
    arch_code = [task['n_qubits'], task['n_layers']]
    arch_code_fold = [task['n_qubits']//task['fold'], task['n_layers']]
    args = Arguments(**task)
    n_qubits, n_layers = arch_code_fold

    # Load search space and create agent
    initial_circuits = generate_random_circuits(10, n_qubits, args.file_single, args.file_enta)

    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
    initial_circuits = [{'single': single, 'enta': enta}]

    # with open('data/random_circuits_mnist_5.json', 'r') as f:
    #     initial_circuits = json.load(f)
    
    checkpoint = torch.load('GVAE_TEST/best_model.pt', map_location=torch.device('cpu'), weights_only=True)


    input_dim = 7 + arch_code_fold[0]
    GVAE_model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE'])
    GVAE_model.load_state_dict(checkpoint)

    snr_values = task['snr_values']
    results_all = []

    for idx, arch in enumerate(initial_circuits[:1]):
        print(f"\033[34mEvaluating circuit {idx+1}/{len(initial_circuits)}\033[0m")
        snr_results = evaluate_langevin_neighborhood(arch, snr_values, task)
        print(f"Results for circuit {idx+1}: {snr_results}")
        results_all.append(snr_results)

    # Print summary
    print("\nResults for each SNR value across circuits:")
    for snr in snr_values:
        print(f"\nSNR={snr}:")
        
        # Get mean performance and standard deviation
        mean_perf_values = [r[tuple(snr)]['mean_perf'] for r in results_all if tuple(snr) in r and r[tuple(snr)]['mean_perf'] is not None]
        std_perf_values = [r[tuple(snr)]['std_perf'] for r in results_all if tuple(snr) in r and r[tuple(snr)]['std_perf'] is not None]
        if mean_perf_values:
            mean_perf = np.mean(mean_perf_values)
            std_perf = np.mean(std_perf_values)
            print(f"  Mean MAE: {mean_perf:.4f}, Std MAE: {std_perf:.6f}")
        else:
            print(f"  Mean MAE: No valid results")
        
        # Get mean differences and standard deviation
        mean_diff_values = [r[tuple(snr)]['mean_diff'] for r in results_all if tuple(snr) in r and r[tuple(snr)]['mean_diff'] is not None]
        std_diff_values = [r[tuple(snr)]['std_diff'] for r in results_all if tuple(snr) in r and r[tuple(snr)]['std_diff'] is not None]
        if mean_diff_values:
            mean_diff = np.mean(mean_diff_values, axis=0)
            std_diff = np.mean(std_diff_values, axis=0)
            print(f"  Mean Diff: {mean_diff}, Std Diff: {std_diff}")
        else:
            print(f"  Mean Diff: No valid differences")

    # Save results to CSV
    import csv
    with open('langevin_snr_old_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit', 'SNR_Single', 'SNR_Enta', 'Mean_MAE', 'Std_MAE', 'Mean_Single_Diff', 'Mean_Enta_Diff', 'Std_Single_Diff', 'Std_Enta_Diff'])
        for idx, snr_results in enumerate(results_all):
            for snr, result_dict in snr_results.items():
                snr_single, snr_enta = snr
                mean_mae = result_dict['mean_perf'] if result_dict['mean_perf'] is not None else 'N/A'
                std_mae = result_dict['std_perf'] if result_dict['std_perf'] is not None else 'N/A'
                if result_dict['mean_diff'] is not None and result_dict['std_diff'] is not None:
                    mean_single_diff, mean_enta_diff = result_dict['mean_diff']
                    std_single_diff, std_enta_diff = result_dict['std_diff']
                else:
                    mean_single_diff, mean_enta_diff = 'N/A', 'N/A'
                    std_single_diff, std_enta_diff = 'N/A', 'N/A'
                writer.writerow([idx, snr_single, snr_enta, mean_mae, std_mae, mean_single_diff, mean_enta_diff, std_single_diff, std_enta_diff])
    
    print(f"\nResults saved to 'langevin_snr_old_results.csv'")