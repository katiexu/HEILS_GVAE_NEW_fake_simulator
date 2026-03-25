import torch
import os
import csv
import pandas as pd  # 添加 pandas 库
import numpy as np

def check_file(task):
    if os.path.exists('results') == False:
        os.makedirs('results')
    result = 'results/{}.csv'.format(task)
    result_fine = 'results/{}_fine.csv'.format(task)
    if os.path.isfile(result) == False:
        with open(result, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['sample_id', 'arch_code', 'sample_node', 'ACC', 'p_ACC'])

    if os.path.isfile(result_fine) == False:
        with open(result_fine, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['iteration', 'arch_code', 'ACC', 'PPG', 'Mean_Diff'])

def check_file_with_prefix(path, prefix):
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            if file.startswith(prefix):
                return file
    return False

state_path = 'states'
if os.path.exists(state_path) == False:
    os.makedirs(state_path)
files = os.listdir(state_path)

init_weight_path = 'init_weights'
if os.path.exists(init_weight_path) == False:
    os.makedirs(init_weight_path)
init_weights = os.listdir(init_weight_path)

weight_path = 'weights'
if os.path.exists(weight_path) == False:
    os.makedirs(weight_path)
weights = os.listdir(weight_path)

def empty_arch(n_layers, n_qubits): 
    single = [[i] + [0]* (2*n_layers) for i in range(1,n_qubits+1)]
    enta = [[i] *(n_layers+1) for i in range(1,n_qubits+1)]
    return [single, enta]

# check_file('MNIST')

def get_list_dimensions(lst):
    if isinstance(lst, list):
        if len(lst) == 0:
            return 1
        return 1 + get_list_dimensions(lst[0])
    return 0


def add_ppg_column(agent, csv_file):
  
    # 打开 CSV 文件并添加 PPG 列
    # csv_file = 'results/QML_fine copy.csv'
    # 读取现有的 CSV 文件
    df = pd.read_csv(csv_file)
    # 确保 agent.performance_per_gate 的长度与现有行数一致
    if len(agent.performance_per_gate) != len(df):
        raise ValueError("Length of performance_per_gate does not match the number of rows in the CSV file.")
    # 添加 PPG 列
    df['PPG'] = agent.performance_per_gate
    # 保存修改后的 CSV 文件
    df.to_csv(csv_file, index=False)

def sample_normal(mu, logvar, step_size, arch_code_fold):
    """
    Sample from N(mu, exp(logvar)) using the reparameterization trick.   
    """
    std = torch.exp(logvar)
    eps = torch.randn_like(mu)
    n_qubits, n_layers = arch_code_fold
    # Ensure step_size is a list of list
    if get_list_dimensions(step_size) == 1:
        step_size = [[item] for item in step_size]
    step_single, step_enta = step_size
    
    # Adjust the step size for single and enta
    step_size = step_single * n_qubits + step_enta * n_qubits
    step_size = step_size * n_layers
    step_size = torch.Tensor(np.diag(step_size)).to(mu.device)
    # return mu + eps * std * step_size
    return mu + torch.matmul(step_size, eps)

def difference_between_archs(original_single, original_enta, decoded_single, decoded_enta):
    single_diff = sum(abs(np.array(a) - np.array(b)).sum() for a, b in zip(original_single, decoded_single))                    
    enta_diff = sum((abs(np.array(a) - np.array(b)) != 0).sum() for a, b in zip(original_enta, decoded_enta))

    return [single_diff, enta_diff]


