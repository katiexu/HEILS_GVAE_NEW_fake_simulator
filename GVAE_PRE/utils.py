import os
import json
import torch
import GVAE_PRE.var_config as vc 
import torch.nn.functional as F
import numpy as np
from FusionModel import gen_arch

current_path = os.getcwd()

def load_json(f_name):
    """load circuit dataset."""
    file_path = os.path.join(current_path, f'circuit\\data\\{f_name}')
    with open(f_name, 'r') as file:
        dataset = json.loads(file.read())
    return dataset

def save_checkpoint(model, optimizer, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-ae-{}.pt'.format(name))
    torch.save(checkpoint, f_path)


def save_checkpoint_vae(model, optimizer, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-{}-{}.pt'.format(name, epoch))
    torch.save(checkpoint, f_path)

def normalize_adj(A):
    # Compute the sum of each row and column in A
    A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)  # Add self-loops
    sum_A_dim1 = A.sum(dim=1)
    sum_A_dim2 = A.sum(dim=2)

    # Check if sum_A_dim1 and sum_A_dim2 contain any zero values
    contains_zero_dim1 = (sum_A_dim1 == 0).any()
    contains_zero_dim2 = (sum_A_dim2 == 0).any()
    # if contains_zero_dim1:
    #     print("sum_A_dim1 contains zero values.")
    # if contains_zero_dim2:
    #     print("sum_A_dim2 contains zero values.")

    # If zero values are present, replace them with a very small number to avoid division by zero
    sum_A_dim1[sum_A_dim1 == 0] = 1e-10
    sum_A_dim2[sum_A_dim2 == 0] = 1e-10    

    D_in = torch.diag_embed(1.0 / torch.sqrt(sum_A_dim1))
    D_out = torch.diag_embed(1.0 / torch.sqrt(sum_A_dim2))
    DA = stacked_spmm(D_in, A)  # swap D_in and D_out
    DAD = stacked_spmm(DA, D_out)
    return DAD

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
    assert A.dim()==3

    if method == 0:
        return A, H

    if method==1:
        # Adding global node with padding
        A = F.pad(A, (0,1), 'constant', 1.0)
        A = F.pad(A, (0,0,0,1), 'constant', 0.0)
        H = F.pad(H, (0,1,0,1), 'constant', 0.0 )
        H[:, -1, -1] = 1.0

    if method==1:
        # using A^T instead of A
        # and also adding a global node
        A = A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A) # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        return DAD, H

    elif method == 2:
        assert lbd!=None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1-lbd)*A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A)  # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        def prep_reverse(DAD, H):
            AD = stacked_spmm(1.0/D_in, DAD)
            A =  stacked_spmm(AD, 1.0/D_out)
            return A.triu(1), H
        return DAD, H, prep_reverse

    elif method == 3:
        # bidirectional DAG
        assert lbd != None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1 - lbd) * A.transpose(-1, -2)
        def prep_reverse(A, H):
            return 1.0/lbd*A.triu(1), H
        return A, H, prep_reverse

    elif method == 4:
        A = A + A.triu(1).transpose(-1, -2)
        def prep_reverse(A, H):
            return A.triu(1), H
        return A, H, prep_reverse


def get_accuracy(inputs, targets):
    N, I, _ = inputs[0].shape
    full_ops_recon, adj_recon = inputs[0], inputs[1]
    ops_recon = full_ops_recon[:,:,0:-(vc.num_qubits)]
    full_ops, adj = targets[0], targets[1]
    ops = full_ops[:,:,0:-(vc.num_qubits)]
    # post processing, assume non-symmetric
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)

    ops_correct = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float()
    adj_correct = adj_recon_thre.eq(adj.type(torch.bool)).float()
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def get_train_acc(inputs, targets):
    acc_train = get_accuracy(inputs, targets)
    return 'training batch: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(*acc_train)

def get_train_NN_accuracy_str(inputs, targets, decoderNN, inds):
    acc_train = get_accuracy(inputs, targets)
    acc_val = get_NN_acc(decoderNN, targets, inds)
    return 'acc_ops:{0:.4f}({4:.4f}), mean_corr_adj:{1:.4f}({5:.4f}), mean_fal_pos_adj:{2:.4f}({6:.4f}), acc_adj:{3:.4f}({7:.4f}), top-{8} index acc {9:.4f}'.format(
        *acc_train, *acc_val)

def get_NN_acc(decoderNN, targets, inds):
    full_ops, adj = targets[0], targets[1]
    ops = full_ops[:,:,0:-(vc.num_qubits)]
    full_ops_recon, adj_recon, op_recon_tk, adj_recon_tk, _, ind_tk_list = decoderNN.find_NN(ops, adj, inds)
    ops_recon = full_ops_recon[:,:,0:-(vc.num_qubits)]
    correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, acc = get_accuracy((ops_recon, adj_recon), targets)
    pred_k = torch.tensor(ind_tk_list,dtype=torch.int)
    correct = pred_k.eq(torch.tensor(inds, dtype=torch.int).view(-1,1).expand_as(pred_k))
    topk_acc = correct.sum(dtype=torch.float) / len(inds)
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, pred_k.shape[1], topk_acc.item()

def get_val_acc(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon,_ = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def get_val_acc_vae(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon, mu, logvar = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def stacked_mm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def is_valid_circuit(adj, ops):
    # allowed_gates = ['PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'U3', 'SWAP']
    # allowed_gates = ['Identity', 'RX', 'RY', 'RZ', 'C(U3)']    # QWAS with data uploading
    allowed_gates = ['Identity', 'U3', 'data', 'data+U3', 'q0', 'q1', 'q2', 'q3']
    if len(adj) != len(ops) or len(adj[0]) != len(ops):
        return False
    # if ops[0] != 'START' or ops[-1] != 'END':
    #     return False
    for i in range(1, len(ops)-1):
        if ops[i] not in allowed_gates:
            return False
    return True

def compute_sum(full_op, N, P):
    """
    对 full_op 矩阵按照每 N 行划分为块，分别计算奇数块和偶数块的列和，并累加结果。
    
    Args:
        full_op (torch.Tensor): 输入矩阵，形状为 (num_rows, M)。
        N (int): 每个块的行数。
        P (int): 奇数块计算后 P 列的和，偶数块计算前 M-P 列的和。
    
    Returns:
        float: 累加的结果。
    """
    # full_op 是一个 2D 矩阵
    max_idx = torch.argmax(full_op, dim=-1)  # 对每一行求 argmax，返回每行最大值的索引
    one_hot = torch.zeros_like(full_op)  # 创建一个与 full_op 相同形状的全零矩阵

    # 在 argmax 索引处赋值为 1
    one_hot.scatter_(1, max_idx.unsqueeze(1), 1)

    full_op = one_hot  # 将 full_op 替换为 one_hot 矩阵    
    num_rows, M = full_op.shape
    total_sum = 0.0

    # 遍历每个块
    for i in range(0, num_rows, N):
        block = full_op[i:i+N]  # 取出当前块
        block_idx = i // N + 1  # 当前块的序号（从 1 开始）

        if block_idx % 2 == 1:  # 奇数序号块
            total_sum += block[:, -P:].sum().item()  # 计算后 P 列的元素和
        else:  # 偶数序号块
            total_sum += block[:, :M-P].sum().item()  # 计算前 M-P 列的元素和

    return total_sum

def get_proj_mask(x, N, P):    
    x = x.squeeze(0)
    mask = torch.zeros_like(x)
    num_rows, M = x.shape

    # 遍历每个块
    # If the block index is odd, it sets the last P columns of the block to -100 in the mask.
    # If the block index is even, it sets the first M-P columns of the block to -100.

    for i in range(0, num_rows, N):        
        idx = i // N + 1  # 当前块的序号（从 1 开始）

        if idx % 2 == 1:  # 奇数序号块
            mask[i:i+N, -P:] = -100
        else:  # 偶数序号块
            mask[i:i+N, :M-P] = -100

    return mask

def is_valid_ops_adj(full_op, n_qubits):
    full_op = full_op.squeeze(0).cpu()
    violation = compute_sum(full_op, n_qubits)
    if violation > 4:
        return False
    else:
        return True
    
def generate_single_enta_op(gate_matrix, n_qubits):
    """
    从gate_matrix中解析量子电路信息
    矩阵结构: (num_rows, 4+n_qubits)
    - 前4列: 单量子门类型 ['Identity', 'U3', 'data', 'data+U3']
    - 后n_qubits列: CU3双量子门的目标量子比特位置
    - 行索引: 包含当前量子比特和层信息
    """
    gate_matrix = gate_matrix.squeeze(0).cpu().detach().numpy()
    
    # 对每行整体求argmax得到最大值位置
    max_indices = np.argmax(gate_matrix, axis=1)
    
    op_results = []
    gate_type_names = ['Identity', 'U3', 'data', 'data+U3']
    
    # 根据矩阵结构解析每一行
    num_rows = gate_matrix.shape[0]
    n_layers = num_rows // (n_qubits * 2)  # 计算层数
    
    for row_idx in range(num_rows):
        # 计算当前行对应的层和量子比特信息
        layer_idx = row_idx // (n_qubits * 2)
        position_in_layer = row_idx % (n_qubits * 2)
        max_idx = max_indices[row_idx]
        
        # 计算当前量子比特索引
        if position_in_layer < n_qubits:
            qubit_idx = position_in_layer
        else:
            qubit_idx = position_in_layer - n_qubits
        
        if max_idx < 4:
            # 最大值在前4列，是单量子门
            gate_name = gate_type_names[max_idx]
            op_results.append((gate_name, [qubit_idx]))
        else:
            # 最大值在后n_qubits列，是CU3双量子门
            target_qubit = max_idx - 4  # 目标量子比特索引
            if qubit_idx == target_qubit:
                # 控制和目标量子比特相同，设为Identity
                op_results.append(('Identity', [qubit_idx]))
            else:
                op_results.append(('C(U3)', [qubit_idx, target_qubit]))
    
    # 转换为single和enta格式
    single = [[i + 1] for i in range(n_qubits)]
    enta = [[i + 1] for i in range(n_qubits)]
    
    # 处理每一层
    for layer in range(n_layers):
        layer_start = layer * n_qubits * 2
        
        # 处理单量子门 (每层的前n_qubits个操作)
        for q in range(n_qubits):
            op_idx = layer_start + q
            if op_idx < len(op_results):
                gate_name, qubits = op_results[op_idx]
                
                # 根据门类型设置single矩阵
                if gate_name == 'Identity':
                    single[q].extend([0, 0])
                elif gate_name == 'U3':
                    single[q].extend([0, 1])
                elif gate_name == 'data':
                    single[q].extend([1, 0])
                elif gate_name == 'data+U3':
                    single[q].extend([1, 1])
        
        # 处理双量子门 (每层的后n_qubits个操作)
        for q in range(n_qubits):
            op_idx = layer_start + n_qubits + q
            if op_idx < len(op_results):
                gate_name, qubits = op_results[op_idx]
                
                if gate_name == 'C(U3)' and len(qubits) == 2:
                    control, target = qubits
                    enta[control].append(target + 1)  # +1因为enta使用1-based索引
                else:
                    # Identity或其他情况，目标设为自身
                    enta[q].append(q + 1)
    
    return single, enta, op_results

def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)
    
    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
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
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)    
    return arch.transpose(1, 0)

def is_valid_ops_adj(full_op, n_qubits, threshold=4):
    full_op = full_op.squeeze(0).cpu()
    violation = compute_sum(full_op, n_qubits, n_qubits)
    if violation > threshold:
        return False
    else:
        return True
    
def compute_scaling_factor(x, decoder, snr_target, d):
            """
            Compute the scaling factor c based on the given formula.
            
            Args:
                x (torch.Tensor): Input tensor.
                decoder (nn.Module): Decoder model.
                snr_target (float): Target signal-to-noise ratio.
                d (int): Dimensionality of the input.

            Returns:
                float: Scaling factor c.
            """
            # Step 1: Compute y = decoder(x)
            x.requires_grad_(True)  # Enable gradient computation for x
            y = decoder(x)
            y = y[0]
            # Step 2: Compute ||y||^2 (mean squared norm of y)
            y_norm_squared = torch.mean(torch.norm(y, dim=-1) ** 2)
            
            # Step 3: Compute Jacobian J of the decoder
            J = []
            for i in range(y.shape[2]):  # Iterate over output dimensions
                grad_outputs = torch.zeros_like(y)
                grad_outputs[:, i] = 1.0  # One-hot vector for each output dimension
                J_i = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
                J.append(J_i)
            J = torch.stack(J, dim=1)  # Stack Jacobian components

            # Step 4: Compute ||J||_F^2 (Frobenius norm squared of the Jacobian)
            J_norm_squared = torch.sum(J ** 2)
            
            # Step 5: Compute scaling factor c
            x_norm = torch.norm(x.reshape(x.shape[0], -1), dim=-1).mean()
            c = torch.sqrt(y_norm_squared / (snr_target * J_norm_squared)) * (x_norm / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
        
            return c.item()

from GVAE_translator import generate_circuits, get_gate_and_adj_matrix
from configs import configs

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

def generate_single_enta(gate_matrix, n_qubits):
 
    # Transfer gate_matrix into one-hot format
    gate_matrix = gate_matrix.squeeze(0).cpu().detach().numpy()
    one_hot_gate_matrix = np.zeros_like(gate_matrix, dtype=int)
    max_indices = np.argmax(gate_matrix, axis=1)
    one_hot_gate_matrix[np.arange(gate_matrix.shape[0]), max_indices] = 1
 
    # Generate single and enta
    single = [[i + 1] for i in range(n_qubits)]
    enta = [[i + 1] for i in range(n_qubits)]
    single_list_info = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for n in range(0, len(gate_matrix), n_qubits * 2):
        qubit_info = one_hot_gate_matrix[n:n+n_qubits, :-n_qubits]
        adj_info = one_hot_gate_matrix[n+n_qubits:n+2*n_qubits, -n_qubits:]
        for q in range(n_qubits):
            single_res = np.sum(qubit_info[q] * np.array((0, 1, 2, 3)))
            single[q].extend(single_list_info[single_res])
            try:
                enta[q].append(int(np.squeeze(np.argwhere(adj_info[q])))+1)
            except:
                enta[q].append(q+1)
 
    # Translate single to single op_results
    single_op_results = {i: [] for i in range(n_qubits)}
    for col_index in range(1, len(single[0]), 2):
        for row_index in range(len(single)):
            value1 = single[row_index][col_index]
            value2 = single[row_index][col_index + 1]
            combined = f"{value1}{value2}"
            if combined == '00':
                single_op_results[(col_index-1)/2].append(('Identity', row_index))
            elif combined == '01':
                single_op_results[(col_index-1)/2].append(('U3', row_index))
            elif combined == '10':
                single_op_results[(col_index-1)/2].append(('data', row_index))
            elif combined == '11':
                single_op_results[(col_index-1)/2].append(('data+U3', row_index))
            else:
                pass
 
    # Translate enta to enta op_results
    enta_op_results = {i: [] for i in range(len(enta[0]) - 1)}
    for col_index in range(1, len(enta[0])):
        for row_index in range(len(enta)):
            control = row_index
            target = enta[row_index][col_index] - 1
            if control == target:
                enta_op_results[col_index - 1].append(('Identity', target))
            else:
                if col_index - 1 in enta_op_results:
                    enta_op_results[col_index - 1].append(('C(U3)', control, target))
 
    # Combine single and enta op_results
    op_results = []
    for layer in range(int(one_hot_gate_matrix.shape[0]/(n_qubits * 2))):
        op_results.extend(single_op_results[layer])
        op_results.extend(enta_op_results[layer]) 
 
    return single, enta, op_results