import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import numpy as np


def transform_operations(max_idx):
    transform_dict = {0: 'START', 1: 'Identity', 2: 'RX', 3: 'RY', 4: 'RZ', 5: 'C(U3)', 6: 'END'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

def generate_single_enta(gate_matrix, n_qubit):

    gate_matrix = gate_matrix.squeeze(0).cpu().detach().numpy()
    # transfer gate_matrix into one-hot format
    one_hot_gate_matrix = np.zeros_like(gate_matrix, dtype=int)
    max_indices = np.argmax(gate_matrix, axis=1)
    one_hot_gate_matrix[np.arange(gate_matrix.shape[0]), max_indices] = 1

    single = [[i + 1] for i in range(n_qubit)]
    enta = [[i + 1] for i in range(n_qubit)]
    single_list_info = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for n in range(0, len(one_hot_gate_matrix), n_qubit * 2):
        qubit_info = one_hot_gate_matrix[n:n+n_qubit, :-n_qubit]
        adj_info = one_hot_gate_matrix[n+n_qubit:n+2*n_qubit, -n_qubit:]
        for q in range(n_qubit):
            single_res = np.sum(qubit_info[q] * np.array((0, 1, 2, 3)))
            single[q].extend(single_list_info[single_res])
            try:
                enta[q].append(int(np.squeeze(np.argwhere(adj_info[q])))+1)
            except:
                enta[q].append(q+1)
   
    return single,enta

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

def compute_sum(full_op, n_qubits):
    """
    对 full_op 矩阵按照每 N 行划分为块，分别计算奇数块和偶数块的列和，并累加结果。
    
    Args:
        full_op (torch.Tensor): 输入矩阵，形状为 (num_rows, M)。
        N (int): 每个块的行数。
        P (int): 奇数块计算后 P 列的和，偶数块计算前 M-P 列的和。
        N=P=NUMBER_OF_QUBITS
    
    Returns:
        float: 累加的结果。
    """    
    max_idx = torch.argmax(full_op, dim=-1)  
    one_hot = torch.zeros_like(full_op)  
    
    one_hot.scatter_(1, max_idx.unsqueeze(1), 1)

    full_op = one_hot     
    num_rows, M = full_op.shape
    total_sum = 0.0
    N = n_qubits

    # 遍历每个块
    for i in range(0, num_rows, N):
        block = full_op[i:i+N]  # 取出当前块
        block_idx = i // N + 1  # 当前块的序号（从 1 开始）

        if block_idx % 2 == 1:  # 奇数序号块
            total_sum += block[:, -N:].sum().item()  # 计算后 P 列的元素和
        else:  # 偶数序号块
            total_sum += block[:, :M-N].sum().item()  # 计算前 M-P 列的元素和

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
    if violation > 6:
        return False
    else:
        return True   


def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

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

def normalize_adj(A):
    A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)  # Add self-loops
    # Compute the sum of each row and column in A
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

def swap_ops(ops, p):
        """
        Swap the first p channels with the remaining ones along dim=2.
        
        """
        return torch.cat([ops[:, :, p:], ops[:, :, :p]], dim=2)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ops, adj):
        ops = F.dropout(ops, self.dropout, self.training)
        support = F.linear(ops, self.weight)
        output = F.relu(torch.matmul(adj, support))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GVAE(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, ops, adj):     
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)

        return ops_recon, adj_recon, mu, logvar

class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        # ops = swap_ops(ops, 4)  # 4 is the category of the single gates
        x = ops
        for gc in self.gcs[:-1]:
            x = gc(x, adj)
        mu = self.gc_mu(x, adj)
        logvar = self.gc_logvar(x, adj)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder, self).__init__()
        if adj_hidden_dim == None:
            self.adj_hidden_dim = embedding_dim
        if ops_hidden_dim == None:
            self.ops_hidden_dim = embedding_dim
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        ops = self.weight(embedding)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        # ops = swap_ops(ops, ops.shape[-1]-4)  # Swap back the operations
        return self.activation_adj(ops), self.activation_adj(adj)

class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD