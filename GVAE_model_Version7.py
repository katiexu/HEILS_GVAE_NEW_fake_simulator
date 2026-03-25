import torch
import torch.nn as nn
import torch.nn.functional as F
from GVAE_model import normalize_adj, swap_ops

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout

    def forward(self, ops, adj):
        ops = F.dropout(ops, p=self.dropout, training=self.training)
        support = self.linear(ops)
        output = F.relu(torch.bmm(adj, support))
        return output

class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu1 = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar1 = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_mu2 = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar2 = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize

    def get_gcs(self, dims, dropout):
        gcs = []
        for k in range(len(dims) - 1):
            gcs.append(GraphConvolution(dims[k], dims[k + 1], dropout))
        return gcs    
    
    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        # ops = swap_ops(ops, 4)  # 4 is the category of the single gates
        x = ops
        for gc in self.gcs[:-1]:
            x = gc(x, adj)
        mu1 = reduce_dimension(self.gc_mu1(x, adj))
        logvar1 = reduce_dimension(self.gc_logvar1(x, adj))
        mu2 = reduce_dimension(self.gc_mu2(x, adj))
        logvar2 = reduce_dimension(self.gc_logvar2(x, adj))
        return mu1, logvar1, mu2, logvar2

class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, **kwargs):
        super(Decoder, self).__init__()
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout
        self.activation_ops = activation_ops
        self.activation_adj = activation_adj

    def combine_ops(self, ops_single, ops_enta, n_qubits):
        chunks = []
        total_nodes = ops_single.shape[1]
        for i in range(0, total_nodes, n_qubits):
            end = min(i + n_qubits, total_nodes)
            chunks.append(ops_single[:, i:end, :])
            chunks.append(ops_enta[:, i:end, :])
        ops = torch.cat(chunks, dim=1)
        return ops

    def forward(self, emb_single, emb_enta, n_qubits):
        emb_single = F.dropout(emb_single, p=self.dropout, training=self.training)
        ops_single = self.weight(emb_single)
        ops_enta = self.weight(emb_enta)
        ops = self.combine_ops(ops_single, ops_enta, n_qubits)
        adj_recon = torch.matmul(ops, ops.permute(0, 2, 1))
        # ops = swap_ops(ops, ops.shape[-1] - 4)  # Swap back the operations
        return self.activation_adj(ops), self.activation_adj(adj_recon)  

class GVAE_Dual(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE_Dual, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu    

    def forward(self, ops, adj, n_qubits=5):
        mu1, logvar1, mu2, logvar2 = self.encoder(ops, adj)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        
        ops_recon, adj_recon = self.decoder(z1, z2, n_qubits)
        return ops_recon, adj_recon, mu1, logvar1, mu2, logvar2

class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu1, logvar1, mu2, logvar2):        
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops_target, adj_target = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops_target)
        loss_adj = self.loss_adj(adj_recon, adj_target)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD_1 = -0.5 / (ops_recon.shape[0] * ops_recon.shape[1]) * torch.mean(
            torch.sum(1 + 2 * logvar1 - mu1.pow(2) - logvar1.exp().pow(2), 2)
        )
        KLD_2 = -0.5 / (ops_recon.shape[0] * ops_recon.shape[1]) * torch.mean(
            torch.sum(1 + 2 * logvar2 - mu2.pow(2) - logvar2.exp().pow(2), 2)
        )
        return loss + KLD_1 + KLD_2
    
def reduce_dimension(mu):
    # Assuming mu1 has shape (batch, N, features)
    # We want to reduce N by averaging every two elements
    B, N, F = mu.shape
    if N % 2 != 0:
        raise ValueError("N must be even for this reduction method.")
    return mu.view(B, N // 2, 2, F).mean(dim=2)