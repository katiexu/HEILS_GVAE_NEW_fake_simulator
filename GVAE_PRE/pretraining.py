import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import argparse
import numpy as np
import torch.nn as nn
import var_config as vc
import json

from torch import optim
from configs import configs

from GVAE_model import VAEReconstructed_Loss, GVAE
# from GVAE_model_Version7 import GVAE_Dual, VAEReconstructed_Loss, reduce_dimension
from utils import load_json, save_checkpoint_vae, preprocessing
from utils import get_val_acc_vae, is_valid_ops_adj, generate_single_enta, compute_sum
from GVAE_translator import get_gate_and_adj_matrix, generate_circuits
from FusionModel import cir_to_matrix

def transform_operations(max_idx):
    transform_dict =  {0:'Identity', 1:'U3', 2:'data', 3:'data+U3', 4:'q0', 5:'q1', 6:'q2', 7:'q3'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

def _build_dataset(dataset, list):
    indices = np.random.permutation(list)
    X_adj = []
    X_ops = []
    for ind in indices:
        X_adj.append(torch.Tensor(dataset[ind]['adj_matrix']))
        X_ops.append(torch.Tensor(dataset[ind]['gate_matrix']))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return X_adj, X_ops, torch.Tensor(indices)

def pretraining_model(dataset, cfg, args):
    train_ind_list, val_ind_list = range(int(len(dataset)*0.9)), range(int(len(dataset)*0.9), len(dataset))
    X_adj_val, X_ops_val, indices_val = _build_dataset(dataset, val_ind_list)
    # model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
    #                num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()
    model = GVAE((args.input_dim, 32, 64, 128, 64, 32, args.dim), normalize=True, dropout=args.dropout, **cfg['GAE']).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
    epochs = args.epochs
    bs = args.bs
    loss_total = []
    
    # Initialize loss function once
    loss_fn = VAEReconstructed_Loss(**cfg['loss'])

    # Load pretrained model if available
    checkpoint = torch.load('pretrained/dim-16/model-circuits_4_qubits-9.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    with open(f'data/random_circuits_mnist_{arch_code[0]}.json', 'r') as f:
        test_circuits = json.load(f)

    # n_layers = 4
    # n_qubits = 5
    # single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    # enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    # test_circuits = [{'single': single, 'enta': enta}]
    
    # evaluate_test_circuits(model, test_circuits)

    for epoch in range(0, epochs):

        # Shuffle the training data at the beginning of each epoch
        # train_ind_list = range(2048)
        X_adj_train, X_ops_train, indices_train = _build_dataset(dataset, train_ind_list)

        chunks = len(train_ind_list) // bs
        if len(train_ind_list) % bs > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)
        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            optimizer.zero_grad()
            adj, ops = adj.cuda(), ops.cuda()
            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            # forward
            # ops_recon, adj_recon, mu_1, logvar_1, mu_2, logvar_2 = model(ops, adj, arch_code[0])  # for GVAE_Dual
            ops_recon, adj_recon, mu_1, logvar_1 = model(ops, adj)
            Z.append(mu_1)
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            
            # Calculate detailed loss components for monitoring
            ops_loss = cfg['loss']['loss_ops'](ops_recon, ops) if hasattr(cfg['loss']['loss_ops'], '__call__') else nn.MSELoss()(ops_recon, ops)
            adj_loss = cfg['loss']['loss_adj'](adj_recon, adj) if hasattr(cfg['loss']['loss_adj'], '__call__') else nn.MSELoss()(adj_recon, adj)
            
            # loss = loss_fn((ops_recon, adj_recon), (ops, adj), mu_1, logvar_1, mu_2, logvar_2) # With KL
            loss = loss_fn((ops_recon, adj_recon), (ops, adj), mu_1, logvar_1) # With KL

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_epoch.append(loss.item())
            if i%400 == 0:
                print('epoch {}: batch {} / {}: total_loss: {:.5f}, ops_loss: {:.5f}, adj_loss: {:.5f}'.format(
                    epoch, i, chunks, loss.item(), ops_loss.item(), adj_loss.item()))
        Z = torch.cat(Z, dim=0)       

        # Test circuit comparison
        evaluate_test_circuits(model, test_circuits)
        
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val, X_ops_val, indices_val)
        print('validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        
        epoch_loss = sum(loss_epoch)/len(loss_epoch)
        print('epoch {}: average loss {:.5f}'.format(epoch, epoch_loss))
        loss_total.append(epoch_loss)
        
        # Step the scheduler
        scheduler.step(epoch_loss)
        
        save_checkpoint_vae(model, optimizer, epoch, epoch_loss, args.dim, args.name, args.dropout, args.seed)

    print('loss for epochs: \n', loss_total)

def testing_model(model, Z, cfg, args):    

    z_mean, z_std = Z.mean(0), Z.std(0)
    validity_counter = [0, 0, 0]
    violation_total = 0
        
    model.eval()
    args.latent_points = 10000
    for _ in range(args.latent_points):
        z = torch.randn(X_adj_train[0].shape[0], args.dim).cuda()
        z = z * z_std + z_mean
        # if epoch == args.epochs - 1:
        #     torch.save(z, 'z.pt')
        full_op, full_ad = model.decoder(z.unsqueeze(0))
        full_op = full_op.squeeze(0).cpu()
        ad = full_ad.squeeze(0).cpu()
        
        violation = compute_sum(full_op, args.n_qubits, args.n_qubits)
        # if is_valid_circuit(ad_decode, op_decode):
        if violation < 4:
            validity_counter[0] += 1
        elif violation >= 4 and violation < 9:
            validity_counter[1] += 1
        else:
            validity_counter[2] += 1
        violation_total += violation           

    validity_counter = np.array(validity_counter)
    validity = validity_counter / args.latent_points
    validity_counter = np.array(validity_counter)
    validity = validity_counter / args.latent_points
    print('Ratio of valid decodings from the prior:', validity, violation_total/args.latent_points)
    # print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter+1e-8)))  

def evaluate_test_circuits(model, test_circuits):
    total_differences = 0
    valid_comparisons = 0

    model.eval()
    for circuit in test_circuits:
        try:
            # Extract original single and enta coding
            original_single = circuit.get('single', [])
            original_enta = circuit.get('enta', [])
            
            # Convert circuit to model input format
            cir = cir_to_matrix(original_single, original_enta, arch_code)
            circuit_ops = generate_circuits(cir, arch_code)
            _, gate_matrix, adj_matrix = get_gate_and_adj_matrix(circuit_ops, arch_code)
            
            # Preprocess
            device = next(model.parameters()).device
            adj_prep, ops_prep, prep_reverse = preprocessing(
                torch.Tensor(adj_matrix).unsqueeze(0).to(device),
                torch.Tensor(gate_matrix).unsqueeze(0).to(device),
                **cfg['prep']
            )
            
            # Forward pass through model
            ops_recon, adj_recon, _, _ = model(ops_prep, adj_prep)
            
            if is_valid_ops_adj(ops_recon, arch_code[0], 4):
                # Convert decoded operations to single and enta coding
                decoded_single, decoded_enta, _ = generate_single_enta(ops_recon.squeeze(0), arch_code[0])
                
                # Count differences
                single_diff = sum(
                    abs(np.array(a) - np.array(b)).sum() for a, b in zip(original_single, decoded_single)
                )
                
                enta_diff = sum(
                    (abs(np.array(a) - np.array(b)) != 0).sum() for a, b in zip(original_enta, decoded_enta)
                )
                
                total_differences += single_diff + enta_diff
                valid_comparisons += 1
            
        except Exception as e:
            print(f"Error processing test circuit: {e}")
            continue

    if valid_comparisons > 0:
        avg_differences = total_differences / valid_comparisons
        print(f'\033[93mAverage differences in test circuits: {avg_differences:.4f} ({valid_comparisons}/{len(test_circuits)} valid comparisons)\033[0m')
    else:
        print('\033[93mNo valid comparisons found in test circuits.\033[0m')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # parser.add_argument('--data', type=str, default=f'circuit\\data\\data_{vc.num_qubits}_qubits.json',
    #                     help='Data file (default: data.json')
    parser.add_argument('--data', type=str, default=f'data/data_4_qubits_1.json',
                        help='Data file (default: data.json')
    parser.add_argument('--name', type=str, default=f'circuits_{vc.num_qubits}_qubits',
                        help='circuits with correspoding number of qubits')
    parser.add_argument('--cfg', type=int, default=4,
                        help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=16,
                        help='training epochs (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='decoder implicit regularization (default: 0.1)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    # parser.add_argument('--input_dim', type=int, default=2+len(vc.allowed_gates)+vc.num_qubits)
    parser.add_argument('--input_dim', type=int, default=8)     # MNIST-4, 4 qubits
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature (latent) dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')

    arch_code = [5, 4]
    arch_code = [4, 4]
    args = parser.parse_args()
    
    
    args.data = f'data/data_{arch_code[0]}_qubits.json'
    args.name = f'circuits_{arch_code[0]}_qubits'
    args.input_dim = arch_code[0] + 4  # 4 single gates
    args.n_qubits = arch_code[0]
    args.epochs = 20

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = configs[args.cfg]
    dataset = load_json(args.data)
    print('using {}'.format(args.data))
    print('feat dim {}'.format(args.dim))
    
    pretraining_model(dataset, cfg, args)