
import torch
import numpy as np

from GVAE_translator import get_gate_and_adj_matrix, generate_circuits
from FusionModel import cir_to_matrix
from utils import load_json, preprocessing, is_valid_ops_adj, generate_single_enta
from model import GVAE
from configs import configs

def evaluate_test_circuits(model, test_circuits, arch_code):
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

if __name__ == "__main__":
    

    arch_code = [5, 4]
    
    input_dim = arch_code[0] + 4  # 4 single gates
    n_qubits = arch_code[0]
    cfg = configs[4]

    model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **cfg['GAE']).to('cuda')

    # Load the pre-trained model state
    checkpoint = torch.load('pretrained/model-circuits_5_qubits-15.pt', map_location='cuda')
    model.load_state_dict(checkpoint['model_state'])

    # Example test circuits (replace with actual data as needed)
    n_layers = 4
    n_qubits = 5
    single = [[i]+[1]*2*n_layers for i in range(1, n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1, n_qubits)]+[[n_qubits]+[1]*n_layers]
    test_circuits = [{'single': single, 'enta': enta}]

    evaluate_test_circuits(model, test_circuits, arch_code)