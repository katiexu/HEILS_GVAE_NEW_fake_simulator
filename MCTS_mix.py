import pickle
import os
import random
import json
import csv
import numpy as np
from sympy import true
from Node import Node, Color
from prepare import *
from schemes import Scheme, Scheme_eval
from FusionModel import single_enta_to_design
import datetime
from FusionModel import cir_to_matrix 
import time
from sampling import sampling_node
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager

from Arguments import Arguments
import argparse
import torch.nn as nn
from GVAE_model import is_valid_ops_adj, generate_single_enta, get_proj_mask

class MCTS:
    def __init__(self, search_space, tree_height, fold, arch_code):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        # assert type(search_space[0]) == type([])

        self.search_space   = search_space 
        self.ARCH_CODE      = arch_code
        self.ROOT           = None
        self.Cp             = 0.2
        self.nodes          = []
        self.samples        = {}
        self.samples_true   = {}
        self.samples_compact = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list    = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []
        self.stages         = 0
        self.sampling_num   = 0   
        self.acc_mean       = 0

        self.tree_height    = tree_height

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(tree_height, fold, None, is_good_kid, self.ARCH_CODE, True))
            else:
                self.nodes.append(Node(tree_height, fold, self.nodes[parent_id], is_good_kid, self.ARCH_CODE, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.weight = 'init'
        self.best_model_weight = None
        self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'regular': [0.001, 0.002, 0.002]}
        self.best = {'acc': 0, 'model':[]}
        self.task_name = ''
        self.history = [[] for i in range(2)]
        self.qubit_used = []
        self.period = 1
        self.fold = fold
        self.performance_per_gate = []
        self.mean_diff = []

    def init_train(self, numbers=50):
        
        print("\npopulate search space...")
        self.populate_prediction_data()
        print("finished")
        print("\npredict and partition nets in search space...")
        self.predict_nodes()
        self.check_leaf_bags()
        print("finished")
        self.print_tree()

        self.sampling_arch(numbers)
        self.reset_node_data() 

        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        
        self.TASK_QUEUE = []
        self.sample_nodes = []        
        
        # strategy = 'base'
        strategy = self.weight
        
        sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1], reverse=True)]
        epochs = 20        
        # pick best 2 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(1)]
        best_change = random.choice(best_changes)
        explicit = (get_list_dimensions(best_change) < 3)
        if explicit:
            self.ROOT.base_code = best_change
            qubits = [code[0] for code in self.ROOT.base_code]
        
            print('Explicit Change: ', best_change)

            if len(best_change[0]) == len(self.explorations['single'][0]):
                best_change_full = self.insert_job(self.explorations['single'], best_change)
                self.explorations['single'] = best_change_full                
            else:
                best_change_full = self.insert_job(self.explorations['enta'], best_change)
                self.explorations['enta'] = best_change_full
            single = self.explorations['single']
            enta = self.explorations['enta']
        else:
            print('Implicit change:', best_change)
            single = best_change[0]
            enta = best_change[1]
            self.explorations['single'] = single
            self.explorations['enta'] = enta
            qubits = []

        # sorted by the first element
        self.explorations['single'] = sorted(self.explorations['single'], key=lambda x: x[0])
        self.explorations['enta'] = sorted(self.explorations['enta'], key=lambda x: x[0])

        best_arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)
        # plot_2d_array(arch)
        design = single_enta_to_design(single, enta, self.ARCH_CODE, args.fold)
        model_weight = check_file_with_prefix('weights', 'weight_{}_'.format(self.ITERATION))
        if model_weight:            
            best_model, report = Scheme_eval(design, task, model_weight)
            print('Test ACC: ', report['mae'])
        else:
            if debug:
                epochs = 0
            best_model, report = Scheme(design, task, strategy, epochs)
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%m-%d-%H')
            torch.save(best_model.state_dict(), 'weights/weight_{}_{}'.format(self.ITERATION, formatted_time))

        self.weight = best_model.state_dict()
        self.samples_true[json.dumps(np.int8(best_arch).tolist())] = report['mae']

        with open('results/{}_fine.csv'.format(self.task_name), 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            if not explicit:
                best_change_full = best_change
            writer.writerow([self.ITERATION, best_change_full, metrics, self.performance_per_gate[-1], self.mean_diff[-1]])
        
        if qubits != []:
            self.history.append(qubits)
        
        self.ROOT.base_code = None               
        self.qubit_used = self.history[-2:]
        
        self.samples_compact = {}
        self.explorations['iteration'] += 1
        arch_last = single + enta

        samples = 20
        search_space_path = os.path.join('search_space', f'search_space_mnist_{arch_code[0]}')
        if not os.path.exists(search_space_path):
            search_space_path = os.path.join('search_space', f'search_space_mnist_{arch_code_fold[0]}')
        with open(search_space_path, 'rb') as file:
                self.search_space = pickle.load(file)
        if args.strategy == 'mix':
            samples = 10 
        
        if args.strategy in ['explicit', 'mix']:
            # remove last configuration
            for i in range(len(arch_last)):
                try:
                    self.search_space.remove(arch_last[i])
                except ValueError:
                    pass
            self.search_space = [x for x in self.search_space if [x[0]] not in self.qubit_used]        
            self.init_train(samples)

        if args.strategy in ['implicit', 'mix']:    
            # implicit search
            
            self.search_space = []
            self.qubit_used = []
            
            
            self.ROOT.base_code = None        
            self.history = [[] for i in range(2)]
            
            # print(Color.BLUE + 'Implicit Switch' + Color.RESET)

            arch_next = self.Langevin_update(best_arch)
            # imp_arch_list = self.projection(arch_next, single, enta)
            for arch in arch_next:
                self.search_space.append(arch)        

            self.init_train(samples)        
            # self.qubit_used = qubits


    def get_grad(self, x):
        
        model = self.ROOT.classifier.model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        x = self.nodes[0].classifier.arch_to_z([x]).cuda()
        x.requires_grad_(True)
        x.retain_grad()
        n_heads = 3
        y = torch.tensor([[1, 1, 1]]).cuda()

        # clear grads        
        optimizer.zero_grad()

        # forward to get predicted values
        outputs = model(x)
        loss = loss_fn(outputs[0], y.long())
        loss.backward(retain_graph=True)
        return x, x.grad

    def compute_scaling_factor(self, x, decoder, snr_target):
        """
        Compute the scaling factor c for each SNR value in snr_target.
        
        Args:
            x (torch.Tensor): Input tensor.
            decoder (nn.Module): Decoder model.
            snr_target (float or list of float): Target signal-to-noise ratio(s).
            d (int): Dimensionality of the input.

        Returns:
            list: List of scaling factors c, one for each snr_target value.
        """
        # Step 1: Compute y = decoder(x)
        d = x.shape[-1]  # Dimensionality
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

        # Step 4: Compute ||J||_2^2 (Frobenius norm squared of the Jacobian)
        J_norm_squared = torch.sum(J ** 2)

        # Step 5: Compute scaling factor c for each snr_target
        x_norm = torch.norm(x.reshape(x.shape[0], -1), dim=-1).mean()
        if isinstance(snr_target, (list, tuple, np.ndarray)):
            c_list = []
            for snr in snr_target:
                c = torch.sqrt(y_norm_squared / (snr * d * J_norm_squared)) * (x_norm / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
                c_list.append(c.item())
            return c_list
        else:
            c = torch.sqrt(y_norm_squared / (snr_target * d * J_norm_squared)) * (x_norm / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
            return [c.item()]
    
    def compute_optimal_snr(self, z, logvar, target_single_diff=4, target_enta_diff=4, snr_range=(0.1, 10.0), n_trials=10):
        """
        Compute the most likely SNR [snr_single, snr_enta] that generates circuits with 
        single_diff and enta_diff <= target values.
        
        Args:
            x: Input architecture
            
            snr_range: Range of SNR values to search
            n_trials: Number of SNR values to test per dimension
            
        Returns:
            Optimal SNR values [snr_single, snr_enta]
        """
        
        decoder = self.ROOT.classifier.GVAE_model.decoder
        decoder.eval()
        n_qubit = self.ARCH_CODE[0] // self.fold
        
        # Get original single and enta from current explorations
        # original_single = self.explorations['single']
        # original_enta = self.explorations['enta']

        x_new = decoder(z)
        mask = get_proj_mask(x_new[0], n_qubit, n_qubit)
        gate_matrix = x_new[0] + mask
        original_single, original_enta, _ = generate_single_enta(gate_matrix, n_qubit)
        
        # snr_values = np.linspace([0.01, 0.01], [0.1, 0.1], n_trials)
        # snr_values = np.linspace([0.1, 0.01], [1, 0.5], n_trials)
        snr_values = np.linspace([0.01, 0.01], [0.1, 0.5], n_trials)


        snr_score_list = []

        total_nodes = z.shape[1]
        chunks1, chunks2 = [], []
        for i in range(0, total_nodes, 2*n_qubit):
            chunks1.append(z[:, i:i + n_qubit, :])
            chunks2.append(z[:, i + n_qubit:i + 2*n_qubit, :])
        z_single = torch.cat(chunks1, dim=1)
        z_enta = torch.cat(chunks2, dim=1)

        for snr_pair in snr_values:
            snr_single, snr_enta = snr_pair[0], snr_pair[1]
            valid_count = 0
            diff = []
            scores = []

            # Compute scaling factors
            c1 = self.compute_scaling_factor(z_single, decoder, snr_single)
            c2 = self.compute_scaling_factor(z_enta, decoder, snr_enta)

            # Test a few samples with this SNR pair
            for _ in range(100):
                step_size = [c1, c2]
                x_new = sample_normal(z, logvar, step_size, arch_code_fold)
                x_new = decoder(x_new)
                # mask = get_proj_mask(x_new[0], n_qubit, n_qubit)

                if is_valid_ops_adj(x_new[0], n_qubit):
                    gate_matrix = x_new[0] + mask
                    single, enta, _ = generate_single_enta(gate_matrix, n_qubit)
                    valid_count += 1

                    # Calculate differences
                    # single_diff = self.calculate_gate_difference(original_single, single)
                    # enta_diff = self.calculate_gate_difference(original_enta, enta)
                    _, single_diff = self.compare_and_mask(original_single, single)
                    _, enta_diff = self.compare_and_mask(original_enta, enta)

                    # Calculate MSE between [single_diff, enta_diff] and [target_single_diff, target_enta_diff]
                    mse = (single_diff - target_single_diff) ** 2 + (enta_diff - target_enta_diff) ** 2
                    scores.append(mse)
                    diff.append((single_diff, enta_diff))

            # Score based on validity and constraint satisfaction
            if valid_count > 0:
                score = np.mean(scores)
                diff = np.mean(diff, axis=0)
                snr_score_list.append((score, diff.tolist(), [snr_single, snr_enta]))
                # diff_list.append((diff, [snr_single, snr_enta]))
            else:
                # If no valid samples, assign a high score
                snr_score_list.append((float('inf'), float('inf'), [snr_single, snr_enta]))

        # Sort the snr_score_list by score
        snr_score_list_sorted = sorted(snr_score_list, key=lambda x: x[0])
        print("Sorted SNR values by score:")
        for score, diff, snr in snr_score_list_sorted:
            score_str = f"{score:.2f}" if isinstance(score, float) and not np.isinf(score) else str(score)
            snr_str = [f"{v:.2f}" for v in snr]
            if isinstance(diff, (list, np.ndarray)):
                diff_str = [f"{v:.2f}" for v in diff]
            else:
                diff_str = diff
            print(f"SNR: {snr_str}, Score: {score_str}, Diff: {diff_str}")
        
        best_snr = [f"{v:.2f}" for v in snr_score_list_sorted[0][2]]
        best_diff = [f"{v:.2f}" for v in snr_score_list_sorted[0][1]] if isinstance(snr_score_list_sorted[0][1], (list, np.ndarray)) else snr_score_list_sorted[0][1]
        print(f"Optimal SNR [single, enta]: {best_snr}, Difference: {best_diff}")
        return snr_score_list_sorted
    
    def calculate_gate_difference(self, gates1, gates2):
        """
        Calculate the difference between two gate configurations.
        
        Args:
            gates1: First gate configuration (list of gate specs)
            gates2: Second gate configuration (list of gate specs)
            
        Returns:
            Total difference count
        """
        if len(gates1) != len(gates2):
            return float('inf')
        
        diff_count = 0
        for g1, g2 in zip(gates1, gates2):            
            for i in range(len(g1)):
                if g1[i] != g2[i]:
                    diff_count += 1
        
        return diff_count
    
    def compare_and_mask(self, original_single, single):
        """
        Compare original_single and single, if corresponding positions are the same, assign 0,
        otherwise keep the value from single.
        
        Args:
            original_single: Original gate configuration
            single: Current gate configuration
            
        Returns:
            tuple: (masked_list, non_zero_count)
                - masked_list: List with same structure as single, but with 0s where values match original_single
                - non_zero_count: Number of non-zero elements (different positions)
        """
        if len(original_single) != len(single):
            # Count all elements as different if lengths don't match
            total_elements = sum(len(gate) for gate in single)
            return single, total_elements
        
        result = []
        diff_count = 0
        
        for orig_gate, curr_gate in zip(original_single, single):
            if len(orig_gate) != len(curr_gate):
                result.append(curr_gate)  # Keep original if sub-lengths don't match
                diff_count += len(curr_gate)  # Count all elements as different
                continue
            
            masked_gate = []
            for orig_val, curr_val in zip(orig_gate, curr_gate):
                if orig_val == curr_val:
                    masked_gate.append(-1)
                else:
                    masked_gate.append(curr_val)
                    diff_count += 1  # Count non-zero (different) elements
            result.append(masked_gate)
        
        return result, diff_count

    def apply_mask_to_single(self, mask_result, single_new):
        """
        Apply mask_result to single_new by copying non-zero values from mask_result
        to the corresponding positions in single_new, keeping other positions unchanged.
        
        Args:
            mask_result: Masked gate configuration (output from compare_and_mask)
            single_new: Target gate configuration to be modified
            
        Returns:
            Modified single_new with non-zero values from mask_result applied
        """
        if len(mask_result) != len(single_new):
            # If lengths don't match, return single_new unchanged
            print(f"Warning: Length mismatch - mask_result: {len(mask_result)}, single_new: {len(single_new)}")
            return single_new
        
        result = []
        
        for mask_gate, new_gate in zip(mask_result, single_new):
            if len(mask_gate) != len(new_gate):
                result.append(new_gate)  # Keep original if sub-lengths don't match
                continue
            
            modified_gate = []
            for mask_val, new_val in zip(mask_gate, new_gate):
                if mask_val != -1:  # Non-zero value from mask_result
                    modified_gate.append(mask_val)
                else:  # Zero value, keep original from single_new
                    modified_gate.append(new_val)
            result.append(modified_gate)
        
        return result

    def Langevin_update(self, x, snr=None, n_steps=20, step_size=0.01): 
        z, logvar = self.ROOT.classifier.arch_to_z([x])
        x_valid_list = []
        print(f"z max value: {torch.max(z).item()}")
        print(f"z min value: {torch.min(z).item()}")
        # Compute scaling factor c
        decoder = self.ROOT.classifier.GVAE_model.decoder
        decoder.eval()

        n_qubit = arch_code_fold[0]
        x_recon = decoder(z)
        mask = get_proj_mask(x_recon[0], n_qubit, n_qubit)
        gate_matrix = x_recon[0] + mask
        original_single, original_enta, _ = generate_single_enta(gate_matrix, n_qubit)

        # Compute optimal SNR if not provided
        if snr is None:
            snr = self.compute_optimal_snr(z, logvar)
        snr_sorted = [[item[2][0] for item in snr], [item[2][1] for item in snr]]

        total_nodes = z.shape[1]
        chunks1, chunks2 = [], []
        for i in range(0, total_nodes, 2*n_qubit):
            chunks1.append(z[:, i:i + n_qubit, :])
            chunks2.append(z[:, i + n_qubit:i + 2*n_qubit, :])
        z_single = torch.cat(chunks1, dim=1)
        z_enta = torch.cat(chunks2, dim=1)
        c1 = self.compute_scaling_factor(z_single, decoder, snr_sorted[0])  # snr_single
        c2 = self.compute_scaling_factor(z_enta, decoder, snr_sorted[1])  # snr_enta

        j = 0
        while len(x_valid_list) < 100 and j < len(c1):
            step_size = [c1[j], c2[j]]
            for i in range(1000):
                # noise = torch.randn_like(z)
                x_new = sample_normal(z, logvar,step_size, arch_code_fold)  # Use fold to adjust the step size
                x_new = decoder(x_new)
                mask = get_proj_mask(x_new[0], n_qubit, n_qubit)
                if is_valid_ops_adj(x_new[0], n_qubit):
                    gate_matrix = x_new[0] + mask
                    single,enta, _ = generate_single_enta(gate_matrix, n_qubit)
                    # update single and enta with the mask
                    single_mask, _ = self.compare_and_mask(original_single, single)
                    enta_mask, _ = self.compare_and_mask(original_enta, enta)
                    single = self.apply_mask_to_single(single_mask, self.explorations['single'])
                    enta = self.apply_mask_to_single(enta_mask, self.explorations['enta'])

                    if [single, enta] not in x_valid_list:
                        x_valid_list.append([single, enta])           
            if j < len(snr_sorted[0]) and j < len(snr_sorted[1]):
                print(f'Number of valid circuits: {len(x_valid_list)} of SNR [{snr_sorted[0][j]:.2f}, {snr_sorted[1][j]:.2f}]')
            else:
                print(f'Number of valid circuits: {len(x_valid_list)}')
            j += 1
        return x_valid_list

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        self.reset_node_data()
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def set_arch(self, phase, best_change):
        # if phase == 0:
        if len(best_change[0]) == len(self.explorations['single'][0]):          
            self.explorations['single'] = best_change
            # self.explorations['single'] = None
        else:
            self.explorations['enta'] = best_change
            # self.explorations['enta'] = None

        self.explorations['phase'] = phase        


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)

    def populate_prediction_data(self):
        # self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train(self.tree_height)


    def predict_nodes(self, method = None, dataset =None):
        for i in self.nodes:            
            if dataset:
                i.predict_validation()
            else:
                i.predict(self.explorations, method)


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag[0])
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        self.ROOT.counter += 1
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            if torch.rand(1) < curt_node.delta:
                # id = torch.randint(0, len(curt_node.kids), (1,))
                id = np.random.choice(np.argwhere(UCT == np.amin(UCT)).reshape(-1), 1)[0]
            else:
                id = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            curt_node = curt_node.kids[id]
            self.nodes[curt_node.id].counter += 1
        return curt_node
    
    def sampling_arch(self, number=10):
        print('Used Qubits:', self.qubit_used)
        h = 2 ** (self.tree_height-1) - 1
        if len(self.search_space) <= number:
            self.TASK_QUEUE += self.search_space
        else:
            for i in range(0, number):
                # select
                target_bin   = self.select()
                qubits = self.qubit_used
                sampled_arch = target_bin.sample_arch(qubits)
                # NOTED: the sampled arch can be None 
                if sampled_arch is not None:                    
                    # push the arch into task queue                
                    self.TASK_QUEUE.append(sampled_arch)                    
                    self.sample_nodes.append(target_bin.id-h)
                else:
                    # trail 1: pick a network from the left leaf
                    for n in self.nodes:
                        if n.is_leaf == True:
                            sampled_arch = n.sample_arch(qubits)
                            if sampled_arch is not None:
                                # print("\nselected node" + str(n.id-7) + " in leaf layer")                            
                                self.TASK_QUEUE.append(sampled_arch)                                
                                self.sample_nodes.append(n.id-h)
                                break
                            else:
                                continue
                # if type(sampled_arch[0]) == type([]):
                if get_list_dimensions(sampled_arch) == 2:
                    arch = sampled_arch[-1]
                else:
                    arch = sampled_arch
                self.search_space.remove(arch)        

    def insert_job(self, change_code, job_input):
        job = copy.deepcopy(job_input)
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job


    def evaluate_jobs_before(self):
        jobs = []
        designs =[]        
        archs = []
        nodes = []
        difference = []
        original_single = self.explorations['single']
        original_enta = self.explorations['enta']
        while len(self.TASK_QUEUE) > 0:            
           
            job = self.TASK_QUEUE.pop()
            try:
                sample_node = self.sample_nodes.pop()
            except IndexError:
                sample_node = None
            if type(job[0]) != type([]):
                job = [job]            
            # if self.explorations['phase'] == 0:
            if get_list_dimensions(job) < 3:
                if len(job[0]) == len(original_single[0]):
                    single = self.insert_job(original_single, job)
                    enta = original_enta
                else:
                    single = original_single
                    enta = self.insert_job(original_enta, job)
            else:
                single = job[0]
                enta = job[1]

                diff = difference_between_archs(original_single, original_enta, single, enta)
                difference.append(diff)
            design = single_enta_to_design(single, enta, self.ARCH_CODE, args.fold)
            arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)           

            jobs.append(job)
            designs.append(design)
            archs.append(arch)
            nodes.append(sample_node)

        mean_diff = np.mean(difference, axis=0) if difference else None
        print(f"\nMean differences for {len(difference)} circuits: {mean_diff}")
        self.mean_diff.append(mean_diff)        

        return jobs, designs, archs, nodes

    def evaluate_jobs_after(self, results, jobs, archs, nodes):
        performance_per_gate = []
        for i in range(len(jobs)):
            acc = results[i]
            job = jobs[i]  
            job_str = json.dumps(job)
            arch = archs[i]
            arch_str = json.dumps(np.int8(arch).tolist())
            
            penaly, gates = count_gates(arch, self.explorations['regular'])
            n_gates = gates['uploading'] + gates['single'] + gates['enta'] 
            if regular == True:
                p_acc = acc - penaly
            else:
                p_acc = acc                  
            
            performance_per_gate.append(acc/n_gates)
            self.samples[arch_str] = p_acc
            self.samples_true[arch_str] = acc
            self.samples_compact[job_str] = p_acc
            sample_node = nodes[i]
            print("job:", job_str, "acc:", acc, "p_acc:", p_acc)
            with open('results/{}.csv'.format(self.task_name), 'a+', newline='') as res:
                writer = csv.writer(res)                                        
                num_id = len(self.samples)
                writer.writerow([num_id, job_str, sample_node, acc, p_acc])
            self.mae_list.append(acc)
        self.performance_per_gate.append(np.mean(performance_per_gate))
        print('Performance per gate:', np.mean(performance_per_gate)) 
            

    def pre_search(self, iter):       
        # save current state
        self.ITERATION = iter
        if self.ITERATION > 0:
            self.dump_all_states(self.sampling_num + len(self.samples))
        print("\niteration:", self.ITERATION)
        if self.task_name == 'MOSI':
            period = 5
            number = 50
        else:
            period = 1
            number = 20

        if (self.ITERATION % period == 0): 
            if self.ITERATION == 0:
                self.init_train(number)                    
            else:
                self.re_init_tree()                                        

        # evaluate jobs:
        print("\nevaluate jobs...")
        self.mae_list = []
        jobs, designs, archs, nodes = self.evaluate_jobs_before()

        return jobs, designs, archs, nodes
    
    def post_search(self, jobs, results, archs, nodes):
                
        self.evaluate_jobs_after(results, jobs, archs, nodes)
        print("\nfinished all jobs in task queue")            

        # assemble the training data:
        print("\npopulate training data...")
        self.populate_training_data()
        print("finished")

        # training the tree
        print("\ntrain classifiers in nodes...")
        if torch.cuda.is_available():
            print("using cuda device")
        else:
            print("using cpu device")
        
        start = time.time()
        self.train_nodes()
        print("finished")
        end = time.time()
        print("Running time: %s seconds" % (end - start))
       
        # clear the data in nodes
        self.reset_node_data()                      

        print("\npopulate prediction data...")
        self.populate_prediction_data()
        print("finished")        
        print("\npredict and partition nets in search space...")
        self.predict_nodes() 
        self.check_leaf_bags()
        print("finished")
        print(self.ROOT.delta_history[-1])
        self.print_tree()
        # # sampling nodes
        # # nodes = [0, 1, 2, 3, 8, 12, 13, 14, 15]
        # nodes = [0, 3, 12, 15]
        # sampling_node(self, nodes, dataset, self.ITERATION)
        
        random.seed(self.ITERATION)
        # self.sampling_arch(10)
    
    def projection(self, arch_next, single, enta):
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


def Scheme_mp(design, job, task, weight, i, q=None):
    step = len(design)    
    if get_list_dimensions(job[0]) < 3:
        epoch = 1
    else:
        epoch = 2
   
    for j in range(step):
        print('Arch:', job[j])
        _, report = Scheme(design[j], task, weight, epoch, verbs=1)
        q.put([i*step+j, report['mae']])

def count_gates(arch, coeff=None):
    # x = [item for i in [2,3,4,1] for item in [1,1,i]]
    qubits = int(args.n_qubits / args.fold)
    layers = args.n_layers
    x = [[0, 0, i]*4 for i in range(1,qubits+1)] 
    x = np.transpose(x, (1,0))
    x = np.sign(abs(x-arch))
    # if coeff != None:
    #     coeff = np.reshape(coeff * 4, (-1,1))
    #     y = (x * coeff).sum()
    # else:
    #     y = 0
    stat = {}
    stat['uploading'] = x[[3*i for i in range(layers)]].sum()
    stat['single'] = x[[3*i+1 for i in range(layers)]].sum()
    stat['enta'] = x[[3*i+2 for i in range(layers)]].sum()

    y = coeff[0] * stat['single'] + coeff[1] * stat['uploading'] + coeff[2] * stat['enta']

    return y, stat

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    mean = np.mean(gate_stat, axis=0)
    return mean

def sampling_qubits(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

def create_agent(task, arch_code, pre_file, node=None):
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        if node: node_path = node        
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.samples), "samples")        
        print("=====>loads:", len(agent.TASK_QUEUE), 'tasks')
    else:        
        
        with open('search_space/search_space_mnist_4', 'rb') as file:
            search_space = pickle.load(file)

        n_qubits = int(arch_code[0] / args.fold)
        n_layers = arch_code[1]
        
        if task['task'] == 'MNIST_10':
            with open('search_space/search_space_mnist_10', 'rb') as file:
                search_space = pickle.load(file)

        agent = MCTS(search_space, 4, args.fold, arch_code)
        agent.task_name = task['task']+'_'+task['option']

        if pre_file in init_weights:
            agent.nodes[0].classifier.model.load_state_dict(torch.load(os.path.join(init_weight_path, pre_file)), strict= True)
       
        # strong entanglement
        # n_qubits = arch_code[0]
        n_layers = arch_code[1]
        
        single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

        # with open('data/random_circuits_mnist_4.json', 'rb') as file:
        #     circuits = json.load(file)

        # single = circuits[0]['single']
        # enta = circuits[0]['enta']
        # args.init_weight = 'random_circuits'
        
        agent.explorations['single'] = single
        agent.explorations['enta'] = enta
        
        design = single_enta_to_design(single, enta, arch_code, args.fold)
                
        if args.init_weight in init_weights:
            agent.weight = torch.load(os.path.join(init_weight_path, args.init_weight))
        else:            
            best_model, report = Scheme(design, task, 'init', 30, None, 'save')            
            agent.weight = best_model.state_dict()

            with open('results/{}_fine.csv'.format(task['task']+'_'+task['option']), 'a+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow([0, [single, enta], report['mae']]) 
        
    return agent


if __name__ == '__main__':
    
     # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    parser = argparse.ArgumentParser(description='Process some parameters.')    
    parser.add_argument('-pretrain', type=str, required=False, default='no_pretrain', help='filename of pretraining weights, e.g. pre_weights')
    import ast

    parser.add_argument(
        '-task',
        type=ast.literal_eval,  # dict
        required=False,
        default="{'task': 'MNIST'}",
        help="Task dict, e.g. \"{'task': 'QML_Linear_32d', 'n_qubits': 8, ...}\""
    )

    args_c = parser.parse_args()
    task = args_c.task
    
    task = {
    'task': 'QML_Linear_64d',
    'n_qubits': 16,
    'n_layers': 4,
    'fold': 4,
    'option': 'mix_reg',
    'regular': True,
    'num_processes': 2
    }

    # task = {
    # 'task': 'QML_Hidden_80d',
    # 'n_qubits': 20,
    # 'n_layers': 4,
    # 'fold': 5,
    # 'option': 'mix_reg',
    # 'regular': True,
    # 'num_processes': 2
    # }

    # task = {
    # 'task': 'MNIST_4',
    # 'option': 'mix_no_reg',
    # 'regular': True,
    # 'n_qubits': 4,
    # 'n_layers': 4,
    # 'fold': 1,
    # 'num_processes': 2
    # }

    # task = {
    # 'task': 'MNIST_10',
    # 'option': 'mix_reg',
    # 'regular': True,
    # 'n_qubits': 10,
    # 'n_layers': 4,
    # 'fold': 2,
    # 'num_processes': 2
    # }

    mp.set_start_method('spawn')

    saved = None
    # saved = 'states/mcts_agent_20'
    num_processes = task.get('num_processes', 1)            
    
    check_file(task['task']+'_'+task['option'])
    
    arch_code = [task['n_qubits'], task['n_layers']]
    arch_code_fold = [task['n_qubits'] // task['fold'], task['n_layers']]
    args = Arguments(**task)
    agent = create_agent(task, arch_code, args_c.pretrain, saved)
    ITERATION = agent.ITERATION
    debug = False
    regular = task.get('regular', False)


    for iter in range(ITERATION, 30):
        jobs, designs, archs, nodes = agent.pre_search(iter)
        # print(agent.weight['QuantumLayer.q_params_enta'][0])
        weight_for_mp = {}
        for k, v in agent.weight.items():
            if isinstance(v, torch.Tensor):
                weight_for_mp[k] = v.cpu().clone().detach()
            else:
                weight_for_mp[k] = v
        results = {}
        n_jobs = len(jobs)
        step = n_jobs // num_processes
        res = n_jobs % num_processes        
        if not debug:
            with Manager() as manager:
                q = manager.Queue()
                with mp.Pool(processes = num_processes) as pool:        
                    pool.starmap(Scheme_mp, [(designs[i*step : (i+1)*step], jobs[i*step : (i+1)*step], task, weight_for_mp, i, q) for i in range(num_processes)])            
                    pool.starmap(Scheme_mp, [(designs[n_jobs-i-1 : n_jobs-i], jobs[i*step : (i+1)*step], task, weight_for_mp, n_jobs-i-1, q) for i in range(res)])
                while not q.empty():
                    [i, acc] = q.get()
                    results[i] = acc
        else:
            for i in range(n_jobs):
                results[i] = random.uniform(0.75, 0.8)

        agent.post_search(jobs, results, archs, nodes)

    print('The best model: ', agent.best['acc'])
    agent.dump_all_states(agent.sampling_num + len(agent.samples))
    # plot_2d_array(agent.best['model'])
    
    Range = [0.8, 0.82]
    rank = 20

    print('<{}:'.format(Range[0]), sum(value < Range[0] for value in list(agent.samples_true.values())))
    print('({}, {}):'.format(Range[0], Range[1]), sum((value in Range)  for value in list(agent.samples_true.values())))
    print('>{}:'.format(Range[1]), sum(value > Range[1]  for value in list(agent.samples_true.values())))
    # print('Gate numbers of top {}: {}'.format(rank, analysis_result(agent.samples_true, rank)))