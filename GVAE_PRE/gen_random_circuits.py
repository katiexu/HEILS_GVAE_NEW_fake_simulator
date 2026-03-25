import random
import pickle

def filter_start_with(search_space, start_value):
            return [lst for lst in search_space if lst[0] == start_value]

def generate_random_circuits(num_circuits, num_qubits, search_space_single, search_space_enta):
    circuits_list = []

    with open(search_space_single, 'rb') as f:
         search_space_single = pickle.load(f)
    with open(search_space_enta, 'rb') as f:
         search_space_enta = pickle.load(f)

    for _ in range(num_circuits):
        circ = {}

        selected_single = []
        selected_enta = []
        for start_value in range(1, num_qubits + 1):
            candidates_single = filter_start_with(search_space_single, start_value)
            selected_single.append(random.sample(candidates_single, 1)[0])

            candidates_enta = filter_start_with(search_space_enta, start_value)
            selected_enta.append(random.sample(candidates_enta, 1)[0])

        circ['single']  = selected_single
        circ['enta']    = selected_enta
        circuits_list.append(circ)

    return circuits_list

if __name__ == '__main__':
    task = 'mnist'
    search_space_single = f'search_space/search_space_{task}_single'
    search_space_enta = f'search_space/search_space_{task}_enta'

    n_circuits = 10000
    n_qubits = 4
    circuits_list = generate_random_circuits(n_circuits, n_qubits, search_space_single, search_space_enta)

    import json

    with open(f'data/random_circuits_{task}_{n_qubits}_{n_circuits}.json', 'w') as f:
        json.dump(circuits_list, f)

    with open(f'data/random_circuits_{task}_{n_qubits}_{n_circuits}.json', 'r') as f:
        data = json.load(f)

    print(f"Generated {len(data)} random circuits for task {task} with {n_qubits} qubits.")
