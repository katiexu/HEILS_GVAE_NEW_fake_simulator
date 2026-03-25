import pickle
import random
from itertools import product

num_layers = 4
num_qubits = 6
num_single = 2 * num_layers
num_enta = num_layers


def generate_binary_list(n, m=2):
    return [
        list(seq)
        for seq in product(range(m), repeat=n)
        # if not all(x == m - 1 for x in seq)
    ]


single = generate_binary_list(num_single, 2)
single_list = []
double_list = []
final_list = []
for i in range(1, num_qubits + 1):
    for s in single:
        item = [i] + s
        single_list.append(item)
double = generate_binary_list(num_enta, 6)
double = [[x + 1 for x in row] for row in double]
for j in range(1, num_qubits + 1):
    for s in double:
        item = [j] + s
        double_list.append(item)
# final_list = single_list + double_list

# with open('search_space', 'wb') as file:
#     pickle.dump(final_list, file)

with open(f'search_space/search_space_mnist_{num_qubits}_single', 'wb') as file:
    pickle.dump(single_list, file)

with open(f'search_space/search_space_mnist_{num_qubits}_enta', 'wb') as file:
    pickle.dump(double_list, file)
