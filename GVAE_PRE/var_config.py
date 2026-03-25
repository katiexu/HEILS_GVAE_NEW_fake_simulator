### Configure gloabal variables ###

# max_depth = 5
num_qubits = 4
num_layers = 4
# num_gates = 10 ### TODO: Can it be not defined in advance? (different-size Tensor), maybe padding with I gate
# max_gates = 32
num_circuits = 100
allowed_gates = ['Identity', 'U3', 'data', 'data+U3', 'C(U3)']