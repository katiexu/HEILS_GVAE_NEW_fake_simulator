class Arguments:
    def __init__(self, **kwargs): #task, n_qubits, n_layers, fold):

        n_qubits = kwargs.get('n_qubits', 4)
        n_layers = kwargs.get('n_layers', 4)
        task = kwargs.get('task', 'MNIST')
        fold = kwargs.get('fold', 1)
        self.backend = kwargs.get('backend', 'tq')

        # Qiskit arguments
        self.use_noise_model_train = False
        self.use_noise_model_inference = True
        self.backend_device = 'CPU'

        # self.device     = 'cpu'
        self.device     = 'cuda'
        self.clr        = 0.005
        self.qlr        = 0.01

        self.allowed_gates = ['Identity', 'U3', 'data', 'data+U3'] #['Identity', 'RX', 'RY', 'RZ']
        
        self.task      = task
        self.n_qubits   = n_qubits        
        self.epochs     = 1
        self.batch_size = 256
        self.sampling = 5

        self.n_layers = n_layers      
        self.exploration = [0.001, 0.002, 0.003]
        
        self.backend    = 'tq'      # 'tq' for TorchQuantum; 'qi' for Qiskit Aer Simulator; 'qml' for PennyLane
        self.noise = False
        self.digits_of_interest = [0, 1, 2, 3]
        self.train_valid_split_ratio = [0.95, 0.05]
        self.center_crop = 24
        self.resize = 28
        self.file_single = 'search_space/search_space_mnist_single'
        self.file_enta   = 'search_space/search_space_mnist_enta'
        self.kernel      = 6
        self.fold        = fold
        self.init_weight = 'init_weight_'+ task
        self.SNR = 0.5
        self.strategy = 'mix'
        self.qiskit_seed=170
        self.shots=10000
        self.name='yorktown'


        if task == ('MNIST_10' or 'FASHION_10'):
            self.n_qubits   = n_qubits
            
            self.epochs     = 1
            self.batch_size = 256 
            self.sampling = 5
            self.kernel      = 4

            self.n_layers = n_layers
            self.base_code = [self.n_layers, 2, 3, 4, 1]
            self.exploration = [0.001, 0.002, 0.003]

            self.backend    = 'tq'
            self.digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.file_single = 'search_space/search_space_mnist_half_single'
            self.file_enta   = 'search_space/search_space_mnist_half_enta'
            self.fold        = fold

        if task.startswith('QML'):            
            self.batch_size = 32
            self.task_name = task
            
            # Set path based on task content
            if 'Hidden' in task:
                self.path = 'hidden_manifold'
            elif 'Linear' in task:
                self.path = 'linearly_separable'
            else:
                self.path = None  # or set a default path
            
