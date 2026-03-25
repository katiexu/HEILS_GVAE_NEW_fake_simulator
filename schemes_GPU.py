import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, qml_Dataloaders
from FusionModel import QNet
from FusionModel import translator, single_enta_to_design

from Arguments import Arguments
import random




def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)

    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in data_loader:
        images = feed_dict['image'].cuda()  # Force GPU
        targets = feed_dict['digit'].cuda()  # Force GPU
        optimizer.zero_grad()
        output = model(images, args.n_qubits, args.task)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor().cuda()  # Force GPU
    output_all = torch.Tensor().cuda()  # Force GPU
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].cuda()  # Force GPU
            targets = feed_dict['digit'].cuda()  # Force GPU
            output = model(images, args.n_qubits, args.task)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0) 
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    
    return total_loss, accuracy

def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].cuda()  # Force GPU
            targets = feed_dict['digit'].cuda()  # Force GPU
            output = model(images, args.n_qubits, args.task)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy    
    return metrics

def Scheme_eval(design, task, weight):
    result = {}  
    args = Arguments(**task) 
    path = 'weights/'  
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).cuda()  # Force GPU
    model.load_state_dict(torch.load(path+weight, map_location='cuda'), strict= False)  # Force GPU
    result['mae'] = evaluate(model, test_loader, args)
    return model, result

def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    # Force CUDA seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = Arguments(**task)
    if epochs == None:
        epochs = args.epochs
    
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).cuda()  # Force GPU
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict= False)
        else:            
            model.load_state_dict(torch.load('init_weights/base_fashion', map_location='cuda'))  # Force GPU
    criterion = nn.NLLLoss().cuda()  # Force GPU
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 0
    start = time.time()
    best_model = model
    
    print("Force using GPU for training and inference")
    if epochs == 0:
        print('No training epochs specified, skipping training.')        
    else:        
        for epoch in range(epochs):
            try:
                train(model, train_loader, optimizer, criterion, args)
            except Exception as e:
                print('No parameter gate exists')
            train_loss = test(model, train_loader, criterion, args)
            train_loss_list.append(train_loss)        
            val_loss = evaluate(model, val_loader, args)
            val_loss_list.append(val_loss)
            metrics = evaluate(model, test_loader, args)
            val_loss = 0.5 *(val_loss+train_loss[-1])
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics, 'saving model')
                best_model = copy.deepcopy(model)           
            else:
                if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics)        
    end = time.time()        
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    print("Running time: %s seconds" % (end - start))
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics}
    
    if save:
        torch.save(best_model.state_dict(), 'init_weights/init_weight_' + task['task'])
    return best_model, report

def pretrain(design, task, weight):    

    args = Arguments(**task)
    
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).cuda()  # Force GPU
    model.load_state_dict(weight, strict= True)
    
    val_loss = evaluate(model, val_loader, args)
    display(val_loss)
    
    return val_loss


if __name__ == '__main__':
    task = {
    'task': 'MNIST_10',
    'option': 'mix_reg',
    'n_qubits': 10,
    'n_layers': 4,
    'fold': 2
    }

    # task = {
    # 'task': 'MNIST_4',
    # 'option': 'mix_reg',
    # 'n_qubits': 4,
    # 'n_layers': 4,
    # 'fold': 1
    # }

    task = {
    'task': 'QML_Hidden_80d',
    'n_qubits': 20,
    'n_layers': 4,
    'fold': 5,
    'option': 'mix_reg',
    'regular': True,
    'num_processes': 2
    }
    
    arch_code = [task['n_qubits'], task['n_layers']]
    args = Arguments(**task)
    n_layers = arch_code[1]
    n_qubits = int(arch_code[0] / args.fold)
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    # single = [[5, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1], [2, 0, 1, 1, 1, 1, 1, 0, 1], [3, 0, 1, 1, 1, 1, 1, 1, 1], [4, 0, 1, 1, 1, 0, 1, 1, 1]]
    # enta =  [[1, 2, 2, 3, 2], [2, 1, 3, 3, 5], [3, 2, 2, 1, 4], [4, 1, 1, 2, 2], [5, 1, 2, 4, 4]]

    
    # design = translator(single, enta, 'full', arch_code, args.fold)
    # design = op_list_to_design(op_list, arch_code)
    design = single_enta_to_design(single, enta, arch_code, args.fold)

    best_model, report = Scheme(design, task, 'init', 30, verbs=False, save=True)
    

    # torch.save(best_model.state_dict(), 'weights/base_fashion')