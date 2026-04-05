import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders
from FusionModel import QNet
from FusionModel import translator, single_enta_to_design

from Arguments import Arguments
import random
from tqdm import tqdm

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
    print(YELLOW + "Test Accuracy: {}\n".format(metrics) + RESET)

    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in tqdm(data_loader, desc="Training", disable=True):
    # for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)    
        optimizer.zero_grad()
        output = model(images, args.n_qubits, args.task)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor().to(args.device)
    output_all = torch.Tensor().to(args.device)
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
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
    if args.backend == 'qi':
        tqdm_disable = False
    else:
        tqdm_disable = True
    
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)
            output = model(images, args.n_qubits, args.task)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy
    # metrics = output.cpu().numpy()
    return metrics

def Scheme_eval(design, task, weight, backend, noise=None,seed=170,shots=10000,name='yorktown'):
    result = {}  
    args = Arguments(**task)
    args.backend = backend
    args.noise = noise
    args.qiskit_seed=seed
    args.shots = shots
    args.name=name
    path = 'weights/'  
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load(path+weight), strict= False)
    result['acc'] = evaluate(model, test_loader, args)
    return result


def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
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
    model = QNet(args, design).to(args.device)
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict= False)
        else:            
            model.load_state_dict(torch.load('init_weights/base_fashion'))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 0
    start = time.time()
    best_model = model   

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
    model = QNet(args, design).to(args.device)
    model.load_state_dict(weight, strict= True)
    
    val_loss = evaluate(model, val_loader, args)
    display(val_loss)
    
    return val_loss


import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compare_two_models(design, weight, task, backend1='tq', backend2='qi',
                       noise=None, shots=10000, save_path='model_comparison.png'):
    """
    Compare predictions of two models and generate visualizations

    Parameters:
    - design: Model architecture design
    - weight: Path to model weights
    - task: Task configuration dictionary
    - backend1: First backend name
    - backend2: Second backend name
    - noise: Noise configuration
    - shots: Number of shots for quantum execution
    - save_path: Path to save the output image

    Returns:
    - metrics: Dictionary containing various evaluation metrics
    - fig: Generated visualization figure
    """

    # Assuming necessary imports and class definitions are available
    # from arguments import Arguments
    # from models import QNet
    # from dataloaders import qml_Dataloaders, MNISTDataLoaders
    # from prediction_utils import get_model_predictions

    # Load model 1
    args1 = Arguments(**task)
    args1.backend = backend1
    args1.noise = noise
    args1.shots = shots

    # Initialize data loader
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args1)
    else:
        dataloader = MNISTDataLoaders(args1, task['task'])

    # Extract data loaders - corrected to access loader attributes
    train_loader, val_loader, test_loader= dataloader

    # Load model
    model1 = QNet(args1, design).to(args1.device)
    weight_path = f'weights/{weight}' if not weight.startswith('weights/') else weight
    model1.load_state_dict(torch.load(weight_path), strict=False)

    # Get model 1 predictions
    model1_probs, model1_preds, targets = get_model_predictions(model1, test_loader, args1)

    # Load model 2
    args2 = Arguments(**task)
    args2.backend = backend2
    args2.noise = noise
    args2.shots = shots

    model2 = QNet(args2, design).to(args2.device)
    model2.load_state_dict(torch.load(weight_path), strict=False)

    # Get model 2 predictions
    model2_probs, model2_preds, _ = get_model_predictions(model2, test_loader, args2)

    # Calculate evaluation metrics
    n_samples = len(model1_probs)
    n_classes = model1_probs.shape[1]

    # Calculate margins for model 1
    sorted_probs1 = np.sort(model1_probs, axis=1)
    margins1 = sorted_probs1[:, -1] - sorted_probs1[:, -2]

    # Calculate MAE between two models' probability vectors
    mae_per_sample = np.mean(np.abs(model1_probs - model2_probs), axis=1)

    # Determine prediction consistency
    consistent = (model1_preds == model2_preds)
    consistency_rate = np.mean(consistent)

    # Calculate accuracy
    accuracy1 = np.mean(model1_preds == targets)
    accuracy2 = np.mean(model2_preds == targets)

    # Calculate error-to-margin ratio
    margin_mae_ratio = mae_per_sample / (margins1 + 1e-8)  # Avoid division by zero

    # Find optimal decision boundary
    try:
        # Use logistic regression to find optimal decision boundary
        X = np.column_stack([margins1, mae_per_sample])
        y = consistent.astype(int)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        # Decision boundary: w0*margin + w1*mae + b = 0
        w0, w1 = clf.coef_[0]
        b = clf.intercept_[0]

        # Calculate decision boundary accuracy
        y_pred = clf.predict(X)
        decision_boundary_acc = accuracy_score(y, y_pred)

        # Decision boundary equation
        decision_boundary_eq = lambda x: (-w0 * x - b) / w1
    except Exception as e:
        # If logistic regression fails, use simple threshold
        print(f"Logistic regression failed: {e}, using median threshold")
        decision_boundary_eq = lambda x: np.median(mae_per_sample) * np.ones_like(x)
        decision_boundary_acc = None

    # Create visualization figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Main plot: Margin vs Model MAE
    ax1 = axes[0, 0]

    # Plot scatter points
    scatter_consistent = ax1.scatter(margins1[consistent], mae_per_sample[consistent],
                                     c='green', alpha=0.6, s=20, label='Consistent predictions')
    scatter_inconsistent = ax1.scatter(margins1[~consistent], mae_per_sample[~consistent],
                                       c='red', alpha=0.6, s=20, label='Inconsistent predictions')

    # Plot decision boundary
    x_range = np.linspace(margins1.min(), margins1.max(), 100)
    y_boundary = decision_boundary_eq(x_range)
    ax1.plot(x_range, y_boundary, 'b--', linewidth=2, label='Decision boundary')

    # Fix: Set y-axis range explicitly
    ax1.set_ylim(0, mae_per_sample.max() * 1.1)  # Add 10% padding

    ax1.set_xlabel('Model 1 Margin (Margin₁)', fontsize=12)
    ax1.set_ylabel('Inter-model Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title(f'A. Margin vs Inter-model MAE (Consistency: {consistency_rate:.2%})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Margin distribution
    ax2 = axes[0, 1]
    bins = 30
    ax2.hist(margins1[consistent], bins=bins, alpha=0.5, label='Consistent', density=True, color='green')
    ax2.hist(margins1[~consistent], bins=bins, alpha=0.5, label='Inconsistent', density=True, color='red')

    # Calculate and display key quantiles
    if np.any(consistent):
        q25_consistent = np.percentile(margins1[consistent], 25)
        q50_consistent = np.percentile(margins1[consistent], 50)
        q75_consistent = np.percentile(margins1[consistent], 75)
        ax2.axvline(x=q50_consistent, color='green', linestyle=':', linewidth=1.5,
                    label=f'Consistent median: {q50_consistent:.3f}')

    if np.any(~consistent):
        q50_inconsistent = np.percentile(margins1[~consistent], 50)
        ax2.axvline(x=q50_inconsistent, color='red', linestyle=':', linewidth=1.5,
                    label=f'Inconsistent median: {q50_inconsistent:.3f}')

    ax2.set_xlabel('Model 1 Margin (Margin₁)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('B. Margin Distribution Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MAE distribution
    ax3 = axes[1, 0]
    ax3.hist(mae_per_sample[consistent], bins=bins, alpha=0.5, label='Consistent', density=True, color='green')
    ax3.hist(mae_per_sample[~consistent], bins=bins, alpha=0.5, label='Inconsistent', density=True, color='red')

    # Calculate and display key quantiles
    if np.any(consistent):
        q50_consistent_mae = np.percentile(mae_per_sample[consistent], 50)
        ax3.axvline(x=q50_consistent_mae, color='green', linestyle=':', linewidth=1.5,
                    label=f'Consistent MAE median: {q50_consistent_mae:.3f}')

    if np.any(~consistent):
        q50_inconsistent_mae = np.percentile(mae_per_sample[~consistent], 50)
        ax3.axvline(x=q50_inconsistent_mae, color='red', linestyle=':', linewidth=1.5,
                    label=f'Inconsistent MAE median: {q50_inconsistent_mae:.3f}')

    ax3.set_xlabel('Inter-model Mean Absolute Error (MAE)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('C. MAE Distribution Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Prepare statistics text
    stats_text = f"""Model Comparison Statistics:

    Model 1 Accuracy ({backend1}): {accuracy1:.4f}
    Model 2 Accuracy ({backend2}): {accuracy2:.4f}
    Accuracy Difference: {abs(accuracy1 - accuracy2):.4f}

    Prediction Consistency Rate: {consistency_rate:.4f}
    Consistent Samples: {np.sum(consistent)}
    Inconsistent Samples: {np.sum(~consistent)}

    Margin Statistics:
      Overall Mean: {np.mean(margins1):.4f}
      Consistent Mean: {np.mean(margins1[consistent]):.4f}
      Inconsistent Mean: {np.mean(margins1[~consistent]):.4f}

    MAE Statistics:
      Overall Mean: {np.mean(mae_per_sample):.4f}
      Consistent Mean: {np.mean(mae_per_sample[consistent]):.4f}
      Inconsistent Mean: {np.mean(mae_per_sample[~consistent]):.4f}

    MAE/Margin Ratio:
      Overall Mean: {np.mean(mae_per_sample / (margins1 + 1e-8)):.4f}
      Consistent Mean: {np.mean(mae_per_sample[consistent] / (margins1[consistent] + 1e-8)):.4f}
      Inconsistent Mean: {np.mean(mae_per_sample[~consistent] / (margins1[~consistent] + 1e-8)):.4f}
    """

    if decision_boundary_acc is not None:
        stats_text += f"\nDecision Boundary Classification Accuracy: {decision_boundary_acc:.4f}"

    # Add statistics text box
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add title
    ax4.set_title('D. Statistics Summary', fontsize=14, y=1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Return statistics metrics
    metrics = {
        'accuracy1': accuracy1,
        'accuracy2': accuracy2,
        'consistency_rate': consistency_rate,
        'n_consistent': np.sum(consistent),
        'n_inconsistent': np.sum(~consistent),
        'margin_mean': np.mean(margins1),
        'margin_mean_consistent': np.mean(margins1[consistent]),
        'margin_mean_inconsistent': np.mean(margins1[~consistent]),
        'mae_mean': np.mean(mae_per_sample),
        'mae_mean_consistent': np.mean(mae_per_sample[consistent]),
        'mae_mean_inconsistent': np.mean(mae_per_sample[~consistent]),
        'mae_margin_ratio': np.mean(mae_per_sample / (margins1 + 1e-8)),
        'mae_margin_ratio_consistent': np.mean(mae_per_sample[consistent] / (margins1[consistent] + 1e-8)),
        'mae_margin_ratio_inconsistent': np.mean(mae_per_sample[~consistent] / (margins1[~consistent] + 1e-8)),
        'decision_boundary_accuracy': decision_boundary_acc,
    }

    return metrics, fig


def get_model_predictions(model, data_loader, args):
    """
    获取模型在数据加载器上的预测结果

    参数:
    - model: 模型
    - data_loader: 数据加载器
    - args: 参数

    返回:
    - probs: 预测概率 [n_samples, n_classes]
    - preds: 预测类别 [n_samples]
    - targets: 真实标签 [n_samples]
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            batch_targets = feed_dict['digit'].to(args.device)

            output = model(images, args.n_qubits, args.task)

            probs = output
            preds = torch.argmax(output, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    # 合并所有批次
    probs = np.vstack(all_probs)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    return probs, preds, targets





if __name__ == '__main__':
    # task = {
    # 'task': 'MNIST_10',
    # 'option': 'mix_reg',
    # 'n_qubits': 10,
    # 'n_layers': 4,
    # 'fold': 2
    # }

    task = {
    'task': 'MNIST_4',
    'option': 'mix_reg',
    'n_qubits': 4,
    'n_layers': 4,
    'fold': 1
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

    # best_model, report = Scheme(design, task, 'init', 1, verbs=False, save=False)
    # torch.save(best_model.state_dict(), 'weights/tmp')
   
    # result = Scheme_eval(design, task, 'tmp_4', backend='tq')
    # display(result)
    # # result = Scheme_eval(design, task, 'tmp_4', backend='qi',noise=False)
    # # display(result)
    # for shots in [1000, 10000]:
    #     result = Scheme_eval(design, task, 'tmp_4', backend='qi', noise=False, seed=170, shots=shots)
    #     display(result)
    # result = Scheme_eval(design, task, 'tmp_4', backend='qi', noise=False, seed=170, shots=10000)
    # display(result)
    for name in ['kolkata','nairobi','montreal','toronto','bel','sant']:
        for shots in [10000, 1000]:
            result = Scheme_eval(design, task, 'tmp_4', backend='qi',noise=True,seed=170,shots=shots,name=name)
            display(result)
    # compare_two_models(design=design,weight='tmp_4',task=task,backend1='tq',backend2='qi',noise=True,shots=10000,save_path='two_models_comparison.png')

    # torch.save(best_model.state_dict(), 'weights/base_fashion')