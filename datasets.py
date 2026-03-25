import pickle
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset
from Arguments import Arguments
import pandas as pd
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class CustomDataset(Dataset):
    def __init__(self, audio, visual, text, target):
        self.audio = audio
        self.visual = visual
        self.text = text
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        audio_val = self.audio[index]
        visual_val = self.visual[index]
        text_val = self.text[index]
        target = self.target[index]
        return audio_val, visual_val, text_val, target


def MOSIDataLoaders(args):
    with open('data/mosi', 'rb') as file:
        tensors = pickle.load(file)
    
    AUDIO = 'COVAREP'
    VISUAL = 'FACET_4.2'
    TEXT = 'glove_vectors'
    TARGET = 'Opinion Segment Labels'

    train_data = tensors[0]
    train_audio = torch.from_numpy(train_data[AUDIO]).float()
    train_visual = torch.from_numpy(train_data[VISUAL]).float()
    train_text = torch.from_numpy(train_data[TEXT]).float()
    train_target = torch.from_numpy(train_data[TARGET]).squeeze()

    val_data = tensors[1]
    val_audio = torch.from_numpy(val_data[AUDIO]).float()
    val_visual = torch.from_numpy(val_data[VISUAL]).float()
    val_text = torch.from_numpy(val_data[TEXT]).float()
    val_target = torch.from_numpy(val_data[TARGET]).squeeze()

    test_data = tensors[2]
    test_audio = torch.from_numpy(test_data[AUDIO]).float()
    test_visual = torch.from_numpy(test_data[VISUAL]).float()
    test_text = torch.from_numpy(test_data[TEXT]).float()
    test_target = torch.from_numpy(test_data[TARGET]).squeeze()

    train = CustomDataset(train_audio, train_visual, train_text, train_target)
    val = CustomDataset(val_audio, val_visual, val_text, val_target)
    test = CustomDataset(test_audio, test_visual, test_text, test_target)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=len(val), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=len(test), shuffle = False, pin_memory=True)
    return train_loader, val_loader, test_loader

import torch
from torchquantum.datasets import MNIST


def MNISTDataLoaders(args, task):
    if task in ('MNIST_4', 'MNIST_10'):
        FAHION = False
    else:
        FAHION = True
    dataset = MNIST(
        root='data',
        train_valid_split_ratio=args.train_valid_split_ratio,
        center_crop=args.center_crop,
        resize=args.resize,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=args.digits_of_interest,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=FAHION,
        n_train_samples=None
        )
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            batch_size = args.batch_size
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = len(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True)

    return dataflow['train'], dataflow['valid'], dataflow['test']


class MyDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        img = self.data[idx]
        digit=self.labels[idx]
        return {"image": img, "digit": digit}


def reshape_to_target(tensor):
    """
    将 (m, 1, n) 的 Tensor 转换为 (m, 1, 16)
    处理逻辑：
    1. 如果 n < 16：用0填充到16
    2. 如果 n > 16 且是完全平方数：转为2D后池化到4x4=16
    3. 其他情况：用1D池化降到16
    """
    m, _, n = tensor.shape

    if n == 16:
        return tensor

    # 情况1：n < 16，填充0
    if n < 16:
        pad_size = 16 - n
        return torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
    
    if  n % 4 == 0:
        return tensor

    # 情况2：n > 16 且是完全平方数
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        # 转为2D (假设可以合理reshape)
        try:
            # 先转为 (m, 1, sqrt_n, sqrt_n)
            tensor_2d = tensor.view(m, 1, sqrt_n, sqrt_n)
            # 自适应池化到 (4,4)
            pool = nn.AdaptiveAvgPool2d((4, 4))
            pooled = pool(tensor_2d)
            return pooled.view(m, 1, 16)
        except:
            # 如果reshape失败，降级到1D池化
            pass

    # 情况3：其他情况使用1D池化
    pool = nn.AdaptiveAvgPool1d(16)
    return pool(tensor)
def create_dataloader(args,train,val,test):
    train_data = reshape_to_target(torch.from_numpy(train.iloc[:, :-1].values.astype(np.float32)).unsqueeze(1))
    train_labels = torch.from_numpy(train.iloc[:, -1].values.astype(np.int64))
    val_data = reshape_to_target(torch.from_numpy(val.iloc[:, :-1].values.astype(np.float32)).unsqueeze(1))
    val_labels = torch.from_numpy(val.iloc[:, -1].values.astype(np.int64))
    test_data = reshape_to_target(torch.from_numpy(test.iloc[:, :-1].values.astype(np.float32)).unsqueeze(1))
    test_labels = torch.from_numpy(test.iloc[:, -1].values.astype(np.int64))
    train_labels =torch.where(train_labels == -1, torch.tensor(0), train_labels)
    val_labels =torch.where(val_labels == -1, torch.tensor(0), val_labels)
    test_labels = torch.where(test_labels == -1, torch.tensor(0),test_labels)
    train_dateset = MyDataset(train_data, train_labels)
    val_dateset = MyDataset(val_data, val_labels)
    test_dateset = MyDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dateset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dateset)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dateset,
        batch_size=len(val_dateset),
        # sampler=RandomSampler(val_dateset)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dateset,
        batch_size=len(test_dateset),
        # sampler=RandomSampler(test_dateset)
    )
    return train_loader, val_loader, test_loader

def qml_Dataloaders(args):
    train=pd.read_csv(f'benchmarks/{args.path}/{args.task_name}_train.csv',header=None)
    val =pd.read_csv(f'benchmarks/{args.path}/{args.task_name}_val.csv',header=None)
    test=pd.read_csv(f'benchmarks/{args.path}/{args.task_name}_test.csv',header=None)
    return create_dataloader(args, train, val, test)

def myBarsAndStripes(args,size):
    train=pd.read_csv(f'benchmarks/bars_and_stripes/bars_and_stripes_{size}_x_{size}_0.5noise_train.csv',header=None)
    test=pd.read_csv(f'benchmarks/bars_and_stripes/bars_and_stripes_{size}_x_{size}_0.5noise_test.csv',header=None)
    return create_dataloader(args,train,test)

def myhyperplanes(args,dim_hyperplanes,n_hyperplanes):
    train=pd.read_csv(f'benchmarks/hyperplanes_diff/hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_train.csv',header=None)
    test=pd.read_csv(f'benchmarks/hyperplanes_diff/hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_test.csv',header=None)
    return create_dataloader(args, train, test)

def myminist_cg(args,a,b,height):
    train=pd.read_csv(f'benchmarks/mnist_cg/mnist_pixels_{a}-{b}_{height}x{height}_train.csv',header=None)
    test=pd.read_csv(f'benchmarks/mnist_cg/mnist_pixels_{a}-{b}_{height}x{height}_test.csv',header=None)
    return create_dataloader(args, train, test)
def mytwo_curves(args,n_features):
    train=pd.read_csv(f'benchmarks/two_curves_diff/two_curves-5degree-0.1offset-{n_features}d_train.csv',header=None)
    test=pd.read_csv(f'benchmarks/two_curves_diff/two_curves-5degree-0.1offset-{n_features}d_test.csv',header=None)
    return create_dataloader(args, train, test)



from tqdm import tqdm
if __name__ == '__main__':
    trainloader, validloader, testloader = qml_Dataloaders(Arguments())
    for feed_dict in tqdm(trainloader):
        images = feed_dict['image'].to('cpu')
        targets = feed_dict['digit'].to('cpu')
        print(f'images_shape: {images.shape}, targets_shape: {targets.shape}')
