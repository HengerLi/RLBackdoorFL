
import numpy as np
import torch
from torchvision import datasets, transforms
from math import floor
import random
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset


__all__ =['get_datasets','add_pattern_bd','poison_dataset','DatasetSplit']

def get_datasets(data):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'

    if data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform) 
    
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  
        
    return train_dataset, test_dataset


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())
    
    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx][d] = 0
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0

    elif dataset == 'mnist':
        if pattern_type == 'square':
            for i in range(5, 7):
                for j in range(6, 11):
                    x[i, j] = 255
                              
    elif dataset == 'fmnist':    
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255
            
        elif pattern_type == 'plus':
            start_idx = 5
            size = 2
            if agent_idx == -1:
                # vertical line  
                for i in range(start_idx, start_idx+size+1):
                    x[i, start_idx] = 255
            
                # horizontal line
                for i in range(start_idx-size//2, start_idx+size//2 + 1):
                    x[start_idx+size//2, i] = 255
            else:# DBA attack
                #upper part of vertical 
                if agent_idx % 4 == 0:
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx] = 255
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for i in range(start_idx+(size//2), start_idx+size+1):
                        x[i, start_idx] = 255
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for i in range(start_idx-size//2, start_idx+size//4+1):
                        x[start_idx+size//2, i] = 255
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for i in range(start_idx-size//4, start_idx+size//2+1):
                        x[start_idx+size//2, i] = 255

            
    return x 

def poison_dataset(dataset, data, base_class, target_class, poison_frac, pattern_type, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        #if args.data == 'fedemnist':
            #clean_img = dataset.inputs[idx]
        #else:
        clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, data, pattern_type=pattern_type, agent_idx=agent_idx)
        #if args.data == 'fedemnist':
             #dataset.inputs[idx] = torch.tensor(bd_img)
        #else:
        dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = target_class    
    return 

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        
    def classes(self):
        return torch.unique(self.targets)    

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target
