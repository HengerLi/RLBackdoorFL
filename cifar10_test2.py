import torch
import random
import numpy as np
import more_itertools as mit

from Aggr import *
from DataProcess import *
from Networks import *
from Util import *
from Post_defenses import *

import copy
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
from Backdoor_attacks import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
setup = dict(device=DEVICE, dtype=torch.float)

batch_size = 128
num_clients= 100
subsample_rate= 0.1
num_attacker= 10
num_class = 10
fl_epoch=1000
lr=0.05
num_class=10
#max_norm=2


train_dataset, val_dataset = get_datasets('cifar10')
base_class=0
data='cifar10'
target_class=9
poison_frac=1
pattern_type='plus'

# poison the validation dataset and trainning dataset
idxs = (val_dataset.targets == base_class).nonzero().flatten().tolist()
poisoned_val_set = DatasetSplit(copy.deepcopy(val_dataset), idxs)
poison_dataset(poisoned_val_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=True)
poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=batch_size, shuffle=False)


idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=False)
poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
globle_poisoned_train_iter = mit.seekable(poisoned_train_loader)

poisoned_train_iter_lis=[]
for i in range(4):
    idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
    poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
    poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=False, agent_idx=i)
    poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
    poisoned_train_iter = mit.seekable(poisoned_train_loader)
    poisoned_train_iter_lis.append(poisoned_train_iter)

#sub_poisoned_train_iter_lis=[]
#for j in range(10):
    #idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
    #poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
    #poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, (j+1)*0.1, pattern_type,  idxs, poison_all=False)
    #poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
    #poisoned_train_iter = mit.seekable(poisoned_train_loader)
    #sub_poisoned_train_iter_lis.append(poisoned_train_iter)

#print('size of sub_poisoned_lis:',len(sub_poisoned_train_iter_lis))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_iter = mit.seekable(trainloader)


val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Load model and data
net = ResNet18().to(**setup)
#net = vgg11().to(**setup)
#net2 = CleanNet().to(**setup)

# poison the validation dataset and trainning dataset
poison_frac=0.5
idxs = (val_dataset.targets == base_class).nonzero().flatten().tolist()
poisoned_val_set = DatasetSplit(copy.deepcopy(val_dataset), idxs)
poison_dataset(poisoned_val_set.dataset, data, base_class, target_class, 1, pattern_type,  idxs, poison_all=True)
poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=batch_size, shuffle=False)


idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=False)
poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
globle_poisoned_train_iter = mit.seekable(poisoned_train_loader)


random.seed(1001) #101, 150, 501
att_ids=random.sample(range(num_clients),num_attacker)
print('attacker ids: ', att_ids)



old_weights = get_parameters(net)


for rnd in range(fl_epoch):
    
    print('---------------------------------------------------')
    print('rnd: ',rnd+1)
    random.seed(rnd)
    cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    print('chosen clients: ', cids)
    print('selected attackers: ',common(cids, att_ids))
    
    weights_lis=[]
    for cid in exclude(cids,att_ids):  #if there is an attack
    #for cid in cids:  #NA env
        set_parameters(net, old_weights)
        train(net, train_iter, epochs=1, lr=lr)
        new_weight=get_parameters(net)
        weights_lis.append(new_weight)


    grads=[weights_to_vector(weight)-weights_to_vector(old_weights) for weight in weights_lis]
    max_weight_diff=max(grads, key=tuple)
    min_weight_diff=min(grads, key=tuple)
    #avg_weight_diff=np.average(grads, axis=0)
    
    #backdoor attack 
    for cid in common(cids, att_ids):
        set_parameters(net, old_weights)     
        train(net, globle_poisoned_train_iter, epochs=10, lr=lr)
        temp_new_weight=get_parameters(net)
        new_weight=Neurotoxin(old_weights, temp_new_weight, grads, max_weight_diff, min_weight_diff, 1000)
        #new_weight=Neurocraft(old_weights, temp_new_weight, grads, max_weight_diff, min_weight_diff, avg_weight_diff, 1000)
        #new_weight=temp_new_weight      
        weights_lis.append(new_weight)
     
    
    #aggregate_weights = average(weights_lis)
    #aggregate_weights = Median(old_weights, weights_lis)
    #aggregate_weights = GeoMedian(old_weights, weights_lis, 10)
    #aggregate_weights = Clipped_Median(old_weights, weights_lis, max_norm)
    aggregate_weights = Krum(old_weights, weights_lis, 2)
    #aggregate_weights = Clipping(old_weights, weights_lis, 0.01)
    
    
    #aggregate_weights=random_noise(aggregate_weights, 200)
    old_weights=aggregate_weights
    set_parameters(net, old_weights)
    
    loss, acc = test(net, val_loader)
    poi_loss, poi_acc= test(net, poisoned_val_loader)
    
    print('global_acc: ', acc)
    print('backdoor_acc: ', poi_acc)

    #set_parameters(net2, old_weights)
    #_, clean_acc = test(net2, val_loader)
    #_, clean_poi_acc= test(net2, poisoned_val_loader)
    #print('clean_global_acc after1: ', clean_acc)
    #print('clean_backdoor_acc after1: ', clean_poi_acc)
    
    f=open('Neurotoxin_main_frac0.5_10local_krum_lr0.05_1000epoch_cifar10_resnet18_100c_10a.txt','a')
    f.write(str(acc)+'\n')
    f.close()

    f=open('Neurotoxin_backdoor_frac0.5_10local_krum_lr0.05_1000epoch_cifar10_resnet18_100c_10a.txt','a')
    f.write(str(poi_acc)+'\n')
    f.close()

    #f=open('clean_main_frac0.5_avg_lr0.05_1000epoch_cifar10_resnet18_100c_10a.txt','a')
    #f.write(str(clean_acc)+'\n')
    #f.close()

    #f=open('clean_backdoor_frac0.5_avg_lr0.05_1000epoch_cifar10_resnet18_100c_10a.txt','a')
    #f.write(str(clean_poi_acc)+'\n')
    #f.close()
    
