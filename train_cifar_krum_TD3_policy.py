import torch
import random
import more_itertools as mit
import numpy as np

from Aggr import *
from DataProcess import *
from Networks import *
from Util import *

import copy
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset


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
max_norm=2

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

#poisoned_train_iter_lis=[]
#for i in range(4):
    #idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
    #poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
    #poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=False, agent_idx=i)
    #poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
    #poisoned_train_iter = mit.seekable(poisoned_train_loader)
    #poisoned_train_iter_lis.append(poisoned_train_iter)



trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_iter = mit.seekable(trainloader)


val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# poison the trainning dataset with different fraction rates

idxs = (val_dataset.targets == base_class).nonzero().flatten().tolist()
poisoned_val_set = DatasetSplit(copy.deepcopy(val_dataset), idxs)
poison_dataset(poisoned_val_set.dataset, data, base_class, target_class, 1, pattern_type,  idxs, poison_all=True)
poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=batch_size, shuffle=False)

global_poisoned_train_iter_lis=[]
for i in range(11):
    poison_frac=i*0.1
    idxs = (train_dataset.targets == base_class).nonzero().flatten().tolist()
    poisoned_train_set = DatasetSplit(copy.deepcopy(train_dataset), idxs)
    poison_dataset(poisoned_train_set.dataset, data, base_class, target_class, poison_frac, pattern_type,  idxs, poison_all=False)
    poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)
    globle_poisoned_train_iter = mit.seekable(poisoned_train_loader)
    global_poisoned_train_iter_lis.append(globle_poisoned_train_iter)


random.seed(1001) #101, 150, 501
att_ids=random.sample(range(num_clients),num_attacker)
print('attacker ids: ', att_ids)

#RL backdoor action: frac, lr, epoch, scalar
import gym
from gym import spaces
from gym.utils import seeding
import time


class Backdoor_Env(gym.Env):

    def __init__(self):
        
        self.rnd=0
        self.weights_dimension=1290 #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(4,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        #self.observation_space = spaces.Box(
            #low=low,
            #high=high,
            #shape=(self.weights_dimension,),
            #dtype=np.float32
        #)
        #self.observation_space = spaces.Dict(
            #loss=spaces.Box(0, np.inf, shape=(1, )),
            #num_selected_attacker= spaces.Discrete(min(num_attacker,int(num_clients*subsample_rate)))
        #)
        #self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (1290,), dtype = np.float32),
                                             #num_attacker = spaces.Discrete(11))

        #self.observation_space =spaces.Discrete(11)
        self.observation_space= spaces.MultiDiscrete([6, 11])
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        
        self.rnd+=1
        
        
        action[0] = int(action[0]*5.5+5.5) #index of poison frac [0:1:10]
        action[1] = action[1]*0.05+0.05 #lr for local trianning [0, 0.1]
        action[2] = int(action[2]*5+6)  #local step [1:1:10]
        action[3] = action[3]*5.0+5.0 #scalar [0, 10]
        #print(action)

        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=lr)
            new_weight=get_parameters(self.net)     
            new_weights.append(new_weight)
       
        att_weights_lis=[]
        #backdoor attack
        for cid in common(self.cids,att_ids):
            set_parameters(self.net, self.aggregate_weights)
            train(self.net, self.poisoned_train_iter_lis[int(action[0])], epochs=int(action[2]), lr=action[1])
            new_weight=get_parameters(self.net)
            loss, _= test(self.net, val_loader)
            if np.isnan(loss): new_weight=self.aggregate_weights
            poi_loss, _= test(self.net, poisoned_val_loader)
            if np.isnan(poi_loss): new_weight=self.aggregate_weights
            att_weights_lis.append(new_weight)


        for cid in common(self.cids,att_ids):
            new_weight=craft(self.aggregate_weights, average(att_weights_lis, num_clients, subsample_rate), -1, action[3])
            new_weights.append(new_weight)
        
        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Clipped_Median(self.aggregate_weights, new_weights, max_norm)
        self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights, max_norm)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))


        set_parameters(self.net,self.aggregate_weights)
        #new_loss, _ = test(self.net, simu_testloader)
        new_loss, new_acc = test(self.net, val_loader)
        poi_loss, poi_acc= test(self.net, poisoned_val_loader)
        #new_loss, _ = test(self.net, testloader)
        # Caculate the reward by l(s^t(tau+1))-l(s^t(tau))
        #reward=new_loss-self.loss

        # Caculate the backdoor reward
        reward=poi_acc-self.poi_acc

        #reward= self.acc-new_acc
        #reward=-new_acc
        #if math.isinf(reward) or math.isnan(reward):
            #print("abnormal reward")
            #reward = 0

        #self.loss=new_loss
        #self.acc=new_acc
        self.poi_acc=poi_acc

        #self.state = weights_to_vector(self.aggregate_weights)

        #last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        #state_min = np.min(last_layer)
        #state_max = np.max(last_layer)
        #norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        #self.state = np.array(norm_state).reshape(1,self.weights_dimension)
        #print(self.state)
        
        done=False
        if self.rnd>=500: done= True #15, 25, 75
        #return {"loss": self.loss, "num_selected_attacker": len(common(self.cids, att_ids))-1}, reward, done, {}
        #return {"pram": self.state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}
        return [min(5,int(self.rnd/100)),len(common(self.cids, att_ids))], reward, done, {}

    def reset(self):

        self.rnd=0
        # Load model
        #self.net = Net().to(**setup)
        #self.net.load_state_dict(torch.load('small_net_init'))
        #self.net = MNISTClassifier().to(**setup)
        #self.net.load_state_dict(torch.load('MNISTClassifier_init'))
        self.net = ResNet18().to(**setup)

        self.aggregate_weights = get_parameters(self.net)
        #self.train_iter = mit.seekable(trainloader)
        self.train_iter=train_iter
        #self.train_iters=train_iters
        self.poisoned_train_iter_lis=global_poisoned_train_iter_lis


        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        
        
        # Initial weight start from random step
        #starts_step=random.randint(0,1000) #20, 100
        #is_attack=check_attack(self.cids, att_ids)
        #starts_step=0
        #is_attack=True
        #while is_attack==False or starts_step>0:
            
            #starts_step-=1
            #self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #is_attack=check_attack(self.cids, att_ids)
            #new_weights=[]
            #for i in range(int(num_clients*subsample_rate)):
            #for cid in self.cids:
                #set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                #train(self.net, self.train_iter, epochs=1, lr=lr)
                #new_weights.append(get_parameters(self.net))

            # Compute average weights
            #aggregate_weights = average(weights_lis)
            #aggregate_weights = Median(old_weights, weights_lis)
            #aggregate_weights = GeoMedian(old_weights, weights_lis, 10)
            #aggregate_weights = Clipped_Median(old_weights, weights_lis, max_norm)
            #aggregate_weights = Krum(old_weights, weights_lis, 2)
            #aggregate_weights=Clipping(old_weights, weights_lis, 0.01)

        
        #self.state = weights_to_vector(self.aggregate_weights) 
        
        #last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        #state_min = np.min(last_layer)
        #state_max = np.max(last_layer)
        #norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        #self.state = np.array(norm_state).reshape(1,self.weights_dimension)
        #print(self.state)
        
        set_parameters(self.net,self.aggregate_weights)
        #self.loss, _ = test(self.net, simu_testloader)
        self.loss, self.acc = test(self.net, val_loader)
        self.poi_loss, self.poi_acc= test(self.net, poisoned_val_loader)

        #self.loss, _ = test(self.net, testloader)
        
        
        #return {"loss": self.loss, "num_selected_attacker": len(common(self.cids, att_ids))-1}
        #return {"pram": self.state, "num_attacker": len(common(self.cids, att_ids))}
        return [min(5,int(self.rnd/100)),len(common(self.cids, att_ids))]



env = Backdoor_Env()

#print(env.observation_space)
print(env.action_space)

#num_states = env.observation_space.shape[0]
#print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))
#print(env.action_space.shape[-1])

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


#torch.autograd.set_detect_anomaly(True)

start = time.time()
env.reset()
#mid = time.time()
done=False
rnd=0
#while done==False:
while rnd<=10:
  rnd+=1
  print('rnd: ',rnd)
  action=np.random.uniform(low=-1.0, high=1.0, size=4)  #510,1290,21840
  state, reward, done, _ = env.step(action)
  print(state)
  #print(reward)  
end = time.time()
print(end-start)
#print(mid-start)
#print(end - mid)
#print((end-mid)/20)

#torch.autograd.set_detect_anomaly(False)

#!pip install stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = Backdoor_Env()

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, optimize_memory_usage=False, buffer_size=100000, gamma=1, 
             learning_rate=1e-4, tau=1, learning_starts=0, train_freq=(1000, 'step'), batch_size=256)

model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False) 
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_5000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_10000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_15000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_20000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False) 
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_25000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_30000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_35000")
model.learn(total_timesteps=5000, log_interval=1, reset_num_timesteps=False)
model.save("plr1e-4_mnist_epoch500_krum_100c_10a_TD3_40000")

del model # remove to demonstrate saving and loading