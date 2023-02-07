import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import statistics

from Networks import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup = dict(device=DEVICE, dtype=torch.float)


__all__ = ['CleanNet', 'Net_prune', 'prune_train', 'get_prune_rank', 'get_prune_mask', 'Net_post','model_weights_change']




class CleanNet(nn.Module):

    def __init__(self):
        super(CleanNet, self).__init__()
        model = ResNet18()
        model = model.to(**setup)
        #model.load_state_dict(torch.load('./'+args.model_dir+'/model.pth'))
        model.eval()
        self.model = model
        self.clamp_w1 = torch.ones([64, 1, 1]) + 7.0
        self.clamp_w2 = torch.ones([64, 1, 1]) + 7.0
        self.clamp_w3 = torch.ones([128, 1, 1]) + 7.0
        self.clamp_w1.requires_grad = True
        self.clamp_w2.requires_grad = True
        self.clamp_w3.requires_grad = True
    def forward(self, x):

        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = torch.min(out, self.clamp_w1)
        out = self.model.layer1(out)
        out = torch.min(out, self.clamp_w2)
        out = self.model.layer2(out)
        out = torch.min(out, self.clamp_w3)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)
        return out

class Net_prune(nn.Module):
    def __init__(self, model):
        super(Net_prune, self).__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        return x

        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x)

def prune_train(net_prune, train_iter):
    try:
        images, labels = next(train_iter)
    except:
        train_iter.seek(0)
        images, labels = next(train_iter)
    #print(labels)
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output = net_prune(images)
    prune_output = torch.sum(output, dim=0)
    prune_output = prune_output.div(len(labels))
    return prune_output


def get_prune_rank(prune_output):
    prune_rank = torch.ones(prune_output.size())
    neuron_sum = torch.sum(prune_output, (1, 2))
    sorted_value, indices = torch.sort(neuron_sum)

    rank = 0
    for i in indices:
        for j in range(len(prune_rank.view(prune_rank.size()[0], -1)[i])):
            prune_rank.view(prune_rank.size()[0], -1)[i][j] = rank
        rank += 1
    assert(rank == len(indices))
    
    return prune_rank


def get_prune_mask(prune_rank, prune_num):
    prune_mask = torch.ones(prune_rank.size())
    neuron_sum = torch.sum(prune_rank, (1, 2))
    sorted_value, indices = torch.sort(neuron_sum)

    for i in indices[:prune_num]:
        for j in range(len(prune_mask.view(prune_mask.size()[0], -1)[i])):
            prune_mask.view(prune_mask.size()[0], -1)[i][j] = 0
    
    return prune_mask



class Net_post(nn.Module):
    def __init__(self, model, prune_mask):
        super(Net_post, self).__init__()
        self.conv1 = copy.deepcopy(model.conv1)
        self.conv2 = copy.deepcopy(model.conv2)
        self.conv2_drop = copy.deepcopy(model.conv2_drop)
        self.fc1 = copy.deepcopy(model.fc1)
        self.fc2 = copy.deepcopy(model.fc2)
        #self.prune_mask = copy.deepcopy(prune_mask).cuda()
        self.prune_mask = copy.deepcopy(prune_mask).to(DEVICE)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        
        x = x.mul(self.prune_mask)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def model_weights_change(model, std_bound=1):
    with torch.no_grad():
        layer = model.conv2.weight
        mean = statistics.mean(layer.view(-1).cpu().tolist())
        stdev = statistics.stdev(layer.view(-1).cpu().tolist())
        upper = torch.tensor(mean + std_bound * stdev)
        lower = torch.tensor(mean - std_bound * stdev)
        print("Mean: %f; Standard Deviation: %f" % (mean, stdev))
        print("Upper Bound: %f; Lower Bound: %f" % (upper, lower))
        
        count = 0
        for i in range(len(layer.view(-1))):
            x = layer.view(-1)[i]
            if x > upper:
                layer.view(-1)[i] = torch.tensor(0)
                continue
            if x < lower:
                layer.view(-1)[i] = torch.tensor(0)
                continue
            count += 1
            
        # print(count, len(layer.view(-1)))
        # print("Pruned Num: %d" % (len(layer.view(-1)) - count))

#net_prune =Net_prune(net).to(**setup)
#prune_output = prune_train(net_prune, train_iter)
#print(prune_output.size())

#prune_rank = get_prune_rank(prune_output)
# for i in range(len(prune_rank)):
#     print(prune_rank[i][0][0])

#prune_mask = get_prune_mask(prune_rank, 5)
# for i in range(len(prune_mask)):
#     print(prune_mask[i][0][0])