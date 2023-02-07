import torch
import numpy as np
from collections import OrderedDict

__all__ = ['weights_to_vector','vector_to_weights','common','exclude','get_parameters','set_parameters','train','test','craft','check_attack']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def relu(x): return max(0.0, x)

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity.""" 

    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)

def weights_to_vector(weights):
    """Convert NumPy weights to 1-D Numpy array."""
    Lis=[np.ndarray.flatten(ndarray) for ndarray in weights]
    return np.concatenate(Lis, axis=0)

def vector_to_weights(vector,weights):
    """Convert 1-D Numpy array tp NumPy weights."""
    indies = np.cumsum([0]+[layer.size for layer in weights]) #indies for each layer of a weight
    Lis=[vector[indies[i]:indies[i+1]].reshape(weights[i].shape) for i in range(len(weights))]
    return Lis

def common(a,b): 
    c = [value for value in a if value in b] 
    return c

def exclude(a,b):
    c = [value for value in a if value not in b]
    return c

def get_parameters(net):
    #for _, val in net.state_dict().items():
        #if np.isnan(val.cpu().numpy()).any(): print(val)
    result = []
    for _, val in net.state_dict().items():
        #print((len(val.cpu().numpy().shape)))
        if len(val.cpu().numpy().shape)!=0:
            result.append(val.cpu().numpy())
        else:
            result.append(np.asarray([val.cpu().numpy()]))
    #return [val.cpu().numpy() for _, val in net.state_dict().items()]
    return result

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, train_iter, epochs, lr):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    for _ in range(epochs):
        #for images, labels in trainloader:
        try:
            images, labels = next(train_iter)
        except:        
            train_iter.seek(0)
            images, labels = next(train_iter)
        #print(labels)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #images, labels = images, labels
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

def test(net, valloader):
    """Validate the network on the 10% training set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
        #data=next(iter(valloader))
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return loss, accuracy


def craft(old_weights, new_weights, action, b):
    
    #zeta_max, zeta_min = b*0.0030664504, b*-0.0024578273
    #zeta_max=[zeta_layer*b for zeta_layer in zeta_max]
    #zeta_min=[zeta_layer*b for zeta_layer in zeta_min]
    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    #crafted_weight_diff = [diff_layer* (action*(zeta_max-zeta_min)/abs(zeta_max)*0.5+(zeta_max+zeta_min)/abs(zeta_max)*0.5) for diff_layer in weight_diff]
    #crafted_weight_diff = [diff_layer* (action*(max_layer-min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5
                                        #+(max_layer+min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5) 
                                        #for diff_layer, max_layer, min_layer in zip(weight_diff, zeta_max, zeta_min)]
    
    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight


def check_attack(cids,att_ids):
    return  np.array([(id in att_ids) for id in cids]).any()
