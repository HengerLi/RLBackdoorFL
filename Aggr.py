
from Util import *
from functools import reduce
import numpy as np
import copy


__all__=['average','Krum','Median','GeoMedian','Clipped_Median','Clipping','random_noise']

def average(new_weights, num_clients, subsample_rate):
        fractions=[1/int(num_clients*subsample_rate) for _ in range(int(num_clients*subsample_rate))]
        fraction_total=np.sum(fractions)
        
        # Create a list of weights, each multiplied by the related fraction
        weighted_weights = [
            [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
        ]

        # Compute average weights of each layer
        aggregate_weights = [
            reduce(np.add, layer_updates) / fraction_total
            for layer_updates in zip(*weighted_weights)
        ]

        return aggregate_weights

def Krum(old_weight, new_weights, num_attacker):
    """Compute Krum average."""
    
    num_attacker=num_attacker
    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    scrs=[]
    for i in grads:
        scr=[]
        for j in grads:
            dif=weights_to_vector(i)-weights_to_vector(j)
            sco=np.linalg.norm(dif)
            scr.append(sco)
        top_k = sorted(scr)[1:len(grads)-2-num_attacker]
        scrs.append(sum(top_k))
    chosen_grads= grads[scrs.index(min(scrs))]
    krum_weights = [w1-w2 for w1,w2 in zip(old_weight, chosen_grads)]
    return krum_weights


def Median(old_weight, new_weights):
    """Compute Median average."""

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)
    
    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))
    Median_weights = [w1-w2 for w1,w2 in zip(old_weight, med_grad)]
    return Median_weights

def GeoMedian(old_weight, new_weights, R):
    """Compute GeoMedian"""
    #R=10
    epsilon=1e-6
    geo_median=np.zeros(len(weights_to_vector(old_weight)))
    vec_weight_lis=[]
    for new_weight in new_weights:
        vec_weight_lis.append(weights_to_vector(new_weight))
    for _ in range(R):
        beta_lis=[]
        for vec_weight in vec_weight_lis:
            beta=1.0/max(np.linalg.norm(vec_weight-geo_median),epsilon)
            beta_lis.append(beta)
        geo_median=np.sum([beta*vec_weight/sum(beta_lis) for beta,vec_weight in zip(beta_lis,vec_weight_lis)], axis=0)
    
    return vector_to_weights(geo_median, old_weight)

def Clipped_Median(old_weights, new_weights, max_norm):
    """Compute Median average."""

    grads=[]
    #for new_weight in new_weights:
        #grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        #grads.append(grad)

    #max_norm=2 #0.5, 1, 2, 5
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)
    
    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))
    Median_weights = [w1-w2 for w1,w2 in zip(old_weights, med_grad)]
    return Median_weights


def Clipping(old_weights, new_weights, max_norm, num_clients, subsample_rate):
    #max_norm=2 #0.5, 1, 2, 5
    grads=[]
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)
    

    fractions=[1/int(num_clients*subsample_rate) for _ in range(int(num_clients*subsample_rate))]
    fraction_total=np.sum(fractions)
        
    # Create a list of weights, each multiplied by the related fraction
    weighted_grads = [
        [layer * fraction for layer in grad] for grad, fraction in zip(grads, fractions)
    ]

    # Compute average weights of each layer
    aggregate_grad = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_grads)
    ]

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, aggregate_grad)]


    return Centered_weights

def random_noise(weight, gau_rate):
    #gau_rate = 10000
    #m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1/gau_rate]))
    noisy_weight = copy.deepcopy(weight)
    noisy_weight_vec = weights_to_vector(noisy_weight)
    noisy_weight_vec = noisy_weight_vec + np.random.laplace(0,1/gau_rate,noisy_weight_vec.shape)
    noisy_weight = vector_to_weights(noisy_weight_vec, weight)
    return noisy_weight
