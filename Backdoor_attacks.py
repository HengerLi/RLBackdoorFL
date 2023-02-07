import numpy as np

from Util import *


def Neurotoxin(old_weights, temp_new_weight, grads, max_weight_diff, min_weight_diff, top_number):
    #top_number=100 #10, 100, 1000, 10000
    
    vec_grad=weights_to_vector(temp_new_weight)-weights_to_vector(old_weights)
    vec_grad_abs=max([np.abs(grad-vec_grad) for grad in grads], key=tuple)
    argmax_ids=np.argsort(vec_grad_abs)[-1-top_number:-1]

    for id in argmax_ids:
        if vec_grad[id]>max_weight_diff[id]: vec_grad[id]=max_weight_diff[id]
        if vec_grad[id]<min_weight_diff[id]: vec_grad[id]=min_weight_diff[id]

    vec_crafted_weight=weights_to_vector(old_weights)+vec_grad
    crafted_weight=vector_to_weights(vec_crafted_weight, old_weights)

    return crafted_weight


def Neurocraft(old_weights, temp_new_weight, grads, max_weight_diff, min_weight_diff, avg_weight_diff, top_number):
    #top_number=100 #10, 100, 1000, 10000
    
    vec_grad=weights_to_vector(temp_new_weight)-weights_to_vector(old_weights)
    vec_grad_abs=max([np.abs(grad-vec_grad) for grad in grads], key=tuple)
    argmax_ids=np.argsort(vec_grad_abs)[-1-top_number:-1]
    #argmin_ids=np.argsort(vec_grad_abs)[:top_number]

    for id in argmax_ids:
        if vec_grad[id]>max_weight_diff[id]*2: vec_grad[id]=max_weight_diff[id]*2
        if vec_grad[id]<min_weight_diff[id]*2: vec_grad[id]=min_weight_diff[id]*2
        if vec_grad[id]<max_weight_diff[id] and vec_grad[id]>min_weight_diff[id]: vec_grad[id]=avg_weight_diff[id]
    
    #for id in argmin_ids:
        #if vec_grad[id]>max_weight_diff[id]: vec_grad[id]=avg_weight_diff[id]
        #if vec_grad[id]<min_weight_diff[id]: vec_grad[id]=avg_weight_diff[id]
        #vec_grad[id]=avg_weight_diff[id]
    
    #for id in range(len(vec_grad)):
        #if id not in argmax_ids: vec_grad[id]=avg_weight_diff[id]
        #else:
            #if vec_grad[id]>max_weight_diff[id]*2: vec_grad[id]=max_weight_diff[id]*2
            #if vec_grad[id]<min_weight_diff[id]*2: vec_grad[id]=min_weight_diff[id]*2


    vec_crafted_weight=weights_to_vector(old_weights)+vec_grad
    crafted_weight=vector_to_weights(vec_crafted_weight, old_weights)

    return crafted_weight