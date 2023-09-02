"""
@authors: Alireza Heshmati and Dr. Sajjad Amini
"""

import torch
from utils import objective_f

# use cuda if it is available
if torch.cuda.device_count() != 0:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')

# proximal Sparsites (SCAD, HT and ST) according to the paper
def prox_rs(delta, sparsity_type, param, theta = 3.7):
    if sparsity_type == 'SCAD':
        alpha = torch.zeros_like(delta)
        
        index1 = abs(delta) <= 2* param
        index2 = (abs(delta) > 2* param) * (abs(delta) <= theta* param)
        index3 = abs(delta) > theta* param
        
        alpha[index1] = torch.sign(delta[index1]) * torch.maximum(abs(delta[index1])- param, torch.tensor(0).to(device))
        alpha[index2] = ((theta-1)*delta[index2]-torch.sign(delta[index2])*theta*param)/((theta-2))
        alpha[index3] = delta[index3]
        
    elif sparsity_type == 'HT':
        alpha = torch.nn.Hardshrink(lambd=param)(delta) # see pytorch
        
    elif sparsity_type == 'ST':
        alpha = torch.nn.Softshrink(lambd=param)(delta)
        
    return alpha

# proximal infinity norm according to the paper
def prox_rinf(beta, zita_inf):
    ind_inf = abs(beta) > zita_inf
    beta[ind_inf] = torch.sign(beta[ind_inf]) * zita_inf
    return beta

# E loss according to the paper
def E(model_outputs, one_hot_labels, delta, alpha, beta, targets, mu, lambda2, o_f_type = 'CE'):
    D = objective_f(model_outputs, one_hot_labels,targets, o_f_type)
    norm2_2 = lambda2 * torch.sum(torch.sum((delta)**2, dim = (-3,-2,-1)))
    MSE_total = torch.sum(torch.sum((delta-alpha)**2, dim = (-3,-2,-1)) + torch.sum((delta-beta)**2, dim = (-3,-2,-1)))/(2*mu)
    return D + norm2_2 + MSE_total

# inner loop according to the paper
def inner_loop(model, images, one_hot_labels, delta, targets, lr , mu, N2, lambda2,
               zita_inf, sparsity_type, param, theta, o_f_type = 'CE'):
    for _ in range(N2) :
        alpha = prox_rs(delta, sparsity_type, param, theta)
        beta = prox_rinf(delta, zita_inf)
        delta.requires_grad_(True)
        
        outputs = model(images+delta)
        E_loss = E(outputs, one_hot_labels, delta, alpha, beta, targets, mu , lambda2, o_f_type)
        
        E_loss.backward()
        with torch.no_grad():
            delta = delta - lr*delta.grad
            delta = torch.clamp( images+delta , min = 0 , max = 1) - images
    
    return delta


def gradual_sparse_attack(model, images, labels ,one_hot_labels, targets, args):
    mu = args.mu_max
    lr = args.lr_max
    delta = torch.zeros_like(images)

    # outer loop according to the paper
    for _ in range(args.N1) :
        
        delta = inner_loop(model, images, one_hot_labels, delta, targets,lr, mu, args.N2, args.lambda2, args.zita_inf, args.sparsity_type, mu * args.lambda1 , args.theta, args.o_f_type)
        if mu >  args.mu_min :
            mu =  max(mu * args.s , args.mu_min)
            
        if lr >  args.lr_min :
            lr = max(lr * args.lr_s , args.lr_min)
            
    delta = torch.round((delta)*255)/255
    
    if targets == None:
        adv_acc = (labels != torch.argmax(model(images+delta), -1)).float()
        targets_ = None
    else :
        adv_acc = (targets == torch.argmax(model(images+delta), -1)).float()
        targets_ = targets[adv_acc > 0]
        
    return delta, adv_acc.cpu().detach()

