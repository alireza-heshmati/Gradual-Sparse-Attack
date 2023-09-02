"""
@author: Alireza Heshmati
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

if torch.cuda.device_count() != 0:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')



def make_path_or_model_name(model_type):
    if model_type == 'convnet':
        p_or_mn = '/supplies/cifar_best.pth'
    elif model_type == 'mobilenetv2' :
        p_or_mn = '/supplies/mobilenetv2_cifar10_weights.pkl'
    elif model_type == 'robustnet':
        p_or_mn = 'Gowal2021Improving_R18_ddpm_100m'
    else :
        p_or_mn = 'No Need'
    return p_or_mn

# Creating attack target according input dataset 
def creat_targets(main_outputs, one_hot_labels, target_type):
    if target_type == 'Best':
        targets = torch.argmax((1-0.9*one_hot_labels)*(main_outputs-1e5*one_hot_labels), dim=1)
    
    elif target_type == 'Average':
        torch.manual_seed(0)
        original_labels = torch.argmax(one_hot_labels, dim=1)
        targets = torch.randint(0, len(one_hot_labels[0]), (len(original_labels),) ,device=device)
        while any(targets == original_labels):
            targets[targets == original_labels] = torch.randint(0, len(one_hot_labels[0]),(len(targets[targets == original_labels]),) ,device=device)
    
    elif target_type == 'Worst':
        targets = torch.argmin((1-0.9*one_hot_labels)*(main_outputs+1e5*one_hot_labels), dim=1)
        
    else :
        targets = None
    
    return targets

# CrossEntropyLoss
def f_cross(outputs_ ,targets_):
    return nn.CrossEntropyLoss(reduction = 'sum')(outputs_,targets_)

# CW loss 
def f_CW(comparable_val , targets_val, kap=0):
    return torch.sum(torch.clamp(comparable_val-targets_val, min=-kap))

# creating objective function
def objective_f(model_outputs, one_hot_labels, targets_, o_f_type = 'CW', kap=0): 
    if o_f_type == 'CE':
        
        if targets_ == None:
            targets_ = torch.argmax((1-0.9*one_hot_labels)*(model_outputs-1e5*one_hot_labels), dim=1)
            
        o_f = f_cross(model_outputs ,targets_)
        
    elif o_f_type == 'CW':
        if targets_ == None:
            targets_val, targets_ = torch.max((1-0.9*one_hot_labels)*(model_outputs-1e5*one_hot_labels), dim=1)
            comparable_val = torch.amax((1-0.9*torch.eye(len(model_outputs[0]))[targets_].to(device))*(
                model_outputs-1e5*torch.eye(len(model_outputs[0]))[targets_].to(device)), dim=1)
            
        else :
            targets_val = torch.amax(torch.eye(len(model_outputs[0]))[targets_].to(device) * model_outputs, dim=-1)
            #comparable_val = torch.amax((one_hot_labels)*model_outputs, dim=-1)
            comparable_val = torch.amax((1-0.9*torch.eye(len(model_outputs[0]))[targets_].to(device))*(
                model_outputs-1e5*torch.eye(len(model_outputs[0]))[targets_].to(device)), dim=1)
            
        o_f = f_CW(comparable_val , targets_val, kap)
        
    return o_f


# assessing validation of model
def validation(test_X,test_y ,net,batch = 16):
    test_X,test_y = test_X.to(device)      , test_y.to(device)
    net.eval()
    correct = 0
    total = 0
    for i in range(0, len(test_X), batch): 
        batch_X = test_X[i:i+batch]
        batch_y = test_y[i:i+batch]  
        predicted_class = torch.argmax(net(batch_X), dim=1)
        a = predicted_class == batch_y
        total += len(batch_y)
        correct += len(batch_y[a])    
    return round(correct/total, 4)      

# defining position of perturbed pixels
def perturbated_pixels(deltas_pixels):
    if len(deltas_pixels.shape) == 3 :
        deltas_pixels = deltas_pixels[None,:]
    
    deltas_pixels = torch.sum(abs(deltas_pixels),dim =1).cpu().detach().numpy()
    
    ind_perturbations = deltas_pixels > 0
    deltas_pixels[ind_perturbations] = 1
    deltas_pixels[~ind_perturbations] = 0                    
    return deltas_pixels

# defining number of perturbed pixels
def perturbations_number(deltas_pixels):
    if len(deltas_pixels.shape) == 3 :
        deltas_pixels = deltas_pixels[None,:]
        
    deltas_pixels = torch.sum(abs(deltas_pixels),dim =1)
    deltas_pixels = (deltas_pixels > 0).float()
    num_pert_ = torch.sum(deltas_pixels,dim = (-2,-1))
    return torch.mean(num_pert_).numpy(), torch.std(num_pert_, unbiased=False).numpy() , torch.max(num_pert_).numpy()

# defining number of perturbed elements of images
def l0_norm(deltas):
    if len(deltas.shape) == 3 :
        deltas = deltas[None,:]
    deltas = (torch.abs(deltas) > 0).float()
    l0_ = torch.sum(deltas,dim = (-3,-2,-1))
    return torch.mean(l0_).numpy(), torch.std(l0_, unbiased=False).numpy() , torch.max(l0_).numpy()


def l1_norm(deltas): 
    if len(deltas.shape) == 3 :
        deltas = deltas[None,:]
    l1_ = torch.sum(torch.abs(deltas),dim = (-3,-2,-1))
    return torch.mean(l1_).numpy(), torch.std(l1_, unbiased=False).numpy() , torch.max(l1_).numpy()

def l2_norm(deltas):
    if len(deltas.shape) == 3 :
        deltas = deltas[None,:]
    l2_ = torch.sum(deltas**2,dim = (-3,-2,-1))**0.5
    return torch.mean(l2_).numpy(), torch.std(l2_, unbiased=False).numpy() , torch.max(l2_).numpy()

# calculating infinity norm
def linf_norm(deltas):
    if len(deltas.shape) == 3 :
        deltas = deltas[None,:]
    linf_ = torch.amax(abs(deltas) ,axis=(1,2,3))
    return torch.mean(linf_).numpy(), torch.std(linf_, unbiased=False).numpy() , torch.max(linf_).numpy()

# calculating number of perturbed pixels
def attack_accuracy(x_adversarial,y,net,batch = None):
    if batch == None :
        x_adversarial = x_adversarial.to(device)
        y = y.to(device)
        att = torch.argmax(net(x_adversarial), 1)
        ind = (att != y)
        len_att = len(att[ind])
    else :
        len_att = 0
        for i in range(0, len(y), batch): 
            batch_X = x_adversarial[i:i+batch]
            batch_y = y[i:i+batch]  
            att = torch.argmax(net(batch_X), dim=1)
            ind = (att != batch_y)
            len_att += len(att[ind]) 
    return len_att/len(y)


# for plotting
def plot_results(or_im, adv_im, correct_labels, attack_labels,imagenet_or_cifar10 = 'cifar10'):
    or_im = or_im.cpu().detach()
    adv_im = adv_im.cpu().detach()
    
    if len(or_im.shape) == 3 :
        or_im = or_im[None,:]
        adv_im = adv_im[None,:]
        
    per_show =  perturbated_pixels(or_im-adv_im)
    
    correct_labels = correct_labels.cpu().detach().numpy()
    attack_labels = attack_labels.cpu().detach().numpy()
    if imagenet_or_cifar10 == 'cifar10':
        dict = {0 : "airplane",1:"automobile", 2 : "bird",3:"cat", 4 : "deer",5:"dog",
                6 : "frog",7:"horse", 8 : "ship", 9:"truck" }
        plt.figure()
        fig, axs = plt.subplots(len(adv_im)//2,6,figsize = (20,len(adv_im)*2))
        for i in range((len(adv_im)//2)*2):
            axs[i//2,3*(i%2)].set_title('main: '+str(dict[int(correct_labels[i])]))
            axs[i//2,3*(i%2)].imshow(or_im[i].permute((1,2,0)).cpu().detach().numpy())
            axs[i//2,3*(i%2)].axis('off')
            axs[i//2,3*(i%2)+1].set_title('%s pixel(s)' % repr(int(perturbations_number(or_im[i]-adv_im[i])[0])))
            axs[i//2,3*(i%2)+1].imshow(per_show[i],cmap = 'gray')
            axs[i//2,3*(i%2)+1].axis('off')
            axs[i//2,3*(i%2)+2].set_title('attack: '+str(dict[int(attack_labels[i])]))
            axs[i//2,3*(i%2)+2].imshow(adv_im[i].permute((1,2,0)).cpu().detach().numpy())
            axs[i//2,3*(i%2)+2].axis('off')
            
    if imagenet_or_cifar10 == 'imagenet':
        with open("/home/dllabsharif/Alireza_Heshmati/datasets_and_pretrained_weights_extra/pytorch_hub_master_imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]       
            
        plt.figure()
        fig, axs = plt.subplots(len(adv_im),3,figsize = (15, len(adv_im)*8))
        
        for i in range(len(adv_im)):
            axs[i,0].set_title('main: '+ categories[correct_labels[i]])
            axs[i,0].imshow(or_im[i].permute((1,2,0)).cpu().detach().numpy())
            axs[i,0].axis('off')
            axs[i,1].set_title('%s pixel(s)' % repr(int(perturbations_number(or_im[i]-adv_im[i])[0])))
            axs[i,1].imshow(per_show[i],cmap = 'gray')
            axs[i,1].axis('off')
            axs[i,2].set_title('attack: '+categories[attack_labels[i]])
            axs[i,2].imshow(adv_im[i].permute((1,2,0)).cpu().detach().numpy())
            axs[i,2].axis('off')
            


def calculate_result(adver_deltas,adv_acc,time):
    adver_deltas = adver_deltas.cpu().detach()
    if len(adver_deltas.shape) == 3 :
        adver_deltas = adver_deltas[None,:]
    norm1 = l1_norm(adver_deltas)
    norm2 = l2_norm(adver_deltas)
    number_pert = perturbations_number(adver_deltas)
    norm0 = l0_norm(adver_deltas)
    linf = linf_norm(adver_deltas)

    accuracy = len(adv_acc[adv_acc == 1])/len(adv_acc)
    
    dict_ = {'norm1_mean' : np.round(norm1[0], 3),
             'norm1_std' : np.round(norm1[1], 3), 
             'norm1_max' : np.round(norm1[2], 3), 
             'norm2_mean' : np.round(norm2[0], 3),
             'norm2_std' : np.round(norm2[1], 3), 
             'norm2_max' : np.round(norm2[2], 3), 
             'number_pert_mean' : np.round(number_pert[0], 3),
             'number_pert_std' : np.round(number_pert[1], 3), 
             'number_pert_max' : np.round(number_pert[2], 3), 
             'norm0_mean' : np.round(norm0[0], 3),
             'norm0_std' : np.round(norm0[1], 3), 
             'norm0_max' : np.round(norm0[2], 3), 
             'linf_mean' : np.round(linf[0], 3),
             'linf_std' : np.round(linf[1], 3), 
             'linf_max' : np.round(linf[2], 3), 
             'accuracy' : round(accuracy,3),
             'time': time}
    return dict_

def print_norm_and_accuracy(result_dict):
    print('   norm1_mean:',result_dict['norm1_mean'] ,'      norm1_std:',result_dict['norm1_std'] ,'      norm1_max:',result_dict['norm1_max'] ,'\n',
          '   norm2_mean:',result_dict['norm2_mean'] ,'      norm2_std:',result_dict['norm2_std'] ,'      norm2_max:',result_dict['norm2_max'] ,'\n',
          '   *linf_mean:',result_dict['linf_mean'] ,'      linf_std:',result_dict['linf_std'] ,'      linf_max:',result_dict['linf_max'] ,'\n',
          '   *norm0_mean:',result_dict['norm0_mean'] ,'      norm0_std:',result_dict['norm0_std'] ,'      norm0_max:',result_dict['norm0_max'] ,'\n',
          '   *number_pert_mean:',result_dict['number_pert_mean'] ,'      number_pert_std:',result_dict['number_pert_std'] ,'      number_pert_max:',result_dict['number_pert_max'] ,'\n',
          '    ****accuracy*****:' , result_dict['accuracy'],'\n')



