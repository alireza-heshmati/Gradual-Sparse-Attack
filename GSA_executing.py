@author: Alireza Heshmati

import torch
from Utils import creat_targets
from GSA_utils import gradual_sparse_attack
import numpy as np

# use cuda if it is available
if torch.cuda.device_count() != 0:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')

# executing GSA cuda if it is available    
def GSA_executing(model, images, labels, args):
    torch.manual_seed(12)
    torch.cuda.manual_seed_all(12)
    torch.backends.cudnn.deterministic = True
    np.random.seed(12)
    
    images = images.to(device)  
    labels = labels.to(device)
    batch = args.attack_batch
    
    adv_deltas = torch.zeros_like(images)
    adv_acc = torch.zeros_like(labels)
    
    if args.dataset_type == 'cifar10' :
        num_class = 10
    else :
        num_class = 1000 


    # initialization
    outputs = model(images)
    one_hot_labels = torch.eye(num_class)[labels].to(device)
    targets = creat_targets(outputs, one_hot_labels, args.target_type)
    
    ind = adv_acc < 1
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    zita_inf = args.zita_inf
    while_do = True
	
    while while_do :
        temp_delta = torch.zeros_like(images[ind])
        temp_acc = torch.ones_like(labels[ind])
        for i in range(0,len(temp_delta), batch):
            if targets == None:
                targets_ = None
            else:
                targets_ = targets[ind][i:i+batch]
            
            deltas_temp, adv_acc_temp = gradual_sparse_attack( model, images[ind][i:i+batch],labels[ind][i:i+batch],one_hot_labels[ind][i:i+batch],targets_,args)
            
            temp_delta[i:i+batch] = deltas_temp
            temp_acc[i:i+batch] =  adv_acc_temp
        adv_deltas[ind] = temp_delta
        adv_acc[ind] =temp_acc
        
        ind = adv_acc < 1
        
        print('number of attacked images : ',len(adv_acc[~ind]))
        
	# continue if modifying mode is on    
        if torch.all(~ind) or args.modifying != True :
            while_do = False
            
        elif args.lambda2 > args.lambda2_min  and ( args.dataset_type == 'cifar10' or args.zita_inf > 0.1):
            args.lambda2 = args.lambda2_min
            print('chanage lambda2 :',args.lambda2)
            
        elif args.lambda1 > args.lambda1_min :
            args.lambda1 = max(round(args.lambda1 * args.lambda1_s, 3), args.lambda1_min)
            print('chanage lambda1 :',args.lambda1)
            
        else :
            args.lambda1 = lambda1
            if args.zita_inf <= 0.1 :
                args.lambda2 = lambda2
                
            if args.zita_inf >= args.max_zita_inf:
                while_do = False
            else:
                args.zita_inf = round(args.zita_inf * args.zita_inf_s , 4)
                print('chanage zita_inf :',args.zita_inf)
            
    args.lambda1 = lambda1
    args.lambda2 = lambda2
    args.zita_inf = zita_inf
    print('\n')

    return adv_deltas.detach().cpu(), adv_acc.detach().cpu().numpy()