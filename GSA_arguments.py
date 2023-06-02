@author: Alireza Heshmati
"""

import argparse

def arguments():
    '''
    input hyper parameters
    '''
    parser = argparse.ArgumentParser(description='Gradual_Sparse_Attack')
    
    #general setting
    
    parser.add_argument('-f')
    
    parser.add_argument('--model_type', type=str, default='convnet',
                        help='base model, e.g. convnet, mobilenetv2, robustnet,inceptionv3, vits16')
    
    parser.add_argument('--dataset_type', type=str, default='cifar10',
                        help='base dataset, e.g. cifar10 and imagenet')
    
    parser.add_argument('--target_type', type=str, default='Untargeted',
                        help='kind of target or untargeted attack, just Untargeted, Best, Average, Worst')
    
    parser.add_argument('--o_f_type', type=str, default='CE',
                        help='objective function type, just CE (Cross Entropy) and CW')
    
    parser.add_argument('--num_data', type= int, default=1024,
                        help='number of data for attack, for cifar10 : 1024 and for imagenet : 100')
    
    parser.add_argument('--modifying', type= bool, default= True,
                        help='modifying mode to relax parameters for successful attack, or direct mode')
    
    parser.add_argument('--attack_batch', type= int, default= 1024,
                        help='number of batch elements for attack')
     
    # Gradual attack setting
    
    parser.add_argument('--lr_max', type= float, default=  0.01 ,
                        help='maximum of learning rate')
    # for min in 5
    parser.add_argument('--lr_min', type= float, default=  0.001 ,
                        help='minimum of learning rate')
    
    parser.add_argument('--lr_s', type= float, default=  0.85 ,
                        help='scaling of learning rate to lr_min')
    
    parser.add_argument('--mu_max', type= float, default=  1 ,
                        help='maximum of mu in the paper')
    
    parser.add_argument('--mu_min', type= float, default=  0.01 ,
                        help='minimum of mu in the paper')
    
    parser.add_argument('--s', type= float, default= 0.72  ,
                        help='scaling for sparsity (mu)')
    
    parser.add_argument('--lambda1', type= float, default= 0.2  ,
                        help='\lambda1 is Rs relaxation')
    
    parser.add_argument('--lambda1_min', type= float, default= 0.02  ,
                        help='minimum of \lambda1 for modifying mode')
    
    parser.add_argument('--lambda1_s', type= float, default= 0.2  ,
                        help='decading of \lambda1 for modifying mode')
    
    parser.add_argument('--lambda2', type= float, default=  1 ,
                        help='\lambda2 in E in the paper')
    
    parser.add_argument('--lambda2_min', type= float, default=  0.1 ,
                        help='minimum of \lambda2 for modifying mode')
    
    parser.add_argument('--N1', type= int , default=  20 ,
                        help='N1 outer iterations')
    
    parser.add_argument('--N2', type= int , default= 100  ,
                        help='N2 inner iterations')
    
    parser.add_argument('--sparsity_type', type= str, default= 'SCAD' ,
                        help='type of sparsity, just SCAD, HT and ST')
    
    parser.add_argument('--theta', type= float, default= 3.7 ,
                        help='\theta for SCAD')
    
    parser.add_argument('--zita_inf', type= float, default= 0.03 ,
                        help='\zita in proximal linf')
    
    parser.add_argument('--zita_inf_s', type= float, default= 2  ,
                        help='scaling of \zita_inf for modifying mode')
    
    parser.add_argument('--max_zita_inf', type= float, default= 1 ,
                        help='maximum \zita in proximal linf for modifying mode')
    
    parser.add_argument('--plot', type= bool, default= False,
                        help='ploting some outputs of attack and originals')
    
    parser.add_argument('--number_of_plots', type= int, default= 10,
                        help='number of images for plotting')
    return parser
