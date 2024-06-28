##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import pickle
from cocoa_emu import cocoa_config, nn_pca_emulator
from cocoa_emu.base_model import get_model
from torchinfo import summary
from datetime import datetime
import sys


from cocoa_emu.nn_emulator import Affine, ResBlock, ResBottle, DenseBlock, True_Transformer, Better_Attention, Better_Transformer, Multi_Head_Attention, Linearized_Softmax_Attention
from cocoa_emu.custom_attention import AttnLayer, split_res_trfs, double_attn_block
from fast_transformers.attention import FullAttention, LinearAttention, ReformerAttention

def get_chi2(dv_predict, dv_exact, mask, cov_inv):
    delta_dv = (dv_predict - np.float32(dv_exact))[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv_masked)) , delta_dv  )   
    return chi2

# adjust config
configfile = sys.argv[1]#'./projects/lsst_y1/train_emulator.yaml'
config = cocoa_config(configfile)

# validation samples
# # T=256
# suffix = 1
# file = './projects/lsst_y1/emulator_output/chains/train_t256'

# # T=128
suffix = 1
file = '/home/grads/extra_data/evan/cosmic_shear_training_data/train_t128'
#'./projects/lsst_y1/emulator_output/chains/train_t128'


# T=64
# suffix = 0
# file = './projects/lsst_y1/emulator_output/chains/valid_post_T64_atplanck' 


samples_validation = np.load(file+'_samples_'+str(suffix)+'.npy')
dv_validation      = np.load(file+'_data_vectors_'+str(suffix)+'.npy')[:,:780]#[::1,:780]

# thin
target_n = 10000
thin_factor = len(samples_validation)//target_n
if thin_factor!=0:
    samples_validation = samples_validation[::thin_factor]
    dv_validation      = dv_validation[::thin_factor]

# adjust as needed
OUTPUT_DIM=780

mask=config.mask[:OUTPUT_DIM]
cov_inv_masked = config.cov_inv_masked

logA   = samples_validation[:,0]
ns     = samples_validation[:,1]
H0     = samples_validation[:,2]
Omegab = samples_validation[:,3]
Omegac = samples_validation[:,4]

bin_count = 0
start_idx = 0
end_idx   = 0

# set needed parameters to initialize emulator
device=torch.device('cpu')
torch.set_num_threads(1) # `Intra-op parallelism
evecs=0


# get list of trained emus
#os.listdir('./projects/lsst_y1/emulator_output/models/for_tables/')
#model_list = [# 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_relu_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-1_Ntrf_1_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-5_Ntrf_1_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_128_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_128_lr_10-4',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_256_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_256_lr_10-4',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_1024_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_1024_lr_10-4',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-4',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_0_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_1_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_4_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_3_Nres_0_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_3_Nres_1_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_3_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_3_Nres_4_bs_2500_lr_10-3',
              # 'T128_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_1024_ch_32',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_1024_ch_128',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_256_ch_8',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_256_ch_32',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_256_ch_128',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_64_ch_8',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_64_ch_32',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_2048_ch_8',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_2048_ch_32',
              # 'T512_600k_tanh_lambda_10-3_Ntrf_1_Nres_3_bs_2500_lr_10-3_trfw_2048_ch_128',
              ###
              # 'T512_600k_optimal_Nres_0_Ntrf_1',
              # 'T512_600k_optimal_Nres_0_Ntrf_3',
              # 'T512_600k_optimal_Nres_1_Ntrf_1',
              # 'T512_600k_optimal_Nres_1_Ntrf_3',  
              # 'T512_600k_optimal_Nres_3_Ntrf_1',    
              # 'T512_600k_optimal_Nres_3_Ntrf_3',  
              # 'T512_600k_optimal_Nres_4_Ntrf_1',
              # 'T512_600k_optimal_Nres_4_Ntrf_3',
              ###
              # 'T512_600k_optimal_tanh', 
              # 'T512_3000k_tanh_Nres3_Ntrf3_optimal',  
              ###     
              # 'T512_600k_optimal',
              # 'T512_1200k_optimal',
              # 'T512_3000k_optimal',
              # 'T256_600k_optimal_run2',
              # 'T256_1200k_optimal_run2',
              # 'T256_1200k_optimal',
              # 'T256_3000k_optimal',
              # 'T128_600k_optimal',
              # 'T128_1200k_optimal',
              # 'T128_3000k_optimal',
              ###
              # 'T512_3000k_tanh_Nres3_Ntrf3_optimal',
              # 'T512_3000k_tanh_optimal_run2',
              # 'T512_1200k_tanh_Nres3_Ntrf3_optimal',
              # 'T512_600k_tanh_Nres3_Ntrf3_optimal',
              # 'T256_600k_tanh_optimal',
              # 'T256_1200k_tanh_optimal',
              # 'T256_3000k_tanh_optimal',
              # 'T128_600k_tanh_optimal',
              # 'T128_1200k_tanh_optimal',
              # 'T128_3000k_tanh_optimal',
              ###
              # 'T512_600k_relu_true_transformer',
              # 'T512_600k_relu_true_transformer_one_linear',
              ###
              # 'T512_600k_tanh_Nres3_Ntrf3_optimal',
              # 'T512_1200k_tanh_Nres3_Ntrf3_optimal',
              ###
              # 'T512_600k_tanh_take_2',
              ###
              # 'baseline_relu_run2',
              # 'baseline_relu_run3',
              # 'baseline_tanh_run_2',
              # 'baseline_tanh_run_3',
#]

results = []

models_folder = '/home/grads/data/amritpal/cocoa_lsst_emu/emulator_output/split_model_testing/'
model_list = os.listdir(models_folder) 


for model in ['split_LRRRL22AL_13hf39ch']:
    # open the trained emulator
    if '.h5' in model:
        continue # these files are not emulators
    if 'models_dict' in model:
        continue # ignore file models_dict.pickle and models_dict script(s)
    if '.ipynb' in model:
        continue # ignore .ipynb checkpoints.            # REMOVE THIS FILE MANUALLY AT GET RID OF THIS LINE
    if '__pycache__' in model:
        continue # ignore whatever this is.              # REMOVE THIS FILE MANUALLY AT GET RID OF THIS LINE
        
    
    print(model)
    
    # if using model_params_dict
    #emu_cs = nn_pca_emulator(get_model(model), config.dv_fid, 0, cov_inv_masked, evecs, device=device)
    
    #===============================#
    
    in_dim=12
    N_layers = 1
    int_dim_res = 256
    n_channels_half = 13   # make sure n_channels is a factor of int_dim_trf//2
    n_channels = 39
    int_dim_trf = 780      # 1024
    out_dim = 780
    n_heads = 1

    att = FullAttention()#CausalLinearAttention(int_dim_trf//n_channels)

    layers = []
    layers.append(nn.Linear(in_dim, int_dim_res))
    layers.append(ResBlock(int_dim_res, int_dim_res))
    layers.append(ResBlock(int_dim_res, int_dim_res))
    layers.append(ResBlock(int_dim_res, int_dim_res))
    layers.append(nn.Linear(int_dim_res, int_dim_trf))

    layers.append(Affine())

    res = nn.Sequential(*layers)

    model_file_path = '/home/grads/data/amritpal/cocoa_lsst_emu/emulator_output/split_model_testing/LRRRL'
    res.load_state_dict(torch.load(model_file_path, map_location=device))

    del layers

    layers = []

    layers.append(double_attn_block(int_dim_trf, n_channels_half, n_heads, att))
    layers.append(double_attn_block(int_dim_trf, n_channels_half, n_heads, att))

    layers.append(AttnLayer(int_dim_trf, n_channels, n_heads, att))
    layers.append(Better_Transformer(int_dim_trf, n_channels))
    #layers.append(AttnLayer(int_dim_trf, n_channels, n_heads, att))
    #layers.append(Better_Transformer(int_dim_trf, n_channels))
    layers.append(nn.Linear(int_dim_trf, out_dim))

    trfs = nn.Sequential(*layers)

    model_struct = split_res_trfs(res, trfs)

    #===============================#
    
    emu_cs = nn_pca_emulator(model_struct, config.dv_fid, 0, cov_inv_masked, evecs, device=device)
    
    # DOUBLE CHECK LOAD LOCATION!!!
    
    emu_cs.load(models_folder + model, state_dict = True)
    print('emulator(s) loaded\n')

    chi2_list=np.zeros(len(samples_validation))
    count_1 = 0 # for chi2>1
    count_2 = 0 # for chi2>0.2
    start_time=datetime.now()
    time_prev=start_time
    predicted_dv = np.zeros(OUTPUT_DIM)

    for j,point in enumerate(samples_validation):
        _j=j+1

        # get params and true dv
        theta = torch.Tensor(point)
        dv_truth = dv_validation[j]

        # reconstruct dv
        dv_cs = emu_cs.predict(theta[:12])[0]
        predicted_dv = dv_cs

        # compute chi2
        chi2 = get_chi2(predicted_dv, dv_truth, mask, cov_inv_masked)

        #count how many points have "poor" prediction.
        chi2_list[j] = chi2
        if chi2>1:
           count_1 += 1
        if chi2>0.2:
           count_2 += 1

        # progress check
        if j%10==0:
            runtime=datetime.now()-start_time
            print('\rprogress: '+str(j)+'/'+str(len(samples_validation))+\
                ' | runtime: '+str(runtime)+\
                ' | remaining time: '+str(runtime*(len(samples_validation)/_j - 1))+\
                ' | s/it: '+str(runtime/_j),end='')

    #summary
    #print("\naverage chi2 is: ", np.average(chi2_list))
    #print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
    #print("points with chi2 > 0.25: "+str(count)+" ( "+str((count*100)/len(samples_validation))+"% )")

    print('\n model: ',model)
    print("average chi2 is: {:.3f}".format(np.mean(chi2_list)))
    print("median chi2 is: {:.3f} ".format(np.median(chi2_list)))
    print('num points: {}'.format(len(chi2_list)))
    print('numer of points chi2>1 {}'.format(count_1))
    print('numer of points chi2>0.2 {}\n'.format(count_2))
    
    # np.savetxt('delta_chi2/'+model+'.txt',chi2_list)