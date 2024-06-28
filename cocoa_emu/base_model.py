import torch
import torch.nn as nn
import pickle
import sys,os

from cocoa_emu.nn_emulator import Affine, ResBlock, ResBottle, DenseBlock, True_Transformer, Better_Attention, Better_Transformer, Multi_Head_Attention, Linearized_Softmax_Attention
from cocoa_emu.custom_attention import AttnLayer

from fast_transformers.attention import FullAttention, LinearAttention, ReformerAttention, AFTFullAttention


class base_model_customizable(nn.Module):
    
    def __init__(self, att_type = "full", n_heads = 1, in_dim = 12, int_dim_res = 256, n_channels = 32, int_dim_trf = 1024, out_dim = 780):
        super(base_model_customizable, self).__init__()
        
        self.layers = []
        
        if att_type == "full":
            att = FullAttention()
        elif att_type == "lin":
            att = LinearAttention(int_dim_trf//n_channels)
        elif att_type == "lsh":
            att = ReformerAttention()
        elif att_type == "aft":
            att = AFTFullAttention(max_sequence_length = n_channels, aft_parameterization = n_channels)
        
        self.layers.append(nn.Linear(in_dim, int_dim_res))
        self.layers.append(ResBlock(int_dim_res, int_dim_res))
        self.layers.append(ResBlock(int_dim_res, int_dim_res))
        self.layers.append(ResBlock(int_dim_res, int_dim_res))
        # layers.append(ResBlock(int_dim_res, int_dim_res))
        self.layers.append(nn.Linear(int_dim_res, int_dim_trf))

        #layers.append(Better_Attention(int_dim_trf, n_channels))
        self.layers.append(AttnLayer(int_dim_trf, n_channels, n_heads, att))

        self.layers.append(Better_Transformer(int_dim_trf, n_channels))

        #layers.append(Better_Attention(int_dim_trf, n_channels))
        self.layers.append(AttnLayer(int_dim_trf, n_channels, n_heads, att))

        self.layers.append(Better_Transformer(int_dim_trf, n_channels))

        #layers.append(Better_Attention(int_dim_trf, n_channels))
        self.layers.append(AttnLayer(int_dim_trf, n_channels, n_heads, att))

        self.layers.append(Better_Transformer(int_dim_trf, n_channels))

        self.layers.append(nn.Linear(int_dim_trf, out_dim))
        self.layers.append(Affine())

        self.compiled_model = nn.Sequential(*self.layers)
    
    def forward(self, x):
        
        out = self.compiled_model(x)
        
        return out

#models_dict = {"base_attn_1head" : {"att_type" :"full", "n_heads":1}, \
#               "linear_attn_1head" : {"att_type":"lin", "n_heads":1}}


#with open('models_dict.pickle', 'wb') as handle:
#    pickle.dump(models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def get_model(model_name):
    with open('/home/grads/data/amritpal/cocoa_lsst_emu/emulator_output/models/models_dict.pickle', 'rb') as f:
        models_dict = pickle.load(f)
    return base_model_customizable(**models_dict[model_name])
