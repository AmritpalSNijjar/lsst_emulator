import torch
import torch.nn as nn
from fast_transformers.attention import AttentionLayer
from fast_transformers.attention import FullAttention
from fast_transformers.masking import FullMask, TriangularCausalMask
from cocoa_emu.nn_emulator import Better_Transformer, Affine

from fast_transformers.attention import FullAttention, LinearAttention, ReformerAttention, AFTFullAttention

class AttnLayer(nn.Module):
    def __init__(self, in_size, n_partitions, n_heads, att, latent_dim = -1):
        super(AttnLayer, self).__init__()
        
        self.n_heads       = n_heads
        self.n_partitions  = n_partitions
        self.d_model       = in_size//n_partitions
        
        # Types of Attention I tested:
        
        # att = FullAttention()  <- Standard
        # att = LinearAttention(int_dim_trf//n_channels)
        # att = ReformerAttention()
        # att = AFTFullAttention(max_sequence_length = n_channels, aft_parameterization = n_channels)
        
        # Details for the above functions can be found at:
        # https://fast-transformers.github.io/api_docs/fast_transformers/attention/
        
        self.layer         = AttentionLayer(att, self.d_model, n_heads, self.d_model, self.d_model)
        
        if latent_dim > 0:
            self.WQ_down      = nn.Linear(self.latent_dim, self.d_model)
            self.WQ_up        = nn.Linear(self.d_model, self.latent_dim)

            self.WK_down      = nn.Linear(self.latent_dim, self.d_model)
            self.WK_up        = nn.Linear(self.d_model, self.latent_dim)

            self.WV_down      = nn.Linear(self.latent_dim, self.d_model)
            self.WV_up        = nn.Linear(self.d_model, self.latent_dim)
            
        else:
            self.WQ            = nn.Linear(self.d_model, self.d_model)
            self.WK            = nn.Linear(self.d_model, self.d_model)
            self.WV            = nn.Linear(self.d_model, self.d_model)
        
        self.attn_mask     = None
        self.query_lengths = None
        self.key_lengths   = None
        
        self.norm          = nn.LayerNorm(in_size)
        
    def forward(self, x):
        x_norm = self.norm(x)
        
        B = x.shape[0]        # Batch Size
        L = self.n_partitions # Sequence Length
        
        _x = x_norm.reshape(B, L, self.d_model)
        
        if self.attn_mask is None:
            self.attn_mask     = FullMask(L, L, device=x.device) # Which keys can attend to which queries?
            self.query_lengths = FullMask(B, L, device=x.device) # Query Lengths (= n_partitions for all our data)
            self.key_lengths   = FullMask(B, L, device=x.device) # Key Lengths   (= n_partitions for all our data)
        
        if latent_dim > 0:
            Q_interim = self.WQ_down(_x)
            queries = self.WQ_up(Q_interim)

            K_interim = self.WK_down(_x)
            keys = self.WK_up(K_interim)

            V_interim = self.WV_down(_x)
            values = self.WV_up(V_interim)
        else:
            queries = self.WQ(_x)
            keys = self.WK(_x)
            values = self.WV(_x)
        
        result = self.layer(queries, keys, values, self.attn_mask, self.query_lengths, self.key_lengths)
        
        out = torch.reshape(result, (B, -1)) + x
        
        return out
    
class split_res_trfs(nn.Module):
    
    def __init__(self, res, trfs):
        super(split_res_trfs, self).__init__()
        
        # res and trfs are both complete models (nn.sequential etc.)
        
        # res should be pretrained
        # res(x{B, 12}) = out{B, 780}
                        # out{B, 780} is low-res cosmic shear data
        
        # trfs needs to be trained
        # trfs(x{B, 780}) = out{B, 780}
        res.eval()
        self.res = res
        
        for param in res.parameters():
            param.requires_grad = False
        
        self.trfs = trfs
        
        self.aff = Affine()
        
    def forward(self, x):
        
        # the res chunk is pretrained, so we do not want to calculate a gradient here
        with torch.no_grad():
            o1 = self.res(x)
        
        o2 = self.trfs(o1)
        out = self.aff(o2)
        
        return out
    
    
class double_attn_block(nn.Module):
    
    def __init__(self, in_size, n_partitions_half, n_heads, att):
        super(double_attn_block, self).__init__()
        
        # in_size MUST be an even number
        # n_partitions_half MUST be a factor of in_size/2

        self.half_len = in_size//2
        self.attn1 = AttnLayer(in_size//2, n_partitions_half, n_heads, att)
        self.attn2 = AttnLayer(in_size//2, n_partitions_half, n_heads, att)
        
        self.trans1 = Better_Transformer(in_size//2, n_partitions_half)
        self.trans2 = Better_Transformer(in_size//2, n_partitions_half)
        
    def forward(self, x):
        
        (half1, half2) = torch.split(x, self.half_len, dim = 1)
        
        out1 = self.attn1(half1)
        out2 = self.attn2(half2)
        
        out11 = self.trans1(out1)
        out22 = self.trans2(out2)
        
        out = torch.cat((out11, out22), dim = 1)
        
        return out