import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
import sys
from torchinfo import summary
from datetime import datetime

from fast_transformers.builders import AttentionBuilder


class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.Tanh()#nn.ReLU()#
        self.act2 = nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        # first layer
        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        # middle layer
        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        # last layer
        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.Tanh()

        self.skip     = nn.Identity()#nn.Linear(size,size)
        self.act_skip = nn.Tanh()

    def forward(self, x):
        x_skip = self.act_skip(self.skip(x))

        o1 = self.act1(self.norm1(self.layer1(x)/np.sqrt(10)))
        o2 = self.act2(self.norm2(self.layer2(o1)/np.sqrt(10)))
        o3 = self.norm3(self.layer3(o2))
        o  = self.act3(o3+x_skip)

        return o

class DenseBlock(nn.Module):
    def __init__(self, size):
        super(DenseBlock, self).__init__()

        self.skip = nn.Identity()

        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, size)

        self.norm1 = torch.nn.BatchNorm1d(size)
        self.norm2 = torch.nn.BatchNorm1d(size)

        self.act1 = nn.Tanh()#nn.SiLU()#nn.PReLU()
        self.act2 = nn.Tanh()#nn.SiLU()#nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)
        o1    = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2    = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10)
        o     = torch.cat((o2,xskip),axis=1)
        return o


class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim, self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim, self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim, self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

##########################################################################################

class Linearized_Softmax_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Linearized_Softmax_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim, self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim, self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.phi          = nn.ELU()
        
        #self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        #self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        # phi(x) generally acts on a row (q_i/k_i), so when using a phi kernel that requires the entire row
        # specify dim = -1 via self.phi(x, dim = -1)
        
        Q = self.phi(self.WQ(_x)) + 1 # query with q_i as rows
        K = self.phi(self.WK(_x)) + 1 # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows
        
        
        KV = torch.einsum("...ij,...ik->...kj", K, V)
        Ksum = K.sum(dim = 1)
        norm = 1/(torch.einsum("...ld,...d->...l",Q,Ksum)+1e-7)
        prod = torch.einsum("...ld,...md,...l->...lm",Q,KV,norm)
        
        
        #dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        #normed_mat  = self.act(dot_product/self.scale)
        #prod        = torch.bmm(normed_mat,V)
        #prod        = torch.zeros_like(_x)
        
        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out
    
##########################################################################################

class Multi_Head_Attention(nn.Module):
    def __init__(self, in_size, n_partitions, num_heads):
        super(Multi_Head_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.num_heads    = num_heads
        
        self.WQ           = nn.Linear(self.embed_dim, self.embed_dim*self.num_heads)
        self.WK           = nn.Linear(self.embed_dim, self.embed_dim*self.num_heads)
        self.WV           = nn.Linear(self.embed_dim, self.embed_dim*self.num_heads)
        self.WO           = nn.Linear(self.embed_dim*self.num_heads, self.embed_dim)

        self.act          = nn.Softmax(dim = 2) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        quers = self.WQ(_x) # query with q_i as rows
        keys = self.WK(_x) # key   with k_i as rows
        vals = self.WV(_x) # value with v_i as rows
        
        ## SHAPE TRANSFORMATION
        
        quers = quers.reshape(quers.shape[0], quers.shape[1], self.num_heads, -1)
        keys = keys.reshape(keys.shape[0], keys.shape[1], self.num_heads, -1)
        vals = vals.reshape(vals.shape[0], vals.shape[1], self.num_heads, -1)
        
        quers = quers.permute(0, 2, 1, 3) #batch, num_heads, n_partitions, embed_dim
        keys = keys.permute(0, 2, 1, 3)   #batch, num_heads, n_partitions, embed_dim
        vals = vals.permute(0, 2, 1, 3)   #batch, num_heads, n_partitions, embed_dim

        ## ATTENTION
        dot_product = torch.einsum("...iq, ...jq->...ij", quers, keys) #Q.(K^T) batch, num_heads, n_partitions, n_partitions
        
        #dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.einsum("...ij, ...jk->...ik", normed_mat, vals) 
        #(QKT).V [batch, num_heads, n_partitions, embed_dim]
        
        #torch.bmm(normed_mat,V) 
        
        ## UNDO SHAPE TRANSFORMATION
        
        prod = prod.permute(0, 2, 1, 3)
        prod = prod.reshape(prod.shape[0], prod.shape[1], -1) #[batch, n_partitions, embed_dim*num_heads]
        
        out = self.WO(prod)
        out = torch.reshape(out, (batch_size, -1)) + x

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        #out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

##########################################################################################


class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = nn.Tanh()#nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights = nn.Parameter(weights) # turn the weights tensor into trainable weights
        bias = torch.Tensor(in_size)
        self.bias = nn.Parameter(bias) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5)) # matrix weights init 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights) # fan_in in the input size, fan out is the output size but it is not use here
        bound = 1 / np.sqrt(fan_in) 
        nn.init.uniform_(self.bias, -bound, bound) # bias weights init

    def forward(self,x):
        mat = torch.block_diag(*self.weights) # how can I do this on init rather than on each forward pass?
        x_norm = self.norm(x)
        _x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        o = self.act(torch.matmul(x_norm,mat)+self.bias)
        return o+x

class True_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(True_Transformer, self).__init__()  
    
        self.int_dim      = in_size//n_partitions
        self.n_partitions = n_partitions
        self.linear       = nn.Linear(self.int_dim,self.int_dim)#ResBlock(self.int_dim,self.int_dim)#
        self.act          = nn.ReLU()
        self.norm         = torch.nn.BatchNorm1d(self.int_dim*n_partitions)

    def forward(self,x):
        batchsize = x.shape[0]
        out = torch.reshape(self.norm(x),(batchsize,self.n_partitions,self.int_dim))
        out = self.act(self.linear(out))
        out = torch.reshape(out,(batchsize,self.n_partitions*self.int_dim))
        return out

class nn_pca_emulator:
    def __init__(self, 
                  model,
                  dv_fid, dv_std, cov_inv,
                  evecs,
                  device,
                  optim=None, lr=1e-3, reduce_lr=True, scheduler=None,
                  weight_decay=1e-3,
                  dtype='float'):
         
        self.optim        = optim
        self.device       = device 
        self.dv_fid       = torch.Tensor(dv_fid)
        self.cov_inv      = torch.Tensor(cov_inv)
        self.dv_std       = torch.Tensor(dv_std)
        self.evecs        = evecs
        self.reduce_lr    = reduce_lr
        self.model        = model
        self.trained      = False
        self.weight_decay = weight_decay
        
        if self.optim is None:
            print('Learning rate = {}'.format(lr))
            print('Weight decay = {}'.format(weight_decay))
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr ,weight_decay=self.weight_decay)
        if self.reduce_lr == True:
            print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',patience=10)#,min_lr=1e-12)#, factor=0.5)
        if dtype=='double':
            torch.set_default_dtype(torch.double)
            print('default data type = double')
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.generator=torch.Generator(device=self.device)

    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=1000, n_epochs=150):
        summary(self.model)
        print('Batch size = ',batch_size)
        print('N_epochs = ',n_epochs)

        # get normalization factors
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True))
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True))
            self.y_mean = self.dv_fid
            self.y_std  = self.dv_std

        # initialize arrays
        losses_train = []
        losses_vali = []
        loss = 100.

        # send everything to device
        self.model.to(self.device)
        tmp_y_std        = self.y_std.to(self.device)
        tmp_cov_inv      = self.cov_inv.to(self.device)
        tmp_X_mean       = self.X_mean.to(self.device)
        tmp_X_std        = self.X_std.to(self.device)
        tmp_X_validation = (X_validation.to(self.device) - tmp_X_mean)/tmp_X_std
        tmp_Y_validation = y_validation.to(self.device)

        # Here is the input normalization
        X_train     = ((X - self.X_mean)/self.X_std)
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        validset    = torch.utils.data.TensorDataset(tmp_X_validation,tmp_Y_validation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=self.generator)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=self.generator)
    
        print('Datasets loaded!')
        print('Begin training...')
        train_start_time = datetime.now()
        for e in range(n_epochs):
            start_time = datetime.now()
            self.model.train()
            losses = []
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device)
                Y_batch = data[1].to(self.device)
                Y_pred  = self.model(X) * tmp_y_std

                # PCA part
                diff = Y_batch - Y_pred
                loss1 = (diff \
                        @ tmp_cov_inv) \
                        @ torch.t(diff)

                ### remove the largest 2 from each batch (ES TESTING!)
                # loss_arr = torch.diag(loss1)
                # sort_loss = torch.sort(loss_arr)
                # loss = torch.mean(sort_loss[:-2])
                ### END TESTING

                loss = torch.mean(torch.diag(loss1)) # commented out for testing
                losses.append(loss.cpu().detach().numpy())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(np.mean(losses))
            ###validation loss
            with torch.no_grad():
                self.model.eval()
                losses = []
                for i, data in enumerate(validloader):  
                    X_v       = data[0].to(self.device)
                    Y_v_batch = data[1].to(self.device)

                    Y_v_pred = self.model(X_v) * tmp_y_std

                    v_diff = Y_v_batch - Y_v_pred 
                    loss_vali1 = (v_diff \
                                    @ tmp_cov_inv) @ \
                                    torch.t(v_diff)

                    ### remove the largest 2 from each batch (ES TESTING!)
                    # loss_vali_arr = torch.diag(loss_vali1)
                    # sort_vali_loss = torch.sort(loss_vali_arr)
                    # loss_vali = torch.mean(sort_vali_loss[:-2])
                    ### END TESTING
                    loss_vali = torch.mean(torch.diag(loss_vali1)) # commented out and replaced with testing portion above
     
                    losses.append(np.float(loss_vali.cpu().detach().numpy()))

                losses_vali.append(np.mean(losses))
                if self.reduce_lr:
                    self.scheduler.step(losses_vali[e])

            end_time = datetime.now()
            print('epoch {}, loss={}, validation loss={}, lr={} (epoch time: {})'.format(
                        e,
                        losses_train[-1],
                        losses_vali[-1],
                        self.optim.param_groups[0]['lr'],
                        (end_time-start_time).total_seconds()
                    ))#, total runtime: {} ({} average))
        
        np.savetxt("losses.txt", np.array([losses_train,losses_vali],dtype=np.float64))
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            y_pred = (self.model((X - self.X_mean) / self.X_std) * self.dv_std) #normalization

        y_pred = y_pred @ torch.linalg.inv(self.evecs)+ self.dv_fid # convert back to data basis
        return y_pred.cpu().detach().numpy()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['evecs']  = self.evecs
        
    def load(self, filename, device=torch.device('cpu'), state_dict=False):
        self.trained = True
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        if state_dict==False:
            self.model = torch.load(filename,map_location=device)
        else:
            print('Loading with "torch.load_state_dict(torch.load(file))"...')
            self.model.load_state_dict(torch.load(filename,map_location=device))
        #summary(self.model)
        self.model.eval()
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
            self.evecs  = torch.Tensor(f['evecs'][:])


