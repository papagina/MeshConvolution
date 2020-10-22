"""
This code is written by Sitao Xiang and modified by Yi Zhou before Yi Zhou joined Facebook.

"""



import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from datetime import datetime


class Affine(nn.Module):

    def __init__(self, num_parameters, scale = True, bias = True, scale_init= 1.0):
        super(Affine, self).__init__()
        if scale:
            self.scale = nn.Parameter(torch.ones(num_parameters)*scale_init)
            
        else:
            self.register_parameter('scale', None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_parameters))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        output = input
        if self.scale is not None:
            scale = self.scale.unsqueeze(0)
            while scale.dim() < input.dim():
                scale = scale.unsqueeze(2)
        output = output.mul(scale)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0)
            while bias.dim() < input.dim():
                bias = bias.unsqueeze(2)
        output += bias

        return output

def compute_loss(input):
    input_flat = input.view(input.size(1), input.numel() // input.size(1))
    mean = input_flat.mean(1)
    lstd = (input_flat.pow(2).mean(1) - mean.pow(2)).sqrt().log()
    return mean.pow(2).mean() + lstd.pow(2).mean()

class BatchStatistics(nn.Module):
    def __init__(self, affine = -1):
        super(BatchStatistics, self).__init__()
        self.affine = nn.Sequential() if affine == -1 else Affine(affine)
        self.loss = 0
    
    def clear_loss(self):
        self.loss = 0

    def forward(self, input):
        self.loss = compute_loss(input)
        return self.affine(input)

class Conv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias)
        self.weight.data.normal_()
        weight_norm = self.weight.pow(2).sum(2, keepdim = True).sum(1, keepdim = True).add(1e-8).sqrt()
        self.weight.data.div_(weight_norm)
        if bias:
            self.bias.data.zero_()

class Meshconv(nn.Module):
    def __init__(self, in_channels, out_channels, connection_matrix,  has_addon=True):
        super(Meshconv, self).__init__()
        
        self.conv1d = Conv1d(in_channels, out_channels, kernel_size=1,stride=1,padding=0, bias=True)
        self.connection_matrix= connection_matrix
        out_size =self.connection_matrix.shape[0]
        
        self.has_addon=has_addon
        if(has_addon==True):
            self.addon = nn.Parameter(torch.zeros(1,out_channels, out_size))
            self.addcon_conv1d = Conv1d(out_channels*2, out_channels, kernel_size=1,stride=1,padding=0,bias=True)
            
            
    def forward(self, input_data):
        batch = input_data.shape[0]
        out_data =input_data.clone()
        out_data = self.conv1d(input_data)
        in_size = self.connection_matrix.shape[1]
        out_size= self.connection_matrix.shape[0]
        if((in_size*out_size*batch)<1234567):
            out_data= out_data.transpose(2,1) #batch*in_size*channel
            connection_matrix_batch = self.connection_matrix.view(1, out_size, in_size).repeat(batch,1,1) #batch*out_size*in_size
            out_data = torch.matmul(connection_matrix_batch, out_data) #batch*out_size*channel
            out_data = out_data.transpose(1,2) #batch*channel*out_size
        else:
            out_data= out_data.transpose(2,1) #batch*in_size*channel
            out_data2 = torch.FloatTensor([]).cuda()
            for b in range(batch):
                out_instance = torch.matmul(self.connection_matrix, out_data[b])
                out_data2=torch.cat((out_data2, out_instance.unsqueeze(0)),0)
            out_data= out_data2 #batch*out_size*channel
            out_data = out_data.transpose(1,2) #batch*channel*out_size
        if(self.has_addon):
            out_data = torch.cat((out_data, self.addon.repeat(batch,1,1)),1)
            out_data = self.addcon_conv1d(out_data)
            
            #out_data = self.affine(out_data.contiguous().view(batch, -1)).contiguous().view(batch, -1,out_size) #batch*channel*out_size
        return out_data


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight.data.normal_()
        weight_norm = self.weight.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
        self.weight.data.div_(weight_norm)
        if bias:
            self.bias.data.zero_()

class CPReLU(nn.Module):

    def __init__(self, num_parameters = 1, init = 0.25):
        super(CPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

        self.post_mean = (1 - init) / math.sqrt(2 * math.pi)
        post_ex2 = (1 + init ** 2) / 2
        self.post_stdv = math.sqrt(post_ex2 - self.post_mean ** 2)

    def forward(self, input):
        return (F.prelu(input, self.weight) - self.post_mean) / self.post_stdv

class ResidueBlock(nn.Module):

    def __init__(self, in_channels, out_channels, residue_ratio, connection_matrix, use_BatchStatistics = False):
        super(ResidueBlock, self).__init__()

        self.residue_ratio = math.sqrt(residue_ratio)
        self.shortcut_ratio = math.sqrt(1 - residue_ratio)
        self.connection_matrix = connection_matrix

        out_c1 = int( (out_channels-in_channels)*0.3 +in_channels)
        out_c2 = int( (out_channels-in_channels)*0.6 +in_channels)
        out_c3 = int( (out_channels-in_channels)*1 +in_channels)
        print (out_c1,out_c2,out_c3)
        self.residue = nn.Sequential(
            Conv1d(in_channels, out_c1,kernel_size=1 ),
            BatchStatistics(out_c1) if use_BatchStatistics==True else nn.Sequential(),
            CPReLU(out_c1),
            Conv1d(out_c1, out_c2,kernel_size=1 ),
            BatchStatistics(out_c2) if use_BatchStatistics==True else nn.Sequential(),
            CPReLU(out_c2),
            Meshconv(out_c2, out_c3, connection_matrix, has_addon=False),
            BatchStatistics(out_c3) if use_BatchStatistics==True else nn.Sequential(),
            CPReLU(out_c3)
        )
        
        out_size = connection_matrix.shape[0]
        in_size = connection_matrix.shape[1]
        self.self_connection_matrix = connection_matrix - connection_matrix.max(1)[0].view(out_size,1).repeat(1,in_size)
        self.self_connection_matrix = (self.self_connection_matrix>=0).float()*1
        if(residue_ratio<1):
            self.shortcut = nn.Sequential(
                    Meshconv(in_channels, out_channels, self.self_connection_matrix, has_addon=False),
                    BatchStatistics(out_channels) if (use_BatchStatistics==True) else nn.Sequential()
                    )

    def forward(self, input):
        if(self.residue_ratio == 1):
            return self.residue(input)
        else:        
            return self.shortcut(input).mul(self.shortcut_ratio) + self.residue(input).mul(self.residue_ratio)



    
    
    
    
