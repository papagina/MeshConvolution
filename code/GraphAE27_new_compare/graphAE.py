# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import numpy as np
from ResidueBlock import Linear, Conv1d, CPReLU
from torch_geometric.nn import SplineConv, ChebConv, GMMConv, GATConv, FeaStConv

zero_threshold = 1e-8

class Model(nn.Module):
    def __init__(self, param, test_mode = False): #layer_info_lst= [(point_num, feature_dim)]
        super(Model, self).__init__()
        
        self.point_num = param.point_num

        self.test_mode = test_mode
        
        
        self.conv_max = param.conv_max
        
        self.perpoint_bias = param.perpoint_bias
        
        self.channel_lst = param.channel_lst

        self.residual_rate_lst = param.residual_rate_lst
        
        #self.channel_lst = [3]*14
        
        self.weight_num_lst = param.weight_num_lst
        
        self.connection_layer_fn_lst = param.connection_layer_fn_lst
        
        self.initial_connection_fn = param.initial_connection_fn
        
        self.use_vanilla_pool = param.use_vanilla_pool
        
        self.conv_method = param.conv_method

        self.relu = nn.ELU()

        self.batch = param.batch

        self.init_layers(self.batch)

        #self.final_linear = Conv1d(self.channel_lst[-1], 3, kernel_size=1)        
        
        #####For Laplace computation######
        self.initial_neighbor_id_lstlst = torch.LongTensor(param.neighbor_id_lstlst).cuda()#point_num*max_neighbor_num
        self.initial_neighbor_num_lst = torch.FloatTensor(param.neighbor_num_lst).cuda() #point_num
        
        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]
        
    """
    #num*channel
    def normalize_weights(self, weights):
        num = weights.shape[0]
        channel = weights.shape[1]
        #weights.normal_()
        weights_norm = weights.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
        weights =  weights/ weights_norm.view(num, 1).repeat(1, channel)
    """    
    def normalize_tensor(self, tensor, sum_dim):
        num = tensor.shape[0]
        channel = tensor.shape[sum_dim]
        norm = tensor.pow(2).sum(sum_dim, keepdim = True).add(1e-8).sqrt()
        repeat_dim = [1]*len(tensor.shape)
        repeat_dim[sum_dim] = tensor.shape[sum_dim]
        
        out_tensor = tensor/ norm.repeat(repeat_dim)

        return out_tensor

    #in out_point_num*max_neighbor_num
    #out 2*num_edges #there should not be any change in the vertex number.
    def get_edge_index_lst_from_neighbor_id_lstlst(self, neighbor_id_lstlst, neighbor_num_lst):
        edge_index_lst = set({})
        for i in range(len(neighbor_id_lstlst)):
            neighborhood=neighbor_id_lstlst[i]
            neighbor_num = int(neighbor_num_lst[i].item())
            for j in range(neighbor_num):
                neighbor_id = neighborhood[j]
                if(i<neighbor_id):
                    edge_index_lst.add((i,neighbor_id))
                elif(i>neighbor_id):
                    edge_index_lst.add((neighbor_id,i))
        
        edge_index_lst = torch.LongTensor(list(edge_index_lst)).cuda()
        edge_index_lst = edge_index_lst.transpose(0,1)
        return edge_index_lst

    #neighbor_num_lst point_num; edge_index_lst 2*edges
    #out edges*2
    def get_GMM_pseudo_coordinates(self, edge_index_lst, neighbor_num_lst):
        pseudo_coordinates = neighbor_num_lst[edge_index_lst.transpose(0,1)]
        pseudo_coordinates = 1/torch.sqrt(pseudo_coordinates)
        
        return pseudo_coordinates





                
            
    
    #channel_lst= [channel]
    ##connection_matrix_lst [out_feature*in_feature]
    def init_layers(self, batch):


        self.layer_lst = [] ##[in_channel, out_channel, in_pn, out_pn, max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst,conv_layer, residual_layer]
        
        self.layer_num = len(self.channel_lst)
        
        in_point_num = self.point_num
        in_channel = 3
        
        for l in range(self.layer_num):
            out_channel = self.channel_lst[l]
            weight_num = self.weight_num_lst[l]
            residual_rate = self.residual_rate_lst[l]
            

            connection_info  = np.load(self.connection_layer_fn_lst[l])
            print ("##Layer",self.connection_layer_fn_lst[l])
            out_point_num = connection_info.shape[0]

            neighbor_num_lst = torch.FloatTensor(connection_info[:,0].astype(float)).cuda() #out_point_num*1
            neighbor_id_dist_lstlst = connection_info[:, 1:] #out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1,2))[:,:,0] #out_point_num*max_neighbor_num
            #neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst)
            avg_neighbor_num = neighbor_num_lst.mean().item()
            max_neighbor_num = neighbor_id_lstlst.shape[1]

            pc_mask  = torch.ones(in_point_num+1).cuda()
            pc_mask[in_point_num]=0
            neighbor_mask_lst = pc_mask[ torch.LongTensor(neighbor_id_lstlst)].contiguous() #out_pn*max_neighbor_num neighbor is 1 otherwise 0
            
            

            if((residual_rate<0) or (residual_rate>1)):
                print ("Invalid residual rate", residual_rate)
            ####parameters for conv###############
            conv_layer = ""
            if(residual_rate<1):
                edge_index_lst = self.get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
                print ("edge_index_lst", edge_index_lst.shape)
                if(self.conv_method=="Cheb"):
                    conv_layer = edge_index_lst, ChebConv(in_channel, out_channel, K=6, normalization='sym', bias=True)
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()
                    ii=0
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()/10
                        self.register_parameter("GMM"+str(l)+"_"+str(ii),p)
                        ii+=1
                elif(self.conv_method=="GAT"):
                    #conv_layer = edge_index_lst,GATConv(in_channel, out_channel, heads=1)
                    if((l!=((self.layer_num/2)-2)) and (l!=(self.layer_num-1))): #not middle or last layer
                        conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=8, concat=True)
                        out_channel=out_channel*8
                    else:
                        conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=8, concat=False) 
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()
                    ii=0
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()
                        self.register_parameter("GAT"+str(l)+"_"+str(ii),p)
                        ii+=1
                    #conv_layer = edge
                elif(self.conv_method=="GMM"):
                    pseudo_coordinates = self.get_GMM_pseudo_coordinates( edge_index_lst, neighbor_num_lst) #edge_num*2
                    #print ("pseudo_coordinates", pseudo_coordinates.shape)
                    conv_layer = edge_index_lst,pseudo_coordinates, GMMConv(in_channel, out_channel, dim=2 , kernel_size=25)
                    ii=0
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()
                        self.register_parameter("GMM"+str(l)+"_"+str(ii),p)
                        ii+=1
                elif(self.conv_method=="FeaST"):
                    
                    conv_layer = edge_index_lst,FeaStConv(in_channel, out_channel, heads=32)
                    ii=0
                    for p in conv_layer[-1].parameters():
                        p.data=p.data.cuda()
                        self.register_parameter("FeaST"+str(l)+"_"+str(ii),p)
                        ii+=1
                elif(self.conv_method=="vw"):
                    weights = torch.randn(out_point_num, max_neighbor_num, out_channel,in_channel)/(avg_neighbor_num*weight_num)
                    
                    weights = nn.Parameter(weights.cuda())
                    
                    self.register_parameter("weights"+str(l),weights)
                    
                    bias = nn.Parameter(torch.zeros(out_channel).cuda())
                    if(self.perpoint_bias == 1):
                        bias= nn.Parameter(torch.zeros(out_point_num, out_channel).cuda())
                    self.register_parameter("bias"+str(l),bias)
                    
                    conv_layer = (weights, bias)
                else:
                    print ("ERROR! Unknown convolution layer!!!")
                

            zeros_batch_outpn_outchannel = torch.zeros((batch, out_point_num, out_channel)).cuda()
            ####parameters for residual###############
            residual_layer = ""

            if(residual_rate>0):
                p_neighbors = ""
                weight_res = ""

                if((out_point_num != in_point_num) and (self.use_vanilla_pool == 0)):
                    p_neighbors = nn.Parameter((torch.randn(out_point_num, max_neighbor_num)/(avg_neighbor_num)).cuda())
                    self.register_parameter("p_neighbors"+str(l),p_neighbors)
                
                if(out_channel != in_channel):
                    weight_res = torch.randn(out_channel,in_channel)
                    #self.normalize_weights(weight_res)
                    weight_res = weight_res/out_channel
                    weight_res = nn.Parameter(weight_res.cuda())
                    self.register_parameter("weight_res"+str(l),weight_res)
                
                residual_layer = (weight_res, p_neighbors)
            
            #####put everythin together
            
            layer = (in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros_batch_outpn_outchannel)
            
            self.layer_lst +=[layer]
            
            print ("in_channel", in_channel,"out_channel",out_channel, "in_point_num", in_point_num, "out_point_num", out_point_num, "weight_num", weight_num,
                   "max_neighbor_num", max_neighbor_num, "avg_neighbor_num", avg_neighbor_num)
            
            in_point_num=out_point_num
            in_channel = out_channel
         
            
    #use in train mode
    #input_pc batch*in_pn*in_channel
    #out_pc batch*out_pn*out_channel
    def forward_one_conv_layer_batch(self, in_pc, layer_info, is_final_layer=False):
        batch = in_pc.shape[0]
        
        in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate,neighbor_mask_lst, zeros_batch_outpn_outchannel=layer_info
        
        
        in_pc_pad = torch.cat((in_pc, torch.zeros(batch, 1, in_channel).cuda()), 1) #batch*(in_pn+1)*in_channel

        in_neighbors = in_pc_pad[:, neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*in_channel


        ####compute output of convolution layer####
        out_pc_conv = zeros_batch_outpn_outchannel.clone()

        if(len(conv_layer) != 0):
            out_pc_conv = torch.FloatTensor([]).cuda()
            if((self.conv_method == "GAT") or (self.conv_method=="Cheb")or(self.conv_method=="FeaST")):
                edge_index_lst, conv = conv_layer #weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
                for b in range(batch):
                    out_pc_one = conv(in_pc[b], edge_index_lst)
                    out_pc_conv = torch.cat( (out_pc_conv, out_pc_one.unsqueeze(0)), 0)
            elif((self.conv_method=="GMM")):
                edge_index_lst, pseudo_coordinates, conv= conv_layer
                for b in range(batch):
                    out_pc_one = conv(in_pc[b], edge_index_lst, pseudo_coordinates)
                    out_pc_conv = torch.cat( (out_pc_conv, out_pc_one.unsqueeze(0)), 0)
            elif(self.conv_method=="vw"):
                weights, bias = conv_layer
                out_neighbors = torch.einsum('pmoi,bpmi->bpmo',[weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel
                out_neighbors= out_neighbors*neighbor_mask_lst.view(1, out_pn, max_neighbor_num, 1).repeat(batch, 1,1, out_channel)
            
                out_pc_conv = out_neighbors.sum(2)
                
                out_pc_conv = out_pc_conv + bias
                
            if(is_final_layer == False):
                out_pc_conv = self.relu(out_pc_conv) ##self.relu is defined in the init function
        
        #if(self.residual_rate==0):
        #    return out_pc
        ####compute output of residual layer####
        out_pc_res = zeros_batch_outpn_outchannel.clone()

        if(len(residual_layer)!=0):
            weight_res, p_neighbors_raw = residual_layer #out_channel*in_channel  out_pn*max_neighbor_num
            
            if(in_channel != out_channel):
                in_pc_pad = torch.einsum('oi,bpi->bpo',[weight_res, in_pc_pad])


            out_pc_res = []
            if(in_pn == out_pn):
                out_pc_res = in_pc_pad[:,0:in_pn].clone()
            else:
                if(self.use_vanilla_pool == 0):
                    in_neighbors = in_pc_pad[:,neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*out_channel

                    p_neighbors = torch.abs(p_neighbors_raw)* neighbor_mask_lst
                    p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
                    p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)

                    out_pc_res = torch.einsum('pm,bpmo->bpo', [p_neighbors, in_neighbors])
                else:
                    in_neighbors = in_pc_pad[:,neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*out_channel
                    if(self.use_vanilla_pool==1): ## avg pool
                        out_pc_res = in_neighbors.mean(2)
                    elif(self.use_vanilla_pool==2): ##max pool
                        out_pc_res, index = in_neighbors.max(2)
                    else:
                        print ("!!!Warning Undefined Pool operation")                
            
        
        #print(out_pc_conv.shape, out_pc_res.shape)
        out_pc = out_pc_conv*np.sqrt(1-residual_rate) + out_pc_res*np.sqrt(residual_rate)
        
        
        return out_pc
    
    
     ##in_pc batch*point_num*3
    ##out_pc batch*point_num*3 
    def forward(self, in_pc): 
        
        #print("in_pc", in_pc.mean(1), in_pc.min(1), in_pc.max(1))
        
        out_pc = in_pc.clone()
        
        
        for i in range(self.layer_num):
            if(i<(self.layer_num-1)):
                out_pc = self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i])
                
            else:
                out_pc =  self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i], is_final_layer=True)
                
        #out_pc = self.final_linear(out_pc.transpose(1,2)).transpose(1,2) #batch*3*point_num
        
        return out_pc
    
    def forward_till_layer_n(self,in_pc,layer_n):
        out_pc = in_pc.clone()
        
        
        for i in range(layer_n):
            if(self.test_mode==False):
                out_pc = self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i])
            else:
                out_pc = self.forward_one_conv_layer_batch_during_test(out_pc, self.layer_lst[i])

        #out_pc = self.final_linear(out_pc.transpose(1,2)).transpose(1,2) #batch*3*point_num
        
        return out_pc
    
    def forward_from_layer_n(self, in_pc, layer_n):
        out_pc = in_pc.clone()
        
        
        for i in range(layer_n, self.layer_num):
            if(i<(self.layer_num-1)):
                if(self.test_mode==False):
                    out_pc = self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i])
                else:
                    out_pc = self.forward_one_conv_layer_batch_during_test(out_pc, self.layer_lst[i])
            else:
                if(self.test_mode==False):
                    out_pc =  self.forward_one_conv_layer_batch(out_pc, self.layer_lst[i], is_final_layer=True)
                else:
                    out_pc = self.forward_one_conv_layer_batch_during_test(out_pc,self.layer_lst[i], is_final_layer=True)


        #out_pc = self.final_linear(out_pc.transpose(1,2)).transpose(1,2) #batch*3*point_num
        
        return out_pc
    
            
    ##weights batch*point_num, weights.sum(1)==1
    def compute_geometric_mean_euclidean_dist_error(self, gt_pc, predict_pc,weights=[]):
        
        if(len(weights)==0):
            error = (gt_pc-predict_pc).pow(2).sum(2).pow(0.5).mean()
        
            return error

        else:
            batch =gt_pc.shape[0]
            point_num=gt_pc.shape[1]
            channel = gt_pc.shape[2]

            dists = (gt_pc-predict_pc).pow(2).sum(2).pow(0.5) * weights
            error = dists.sum()

            return error


    ##weights batch*point_num, weights.sum(1)==1
    def compute_geometric_loss_l1(self, gt_pc, predict_pc,weights=[]):
        
        if(len(weights)==0):
            loss = torch.abs(gt_pc-predict_pc).mean()
        
            return loss

        else:
            batch =gt_pc.shape[0]
            point_num=gt_pc.shape[1]
            pc_weights = weights.view(batch, point_num,1).repeat(1,1,3)

            loss = torch.abs(gt_pc*pc_weights - predict_pc*pc_weights).sum()/(batch*3)

            return loss
        
    
    
    #in_pc size batch*in_size*3
    #out_pc batch*in_size*3
    #neighbor_id_lstlst out_size*max_neighbor_num
    #neighbor_dist_lstlst out_size*max_neighbor_num
    def compute_laplace_loss_l1(self, gt_pc_raw, predict_pc_raw):
        gt_pc = gt_pc_raw*1
        predict_pc = predict_pc_raw*1
        
        batch = gt_pc.shape[0]
        
        gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
        predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)
        
        batch = gt_pc.shape[0]
        
        gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        gt_pc_laplace = gt_pc_laplace*(self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1,3)-1)
        
        for n in range(1, self.initial_max_neighbor_num):
            #print (neighbor_id_lstlst[:,n])
            neighbor = gt_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            gt_pc_laplace -= neighbor
        
        
        predict_pc_laplace = predict_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        predict_pc_laplace = predict_pc_laplace*(self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1,3)-1)
        
        for n in range(1, self.initial_max_neighbor_num):
            #print (neighbor_id_lstlst[:,n])
            neighbor = predict_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            predict_pc_laplace -= neighbor
        
            
        loss_l1 = torch.abs(gt_pc_laplace - predict_pc_laplace).mean()
        
        #gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
        #predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
        #loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()
        
        return loss_l1 #, loss_curv
    
    #in_pc size batch*in_size*3
    #out_pc batch*in_size*3
    #neighbor_id_lstlst out_size*max_neighbor_num
    #neighbor_dist_lstlst out_size*max_neighbor_num
    def compute_laplace_Mean_Euclidean_Error(self, gt_pc_raw, predict_pc_raw):
        gt_pc = gt_pc_raw*1
        predict_pc = predict_pc_raw*1
        
        batch = gt_pc.shape[0]
        
        gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
        predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)
        
        batch = gt_pc.shape[0]
        
        gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        gt_pc_laplace = gt_pc_laplace*(self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1,3)-1)
        
        for n in range(1, self.initial_max_neighbor_num):
            #print (neighbor_id_lstlst[:,n])
            neighbor = gt_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            gt_pc_laplace -= neighbor
        
        
        
        predict_pc_laplace = predict_pc[:, self.initial_neighbor_id_lstlst[:,0]]   ## batch*point_num*3 the first point is itself
        predict_pc_laplace = predict_pc_laplace*(self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1,3)-1)
        
        for n in range(1, self.initial_max_neighbor_num):
            #print (neighbor_id_lstlst[:,n])
            neighbor = predict_pc[:,self.initial_neighbor_id_lstlst[:,n]]
            predict_pc_laplace -= neighbor
        
            
        error  = torch.pow(torch.pow(gt_pc_laplace - predict_pc_laplace,2).sum(2), 0.5).mean()
        
        #gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
        #predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
        #loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()
        
        return error #, loss_curv
        
        #use in train mode
    def compute_param_num(self, ):
        total_param=0
        for i in range(self.layer_num):
            in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros=self.layer_lst[i]
            
            if(len(conv_layer)!=0 ):
                param_num=0
                if(self.conv_method!="vw"):
                    conv = conv_layer[-1]
                    for p in conv.parameters():
                        if p.requires_grad:
                            p_dim = 1
                            for i in range(len(p.shape)):
                                p_dim = p_dim*p.shape[i]
                            param_num += p_dim
                else:
                    param_num = out_pn + neighbor_mask_lst.sum()*in_channel*out_channel
                total_param+=param_num
                print ("Conv Layer",i, "param num:", param_num)
                


            if(len(residual_layer)!=0):
                param_num=0
                weight_res, p_neighbors_raw = residual_layer #out_channel*in_channel  out_pn*max_neighbor_num

                if(len(weight_res)>0):
                    param_num = np.array(list(weight_res.shape)).sum()
                    total_param += param_num
                if(len(p_neighbors_raw)>0):
                    param_num =  np.array(list(p_neighbors_raw.shape)).sum()
                    total_param += param_num
                print ("Pool Layer", i, "param num:", param_num)

        
            
        print ("Total network param num:", total_param)    
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
