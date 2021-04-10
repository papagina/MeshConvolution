# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, param, test_mode = False): #layer_info_lst= [(point_num, feature_dim)]
        super(Model, self).__init__()
        
        self.point_num = param.point_num

        self.test_mode = test_mode
        
        
        self.perpoint_bias = param.perpoint_bias
        
        self.channel_lst = param.channel_lst

        self.residual_rate_lst = param.residual_rate_lst
        
        #self.channel_lst = [3]*14
        
        self.weight_num_lst = param.weight_num_lst
        
        self.connection_layer_fn_lst = param.connection_layer_fn_lst
        
        self.initial_connection_fn = param.initial_connection_fn
        

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
            neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst).cuda()
            max_neighbor_num = neighbor_id_lstlst.shape[1]
            avg_neighbor_num = round(neighbor_num_lst.mean().item())
            effective_w_weights_rate = (neighbor_num_lst.sum()/ float(max_neighbor_num * out_point_num))
            effective_w_weights_rate = round(effective_w_weights_rate.item(),3)

            pc_mask  = torch.ones(in_point_num+1).cuda()
            pc_mask[in_point_num]=0
            neighbor_mask_lst = pc_mask[neighbor_id_lstlst].contiguous() #out_pn*max_neighbor_num neighbor is 1 otherwise 0
            
            zeros_batch_outpn_outchannel = torch.zeros((batch, out_point_num, out_channel)).cuda()

            if((residual_rate<0) or (residual_rate>1)):
                print ("Invalid residual rate", residual_rate)
            ####parameters for conv###############
            conv_layer = ""

            if(residual_rate <1):
                weights = torch.randn(weight_num, out_channel*in_channel).cuda()
                
                weights = nn.Parameter(weights).cuda()
                
                self.register_parameter("weights"+str(l),weights)
                
    
                bias = nn.Parameter(torch.zeros(out_channel).cuda())
                if(self.perpoint_bias == 1):
                    bias= nn.Parameter(torch.zeros(out_point_num, out_channel).cuda())
                self.register_parameter("bias"+str(l),bias)
                

                w_weights=torch.randn(out_point_num, max_neighbor_num, weight_num)/(avg_neighbor_num*weight_num)

                w_weights = nn.Parameter(w_weights.cuda())
                self.register_parameter("w_weights"+str(l),w_weights)
                
                
                conv_layer = (weights, bias, w_weights)
            
            ####parameters for residual###############

            ## a residual layer with out_point_num==in_point_num and residual_rate==1 is a pooling or unpooling layer
            
            residual_layer = ""

            if(residual_rate>0):
                p_neighbors = ""
                weight_res = ""

                if(out_point_num != in_point_num) : 
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
                   "max_neighbor_num", max_neighbor_num, "avg_neighbor_num", avg_neighbor_num, "effective_w_weights_rate", effective_w_weights_rate )
            
            in_point_num=out_point_num
            in_channel = out_channel
         
         
    #precompute the parameters so as to accelerate forwarding in testing mode
    def init_test_mode(self):
        for l in range(len(self.layer_lst)):
            layer_info = self.layer_lst[l]
        
            
            in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros_batch_outpn_outchannel=layer_info
            

            if(len(conv_layer) != 0):

                (weights, bias, raw_w_weights) = conv_layer #weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
                
                w_weights=""
                
                w_weights = raw_w_weights*neighbor_mask_lst.view(out_pn, max_neighbor_num, 1).repeat(1,1,weight_num) #out_pn*max_neighbor_num*weight_num

                

                weights = torch.einsum('pmw,wc->pmc',[w_weights, weights]) #out_pn*max_neighbor_num*(out_channel*in_channel)
                weights = weights.view(out_pn, max_neighbor_num, out_channel,in_channel)
                
                conv_layer = weights, bias
            

            ####compute output of residual layer####

            if(len(residual_layer)!=0):
                weight_res, p_neighbors_raw = residual_layer #out_channel*in_channel  out_pn*max_neighbor_num
                if(in_pn != out_pn):
                    p_neighbors = torch.abs(p_neighbors_raw)* neighbor_mask_lst
                    p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
                    p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)

                    residual_layer = weight_res, p_neighbors
            
            self.layer_lst[l] = in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros_batch_outpn_outchannel

    def analyze_gradients(self):
        weight_basis_gradient = []   
        weight_basis = []
        w_weight_gradient = []
        w_weight = []
        for name, param in self.named_parameters():
            if (len(name)>=7): 
                if( name[0:7] == "weights"):
                    weight_basis_gradient += [param.grad.data.abs().mean().item()]
                    weight_basis += [param.data.abs().mean().item()]
                elif(name[0:9] == "w_weights"):
                    w_weight_gradient += [param.grad.data.abs().mean().item()]
                    w_weight += [param.data.abs().mean().item()]

        print("basis gradient:", np.array(weight_basis_gradient).mean(), np.array(weight_basis).mean() )
        print("coeff gradient:", np.array(w_weight_gradient).mean(), np.array(w_weight).mean())


    #a faster mode for testing
    #input_pc batch*in_pn*in_channel
    #out_pc batch*out_pn*out_channel
    def forward_one_conv_layer_batch_during_test(self, in_pc, layer_info, is_final_layer=False):

        batch = in_pc.shape[0]
        
        in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros_batch_outpn_outchannel=layer_info
        
        
        in_pc_pad = torch.cat((in_pc, torch.zeros(batch, 1, in_channel).cuda()), 1) #batch*(in_pn+1)*in_channel

        in_neighbors = in_pc_pad[:, neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*in_channel

        ####compute output of convolution layer####
        out_pc_conv = zeros_batch_outpn_outchannel.clone()

        if(len(conv_layer) != 0):

            (weights, bias) = conv_layer #weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
                
            
            out_neighbors = torch.einsum('pmoi,bpmi->bpmo',[weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel
            
            out_pc_conv = out_neighbors.sum(2)
            
            out_pc_conv = out_pc_conv + bias
            
            if(is_final_layer == False):
                out_pc_conv = self.relu(out_pc_conv) ##self.relu is defined in the init function
        
        #if(self.residual_rate==0):
        #    return out_pc
        ####compute output of residual layer####
        out_pc_res = zeros_batch_outpn_outchannel.clone()

        if(len(residual_layer)!=0):
            weight_res, p_neighbors = residual_layer #out_channel*in_channel  out_pn*max_neighbor_num
            
            if(in_channel != out_channel):
                in_pc_pad = torch.einsum('oi,bpi->bpo',[weight_res, in_pc_pad])


            out_pc_res = []
            if(in_pn == out_pn):
                out_pc_res = in_pc_pad[:,0:in_pn].clone()
            else:
                in_neighbors = in_pc_pad[:,neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*out_channel
                out_pc_res = torch.einsum('pm,bpmo->bpo', [p_neighbors, in_neighbors])              
        
        out_pc = out_pc_conv*np.sqrt(1-residual_rate) + out_pc_res*np.sqrt(residual_rate)
        
        return out_pc
            
    #use in train mode. Slower than test mode
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

            (weights, bias, raw_w_weights) = conv_layer #weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
            
            w_weights = raw_w_weights*neighbor_mask_lst.view(out_pn, max_neighbor_num, 1).repeat(1,1,weight_num) #out_pn*max_neighbor_num*weight_num


            weights = torch.einsum('pmw,wc->pmc',[w_weights, weights]) #out_pn*max_neighbor_num*(out_channel*in_channel)
            weights = weights.view(out_pn, max_neighbor_num, out_channel,in_channel)
            
            out_neighbors = torch.einsum('pmoi,bpmi->bpmo',[weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel
            
            out_pc_conv = out_neighbors.sum(2)
            
            out_pc_conv = out_pc_conv + bias
            
            if(is_final_layer == False):
                out_pc_conv = self.relu(out_pc_conv) ##self.relu is defined in the init function
        
        
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
                in_neighbors = in_pc_pad[:,neighbor_id_lstlst] #batch*out_pn*max_neighbor_num*out_channel

                p_neighbors = torch.abs(p_neighbors_raw)* neighbor_mask_lst
                p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
                p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)

                out_pc_res = torch.einsum('pm,bpmo->bpo', [p_neighbors, in_neighbors])             
            
        
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
        

    #parameters for training
    def compute_param_num(self, ):
        non_zero_w_weights_num_sum=0
        total_param=0
        for i in range(self.layer_num):
            in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, conv_layer, residual_layer, residual_rate, neighbor_mask_lst, zeros=self.layer_lst[i]
            if(len(conv_layer)!=0 ):
                param_num=0
                (raw_weights, bias, raw_w_weights) = conv_layer

                w_weights =[]

                w_weights_num =  neighbor_mask_lst.sum()*weight_num
                w_weights_num = w_weights_num.item()
                    
                param_num += w_weights_num 

                param_num +=np.array(list(raw_weights.shape)).sum()
                param_num +=np.array(list(bias.shape)).sum()
                
                print ("Layer",i, "param num:", param_num)
                total_param+=param_num


            if(len(residual_layer)!=0):
                weight_res, p_neighbors_raw = residual_layer #out_channel*in_channel  out_pn*max_neighbor_num
                if(len(weight_res)>0):
                    total_param += np.array(list(weight_res.shape)).sum()
                if(len(p_neighbors_raw)>0):
                    total_param += np.array(list(p_neighbors_raw.shape)).sum()

        
            
        print ("Total network param num:", total_param)   
   
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
