import math
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
from torch import nn as nn 
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch




class ZERON_GCN(Module):
	def __init__(self, in_features, out_features, bias=True):
		super(ZERON_GCN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))


		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 6. / math.sqrt(self.weight.size(1) + self.weight.size(0))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-0, 0)

	def forward(self, input, adj, activation):
		
		support = torch.mm(input, self.weight)
		output = torch.cat((torch.mm(adj, support[:, :support.shape[1]//10]), support[:, support.shape[1]//10:]), dim = 1)
		
		if self.bias is not None:
			output = output + self.bias
		return activation(output)

class GCNMax(Module):
	def __init__(self, in_features, print_length):
		super(GCNMax, self).__init__()
		self.in_features = in_features
		self.print_length = print_length
		self.weight_Ws = nn.ParameterList(Parameter(torch.Tensor(in_features, print_length)) for i in range(1))
		self.weight_Bs = nn.ParameterList(Parameter(torch.Tensor(print_length)) for i in range(1))
		self.reset_parameters()

	def reset_parameters(self):
		for i in range(1):
			stdv = 6. / math.sqrt(self.weight_Bs[i].size(0))

			self.weight_Bs[i].data.uniform_(-stdv, stdv)
			stdv = 6. / math.sqrt(self.weight_Ws[i].size(0) + self.weight_Ws[i].size(1))
			self.weight_Ws[i].data.uniform_(-stdv, stdv)
			

	def forward(self, r_s, adj, activation):

		
		bias = self.weight_Bs[0]
		weight_W = self.weight_Ws[0]
		


		v_s = torch.mm(r_s, weight_W)  ## 10
		v_s = torch.cat((torch.mm(adj, v_s[:, :v_s.shape[1]//10]), v_s[:, v_s.shape[1]//10:]), dim = 1)


		v_s = v_s + bias
		

		i_s = activation(v_s)       ## 10
	
		f   = torch.max(i_s, dim = 0)[0]       ## 11
		return f                            ## 12                     

