import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import ipdb
import re
import numpy as np

from .layers import LoRALayer 
from typing import Optional, List 




def generate_un_from_vn(v_n):
    # Compute the sign of the first component of v_n
    sign_vn1 = torch.sign(v_n[0])
    
    # Compute the norm of v_n
    norm_vn = v_n.norm()
    
    # Generate the unit vector e_1
    e_1 = torch.zeros_like(v_n)
    e_1[0] = 1.0
    
    # Compute u_n
    u_n = v_n + sign_vn1 * norm_vn * e_1
    return u_n


def sample_unit_sphere(dim):
    vec = torch.randn(dim)
    vec /= vec.norm()
    if vec[0]>0: vec = -vec
    return vec

def lower_triangular_unit_spheres(n, r):
    assert r <= n, "r should be less than or equal to n"
    matrix = torch.zeros(n, r)
    for i in range(r):
        vec = generate_un_from_vn(sample_unit_sphere(n - i))
        matrix[i:, i] = r * vec
    return matrix # M_u INIT



class trueSVDLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        normalized: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.eyer = torch.eye(self.r, requires_grad = False).to("cuda")
        self.eyeU = torch.eye(out_features, r, requires_grad=False).to("cuda")
        self.eyeV = torch.eye(in_features,r,requires_grad = False).to("cuda")
        # Actual trainable parameters
        if r > 0:
            self.lora_V = nn.Parameter(self.weight.new_zeros((in_features, r)))
            self.lora_S = nn.Parameter(self.weight.new_zeros(r))
            self.lora_U = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        #print("--------------1-11----------------")
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_U'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            #nn.init.uniform_(self.lora_V, mean=0.0, std=0.02)
            self.lora_V = nn.Parameter(lower_triangular_unit_spheres(self.in_features, self.r))
            nn.init.zeros_(self.lora_S)
            self.lora_U = nn.Parameter(lower_triangular_unit_spheres(self.out_features, self.r))
            #self.lora_U = nn.Parameter(self.lora_U * torch.tril(torch.ones_like(self.lora_U)))
            #self.lora_V = nn.Parameter(self.lora_V * torch.tril(torch.ones_like(self.lora_U)))


    def get_U(self):
        #self.lora_U *= torch.tril(torch.ones_like(self.lora_U))
        self.lora_U.data = F.normalize(self.lora_U,p = 2,dim = 0)
        UTU = torch.matmul(self.lora_U.T,self.lora_U)
        K_u = 0.5*self.eyer+torch.triu(UTU, diagonal = 1)
        K_u=K_u.to("cuda")
        #print(self.lora_U)
        #print(self.lora_S)
        result= self.eyeU-self.lora_U.to("cuda") @ torch.inverse(K_u.cpu()).to("cuda") @ self.lora_U.T[:,:self.r].to("cuda")
        del UTU, K_u
        torch.cuda.empty_cache()
        return result

    def get_V(self):
        #lower_V = self.lora_V# * torch.tril(torch.ones_like(self.lora_V))
        self.lora_V.data = F.normalize(self.lora_V,p = 2,dim = 0)
        VTV = torch.matmul(self.lora_V.T,self.lora_V)
        K_u = 0.5*self.eyer.to("cuda")+torch.triu(VTV, diagonal = 1).to("cuda")
        #print(self.lora_U)
        #print(self.lora_S)
        result= self.eyeV-self.lora_V.to("cuda") @ torch.inverse(K_u.cpu()).to("cuda") @ self.lora_V.T[:,:self.r].to("cuda")
        #print(result)
        del VTV, K_u
        torch.cuda.empty_cache()
        return result
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data = self.weight.data.to("cuda")
                    self.weight.data -= T(self.get_U()[:,:self.r].to("cuda") @ torch.diag(self.lora_S).to("cuda")@self.get_V()[:,:self.r].T.to("cuda")).to("cuda") * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data = self.weight.data.to("cuda")
                    self.weight.data +=T(self.get_U()[:,:self.r].to("cuda") @ torch.diag(self.lora_S).to("cuda")@self.get_V()[:,:self.r].T.to("cuda")).to("cuda") * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.get_V()[:,:self.r]@torch.diag(self.lora_S) @ self.get_U()[:,:self.r].T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)






# exit()


# class Linear(nn.Linear, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         merge_weights: bool = True,
#         normalized: bool = False,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)

#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_V = nn.Parameter(self.weight.new_zeros((in_features, r)))
#             self.lora_S = nn.Parameter(self.weight.new_zeros(r))
#             self.lora_U = nn.Parameter(self.weight.new_zeros((out_features, r)))
#             self.scaling = self.lora_alpha / self.r
#             self.maskU = torch.tril(torch.ones_like(self.lora_U))
#             #self.maskU.requires_grad_(False)
#             self.maskV = torch.tril(torch.ones_like(self.lora_V))
#             #self.maskV.requires_grad_(False)
#             self.half_eyeR = 0.5*torch.eye(self.r)
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)

#     def reset_parameters(self):
#         #print("--------------1-11----------------")
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_U'):
#             # initialize B the same way as the default for nn.Linear and A to zero
#             # this is different than what is described in the paper but should not affect performance
#             #nn.init.uniform_(self.lora_V, mean=0.0, std=0.02)
#             self.lora_V = nn.Parameter(lower_triangular_unit_spheres(self.in_features, self.r))
#             nn.init.zeros_(self.lora_S)
#             self.lora_U = nn.Parameter(lower_triangular_unit_spheres(self.out_features, self.r))
#             #self.lora_U = nn.Parameter(self.lora_U * torch.tril(torch.ones_like(self.lora_U)))
#             #self.lora_V = nn.Parameter(self.lora_V * torch.tril(torch.ones_like(self.lora_U)))


#     def get_U(self):
#         lower_U = self.lora_U.to("cuda") * self.maskU.to("cuda")
#         lower_U = F.normalize(lower_U,p = 2,dim = 0)
#         UUT = torch.matmul(lower_U.T,lower_U)
#         K_u = self.half_eyeR.to("cuda")+torch.triu(UUT, diagonal = 1).to("cuda")
#         #print(self.lora_U)
#         #print(self.lora_S)
#         return torch.eye(self.out_features,device = self.lora_U.device)[:,:self.r].to("cuda")-lower_U.to("cuda") @ torch.inverse(K_u.cpu()).to("cuda") @ lower_U[:self.r,:].T.to("cuda")

#     def get_V(self):
#         lower_V = self.lora_V.to("cuda") * self.maskV.to("cuda")
#         lower_V = F.normalize(lower_V,p = 2,dim = 0)
#         VVT = torch.matmul(lower_V.T,lower_V)
#         K_u = self.half_eyeR.to("cuda")+torch.triu(VVT, diagonal = 1).to("cuda")
#         #print(self.lora_U)
#         #print(self.lora_S)
#         result= torch.eye(self.out_features,device = self.lora_V.device)[:,:self.r].to("cuda")-lower_V.to("cuda") @ torch.inverse(K_u.cpu()).to("cuda") @ lower_V[:self.r,:].T.to("cuda")
#         #print(result)
#         return result
#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data = self.weight.data.to("cuda")
#                     self.weight.data -= T(self.get_U().to("cuda") @ torch.diag(self.lora_S).to("cuda")@self.get_V().T.to("cuda")).to("cuda") * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data = self.weight.data.to("cuda")
#                     self.weight.data +=T(self.get_U().to("cuda") @ torch.diag(self.lora_S).to("cuda")@self.get_V().T.to("cuda")).to("cuda") * self.scaling
#                 self.merged = True       

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)            
#             result += (self.lora_dropout(x) @ self.get_V()@torch.diag(self.lora_S) @ self.get_U().T) * self.scaling
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)







# class SVDLinear(nn.Linear, LoRALayer):
#     # SVD-based adaptation implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         fan_in_fan_out: bool = False, 
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)

#         self.module_name = ""
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.ParameterList([nn.Parameter(
#                 self.weight.new_zeros((r, in_features))
#             )])
#             self.lora_E = nn.ParameterList([nn.Parameter(
#                 self.weight.new_zeros(r, 1)
#             )])
#             self.lora_B = nn.ParameterList([nn.Parameter(
#                 self.weight.new_zeros((out_features, r))
#             )])
#             self.W = loraW()
#             self.hook_handle = self.W.register_full_backward_hook(self.backward_hook)
            
#             self.score = 0
#             self.gradMatrix_trace = 0
#             self.ranknum = nn.Parameter(
#                 self.weight.new_zeros(1), requires_grad=False
#             )
#             self.ranknum.data.fill_(float(self.r))
#             self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#             self.ranknum.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T

#     def backward_hook(self, module, grad_input, grad_output):
#         # print("Output_Grad:", grad_output)
#         grad_Matrix = grad_output[0]
#         try:
#             W = (
                
#                  self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
#                  ).abs()
#             # scale_W = torch.mean(W)
#             scale_W=1
#             self.score = torch.sum(((W / scale_W) * grad_Matrix).abs().detach()) / math.sqrt(W.numel())
#             # self.score = torch.mean((grad_Matrix ** 2).detach())
#         except:
#             ipdb.set_trace()
        
#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A,B the same way as the default for nn.Linear 
#             # and E (singular values) for zero 
#             nn.init.zeros_(self.lora_E[0])
#             nn.init.normal_(self.lora_A[0], mean=0.0, std=0.02)
#             nn.init.normal_(self.lora_B[0], mean=0.0, std=0.02)

#     def add_reserve_param(self, add_r, advance_learn=True):
#         for _ in range(add_r):
#             e = nn.Parameter(self.weight.new_zeros(1, 1), requires_grad=False)
#             a = nn.Parameter(self.weight.new_zeros((1, self.in_features)), requires_grad=advance_learn)
#             b = nn.Parameter(self.weight.new_zeros((self.out_features, 1)), requires_grad=advance_learn)
#             e[0][0] = 1e-5 if advance_learn else 0.
#             nn.init.normal_(a, mean=0.0, std=0.02)
#             nn.init.normal_(b, mean=0.0, std=0.02)
#             self.lora_E.append(e)
#             self.lora_A.append(a)
#             self.lora_B.append(b)
    
#     def train(self, mode: bool = True):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode == True:
#             self.lora_A.requires_grad = True
#             self.lora_E.requires_grad = True
#             self.lora_B.requires_grad = True
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= T(
#                         self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
#                     )
#                 self.merged = False
#         else:
#             self.lora_A.requires_grad = False
#             self.lora_E.requires_grad = False
#             self.lora_B.requires_grad = False
    
#     def eval(self):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         nn.Linear.eval(self)
#         if self.merge_weights and not self.merged:
#             # Merge the weights and mark it
#             if self.r > 0:
#                 self.weight.data += T(
#                     self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
#                 )
#             self.merged = True

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 try:
#                     result += (
#                         self.lora_dropout(x) @ self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum).T
#                     )
#                 except:
#                     ipdb.set_trace()
#                     print(self.W)
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)


# class SVDLinear(nn.Linear, LoRALayer):
#     # SVD-based adaptation implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         fan_in_fan_out: bool = False, 
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)

#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(
#                 self.weight.new_zeros((r, in_features))
#             )
#             self.lora_E = nn.Parameter(
#                 self.weight.new_zeros(r, 1)
#             ) 
#             self.lora_B = nn.Parameter(
#                 self.weight.new_zeros((out_features, r))
#             )
#             self.ranknum = nn.Parameter(
#                 self.weight.new_zeros(1), requires_grad=False
#             )
#             self.ranknum.data.fill_(float(self.r))
#             self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#             self.ranknum.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A,B the same way as the default for nn.Linear 
#             # and E (singular values) for zero 
#             nn.init.zeros_(self.lora_E)
#             nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
#             nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

#     def train(self, mode: bool = True):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if self.merge_weights and self.merged:
#             # Make sure that the weights are not merged
#             if self.r > 0:
#                 self.weight.data -= T(
#                     self.lora_B @ (self.lora_A*self.lora_E)
#                 ) * self.scaling / (self.ranknum+1e-5)
#             self.merged = False
    
#     def eval(self):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         nn.Linear.eval(self)
#         if self.merge_weights and not self.merged:
#             # Merge the weights and mark it
#             if self.r > 0:
#                 self.weight.data += T(
#                     self.lora_B @ (self.lora_A * self.lora_E)
#                 ) * self.scaling / (self.ranknum+1e-5)
#             self.merged = True

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 result += (
#                     self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
#                 ) * self.scaling / (self.ranknum+1e-5)
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)