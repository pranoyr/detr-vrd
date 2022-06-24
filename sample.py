import torch.nn as nn
import torch

a = torch.Tensor(6 ,2, 300, 512)
print(a.shape)
sbj = a[:,:,::3]
prd = a[:,:,1::3]
obj = a[:,:, 2::3]
print(obj.shape)


query_embed1 = nn.Parameter(torch.zeros(100, 3, 256))
a = nn.Embedding(100, 256, 3)


# multihead_attn = nn.MultiheadAttention(256, 8)

# query = torch.Tensor(10,2,256)
# key = torch.Tensor(10,2,256)
# value = torch.Tensor(10,2,256)


# attn_output, attn_output_weights = multihead_attn(query, key, value)


# print(attn_output.shape)

# create a tensor in pytorch
