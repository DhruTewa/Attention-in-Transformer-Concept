import torch 
import torch.nn as nn # for modules and linear classes
import torch.nn.functional as F # for Softmax activation function

class SelfAttention(nn.Module):
    def __init__(self,d_model =2, row_dim = 0, col_dim = 1):
        super().__init__()
        self.W_q = nn.Linear(in_features = d_model, out_features = d_model,bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model,bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model,bias = False)
        self.row_dim = row_dim
        self.col_dim = col_dim
        
    def forward(self,token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        
        sims = torch.matmul(q,k.transpose(dim0 = self.row_dim,dim1 = self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        attention_percents = F.softmax(scaled_sims,dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents,v)
        return attention_scores
    
    
# Testing the SelfAttention class
encoding_matrix = torch.tensor([[1.16,20.23],
                                [0.57,1.36],
                                [4.41,-2.16]]
                               )
torch.manual_seed(42)

self_attention = SelfAttention(d_model = 2,row_dim=0,col_dim=1)
self_attention(encoding_matrix)

        
        
        