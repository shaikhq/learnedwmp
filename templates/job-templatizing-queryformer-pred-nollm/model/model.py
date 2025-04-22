import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out

        
class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, tables = 10, types=20, joins = 40, columns= 30, \
                 ops=4, use_sample = True, use_hist = True, bin_number = 50):
        super(FeatureEmbed, self).__init__()
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(bin_number, embed_size)
        
        self.linearFilterFactors = nn.Linear(3, embed_size)  # Map 3 floats to embed_size

        self.joinEmbed = nn.Embedding(joins, embed_size)
        
        # self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size*5 + embed_size//8+1)
        # SQ: the input dimension needs to match the output dimension from the forward function
        # currently it's 1408
        # self.project = nn.Linear(1408, embed_size*4 + 1152 + embed_size//8+1)
        # changing the first dimension of the following layer to be 256, to match the final dim of the node feat vector
        self.project = nn.Linear(256, embed_size*4 + 256 + embed_size//8+1)
       
    
    # input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)
    def forward(self, feature):
        typeId, joinId, tableId, filtersMask, filterEmbed, filterFactors = torch.split(feature, (1, 1, 1, 3, 3 * 384, 3), dim=-1)
        
        # Process existing features
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        tableEmb = self.getTable(tableId)
        
        # Process filterFactors
        filterFactorsEmb = F.relu(self.linearFilterFactors(filterFactors))  # Transform filterFactors
        
        # Concatenate all features
        # final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, filterFactorsEmb), dim=1)
        final = torch.cat((typeEmb, joinEmb, tableEmb, filterFactorsEmb), dim=1)
        
        print(f'final.shape: {final.shape}')
        # final shape: 256
        
        # Apply projection
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)
    
    def getTable(self, tableId):
        emb = self.tableEmbed(tableId.long()).squeeze(1)
        
        return emb.squeeze(1)
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # batch * 50 * 3
        histExpand = hists.view(-1,self.bin_number,3).transpose(1,2)
        
        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.  # mask out space holder
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(emb, dim = 1)
        # Add a small constant to avoid division by zero
        avg = total / (num_filters.view(-1, 1) + 1e-6)
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        colsId = filterExpand[:,:,0].long()
        opsId = filterExpand[:,:,1].long()
        vals = filterExpand[:,:,2].unsqueeze(-1) # b by 3 by 1
        
        # b by 3 by embed_dim
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op, vals), dim = -1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        ## apply mask
        concat[~filtersMask.bool()] = 0.
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(concat, dim = 1)
        avg = total / num_filters.view(-1,1)
                
        return avg
    
#     def get_output_size(self):
#         size = self.embed_size * 5 + self.embed_size // 8 + 1
#         return size



class QueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormer,self).__init__()
        
        # In the following the number 5 or 4 needs to match the number of featuers that have embeddings:
        # if use_hist:
        #     hidden_dim = emb_size * 6 + emb_size //8 + 1
        # else:
        #     hidden_dim = emb_size * 5 + emb_size //8 + 1
        # Changed the following to match the output dimension from feature embed dim plus embed_size //8 + 1
        # hidden_dim = emb_size * 4 + 1152 + emb_size //8 + 1
        hidden_dim = emb_size*4 + 256 + emb_size//8+1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        
    def forward(self, batched_data, return_embedding=False):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        heights = batched_data.heights
        
        # print(f'x.shape: {x.shape}')
        
        #print(f"At QueryFormer Forward - Attention Bias: {attn_bias}")
        # print(f"Relative Positions: {rel_pos}")
        # print(f"x: {x}")
        # print(f"heights: {heights}")
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # Relative position bias
        # print()
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # Reset relative positions
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        x_view = x.view(-1, 1161)
        #print('At QueryFormer Forward: x_view', x_view)
        # print('At QueryFormer Forward: x_view shape', x_view.shape)

        # node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)
        # First, process x_view through embbed_layer
        processed_features = self.embbed_layer(x_view)
        # print(f'processed_features: {processed_features.shape}')

        # Then reshape the result back to the batch structure
        node_feature = processed_features.view(n_batch, -1, self.hidden_dim)
        
        # print('QueryFormer Forward - node_feature: ', node_feature)

        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # Transformer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers[:-1]:  # Pass through all but the last encoder layer
            output = enc_layer(output, tree_attn_bias)

        second_to_last_embedding = output  # Save the output before the final encoder layer

        # Pass through the final encoder layer
        output = self.layers[-1](output, tree_attn_bias)
        output = self.final_ln(output)

        # Prediction
        # Generate predictions
        pred_output = self.pred(output[:, 0, :])  # Shape: [batch_size, 1]
        #pred_output = pred_output.squeeze(-1)  # Shape: [batch_size]
        
        # print(f"pred_output shape: {pred_output.shape}")
        
        return pred_output, second_to_last_embedding
       

    

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x























