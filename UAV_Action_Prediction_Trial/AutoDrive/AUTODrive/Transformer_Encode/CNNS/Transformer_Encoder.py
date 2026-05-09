############################################################################################################################
import numpy
import torch
import torch.nn as nn


class PadSubMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.Pad_mask0 = False
        self.Sub_mask0 = False

    def Pad_mask(self, Order_Q, Order_K):
        self.Pad_mask0 = True

        Batch_Size, Len_Q, _ = Order_Q.size()
        Batch_Size, Len_K, _ = Order_K.size()
        is_Pad = Order_K.sum(dim=-1).eq(0)
        Pad_Attn_mask = is_Pad.unsqueeze(1)

        return Pad_Attn_mask.expand(Batch_Size, Len_Q, Len_K)


class ScaledAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ScaledAttention0 = False

    def forward(self, Order_Q, Order_K, dim_K, Order_V, Attn_mask):
        self.ScaledAttention0 = True

        Scores = torch.matmul(Order_Q, Order_K.transpose(-1, -2)) / numpy.sqrt(dim_K)
        Scores.masked_fill_(Attn_mask, -1e9)

        Attn = nn.Softmax(dim=-1)(Scores)
        Context = torch.matmul(Attn, Order_V)

        return Context, Attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model=512, dim_heads=8, dim_Q=64, dim_K=64, dim_V=64, dropout=0.1):
        super().__init__()

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.dim_model = dim_model
        self.dim_heads = dim_heads
        self.dropout = nn.Dropout(dropout)
        self.Layer_Norm = nn.LayerNorm(dim_model)
        self.scaled_attention = ScaledAttention()
        self.Tensor_Q = nn.Linear(self.dim_model, self.dim_Q * self.dim_heads, bias=False)
        self.Tensor_K = nn.Linear(self.dim_model, self.dim_K * self.dim_heads, bias=False)
        self.Tensor_V = nn.Linear(self.dim_model, self.dim_V * self.dim_heads, bias=False)
        self.Tensor_Fc = nn.Linear(self.dim_heads * self.dim_V, self.dim_model, bias=False)

    def forward(self, Order_Q, Order_K, Order_V, Attn_mask):
        Residual, Batch_Size = Order_Q, Order_Q.size(0)
        Q = self.Tensor_Q(Order_Q).view(Batch_Size, -1, self.dim_heads, self.dim_Q).transpose(1, 2)
        K = self.Tensor_K(Order_K).view(Batch_Size, -1, self.dim_heads, self.dim_K).transpose(1, 2)
        V = self.Tensor_V(Order_V).view(Batch_Size, -1, self.dim_heads, self.dim_V).transpose(1, 2)

        Attention_mask = Attn_mask.unsqueeze(1).repeat(1, self.dim_heads, 1, 1)

        Context, Attn = self.scaled_attention(Q, K, self.dim_K, V, Attention_mask)
        Context = Context.transpose(1, 2).reshape(Batch_Size, -1, self.dim_heads * self.dim_V)
        Softmax_OUT = self.Tensor_Fc(Context)
        Softmax_OUT = self.dropout(Softmax_OUT)

        return self.Layer_Norm(Softmax_OUT + Residual), Attn


class FeedForward(nn.Module):
    def __init__(self, Feed_dim=2048, dim_models=512, dropout=0.1):
        super().__init__()

        self.Feed_dim = Feed_dim
        self.dim_model = dim_models
        self.dropout = nn.Dropout(dropout)
        self.Layer_Norm = nn.LayerNorm(dim_models)
        self.Sequential = nn.Sequential(nn.Linear(self.dim_model, self.Feed_dim, bias=False),
                                        nn.Dropout(dropout),
                                        nn.Tanh(),
                                        nn.Linear(self.Feed_dim, self.dim_model, bias=False))

    def forward(self, value):
        Residual = value
        Sequential_layer = self.Sequential(value)
        Sequential_layer = self.dropout(Sequential_layer)

        return self.Layer_Norm(Sequential_layer + Residual)


class EncoderLinear(nn.Module):
    def __init__(self, Feed_dim, dim_models, dim_heads, dim_Q, dim_K, dim_V):
        super().__init__()

        self.MultiHeadAttention_Enc_Self = MultiHeadAttention(dim_models, dim_heads, dim_Q, dim_K, dim_V)
        self.FeedForward = FeedForward(Feed_dim, dim_models)

    def forward(self, Enc_IN, Enc_Self_Attn_Mask):
        Enc_OUT, Enc_Self_Attn = self.MultiHeadAttention_Enc_Self(Enc_IN, Enc_IN, Enc_IN, Enc_Self_Attn_Mask)
        Enc_OUT = self.FeedForward(Enc_OUT)

        return Enc_OUT, Enc_Self_Attn


class Encoder(nn.Module):
    def __init__(self, Enc_num_Linear0, dim_models, dim_heads, Feed_dim, dim_Q, dim_K, dim_V):
        super().__init__()

        self.Encoder_Linear0 = nn.ModuleList(
            [EncoderLinear(Feed_dim, dim_models, dim_heads, dim_Q, dim_K, dim_V) for _ in range(Enc_num_Linear0)])
        self.PubSubMask = PadSubMask()

    def forward(self, Enc_IN):

        Enc_Self_Attn_Mask = self.PubSubMask.Pad_mask(Enc_IN, Enc_IN)
        Enc_Self_Attn_S = []
        Enc_OUT = Enc_IN
        for Linear0 in self.Encoder_Linear0:
            Enc_OUT, Enc_Self_Attn = Linear0(Enc_OUT, Enc_Self_Attn_Mask)
            Enc_Self_Attn_S.append(Enc_Self_Attn)

        return Enc_OUT, Enc_Self_Attn_S


class EncoderWithOutHeads(nn.Module):
    def __init__(self, Enc_num_Linear0, dim_models, dim_heads, Feed_dim, dim_Q, dim_K, dim_V, Batch_size, Len):
        super().__init__()

        # 原始编码器
        self.encoder = Encoder(Enc_num_Linear0, dim_models, dim_heads, Feed_dim, dim_Q, dim_K, dim_V)
        B_size = Batch_size*Len
        # 全连接控制层
        self.fcA0 = nn.Linear(dim_models, 1)
        self.fcB0 = nn.Linear(B_size, 1)
        self.fcA1 = nn.Linear(dim_models, 1)
        self.fcB1 = nn.Linear(B_size, 1)
        self.fcA2 = nn.Linear(dim_models, 1)
        self.fcB2 = nn.Linear(B_size, 1)
        self.fcA3 = nn.Linear(dim_models, 1)
        self.fcB3 = nn.Linear(B_size, 1)
        # 全连接备用层
        self.fcA4 = nn.Linear(dim_models, 1)
        self.fcB4 = nn.Linear(B_size, 1)
        self.fcA5 = nn.Linear(dim_models, 1)
        self.fcB5 = nn.Linear(B_size, 1)
        self.fcA6 = nn.Linear(dim_models, 1)
        self.fcB6 = nn.Linear(B_size, 1)
        self.fcA7 = nn.Linear(dim_models, 1)
        self.fcB7 = nn.Linear(B_size, 1)


    def forward(self, Enc_IN):
        enc_out, enc_attns = self.encoder(Enc_IN)  # Encoder_OUT
        flattened = enc_out.view(-1, enc_out.size(-1))

        OUTPUT0 = self.fcA0(flattened)
        flattened_all0 = OUTPUT0.view(-1)
        final_OUTPUT0 = self.fcB0(flattened_all0.unsqueeze(0))
        
        OUTPUT1 = self.fcA1(flattened)
        flattened_all1 = OUTPUT1.view(-1)
        final_OUTPUT1 = self.fcB1(flattened_all1.unsqueeze(0))

        OUTPUT2 = self.fcA2(flattened)
        flattened_all2 = OUTPUT2.view(-1)
        final_OUTPUT2 = self.fcB2(flattened_all2.unsqueeze(0))
        
        OUTPUT3 = self.fcA3(flattened)
        flattened_all3 = OUTPUT3.view(-1)
        final_OUTPUT3 = self.fcB3(flattened_all3.unsqueeze(0))
        
        

        return final_OUTPUT0, final_OUTPUT1, final_OUTPUT2, final_OUTPUT3
############################################################################################################################