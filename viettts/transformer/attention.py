import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math

class MultiHeadedAttention(nn.Module):

    def __init__(
        self, 
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        key_bias :bool = True
    ):
        "Construct for MultiHeaded Attention"

        super().__init__()
        assert n_feat % n_head == 0 ,"n_feat must be divisible by n_head"

        self.n_head = n_head

        self.head_dim = n_feat // n_head

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias = key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.fc_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    )->Tuple[Tensor, Tensor, Tensor]:
        """
        Cấu hình kích thước
        query = B x T x C với C là số lượng đặc trưng và T là sô lượng bước thời gian
        
        Trả về:
        torch.Tensor: Transformed query tensor, size
            (#batch, n_head, time1, d_k).
        torch.Tensor: Transformed key tensor, size
            (#batch, n_head, time2, d_k).
        torch.Tensor: Transformed value tensor, size
            (#batch, n_head, time2, d_k).
        """
        batch_size = query.shape[0]

        #Chuyển đổi thành dạng (B, T, C) -> (B, T, n_head, head_dim)
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)  

        """
        tại sao cần transpose(1,2) ?
        để chuyển đổi kích thước tensor từ dạng (B, T, n_head, head_dim) sang (B, n_head, T, head_dim)
        Điều này cho phép chúng ta dễ dàng thực hiện phép nhân ma trận giữa các đầu (n_head) và các bước thời gian (T) trong quá trình tính toán attention.
        """
        return q,k,v

    def forward_attn(
        self,
        value : Tensor,
        scores: Tensor,
        mask : Tensor = torch.ones((0,0,0), dtypes = torch.bool),
    ) -> Tensor:
        """
        tính toán attention score vector
        value Tensor : batch_size x n_head x T x head_dim
        scores Tensor : batch_size x n_head x T1 x T2
        mask = size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
        """
        B = value.shape[0]


        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:,:,:, scores.shape[-1]]
            scores = scores.masked_fill(mask, -float("inf"))

        attn = F.softmax(scores, dim = -1).masked_fill(mask, 0.0) if mask.size(2) > 0 else F.softmax(scores, dim = -1)

        attn = self.dropout(attn)

        x = torch.matmul(attn, value)

        x = (x.transpose(1, 2).contiguous().view(B, -1,
                                                 self.n_head * self.head_dim)
             )  # (batch, time1, d_model)

        return self.fc_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value : Tensor,
        mask : Tensor = torch.ones((0,0,0), dtype=torch.bool),
        pos_emb : Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[Tensor, Tensor]:
        q,k,v = self.forward_qkv(query, key, value)
        if cache.shape[0] > 0:
            key_cache, value_cache = torch.split(
                cache,
                cache.size(-1) // 2,
                dim = -1
            )
            k = torch.cat([key_cache, k ], dim = 2)
            v = torch.cat([value_cache, v], dim = 2)
        new_cache = torch.cat((k,v), dim = -1)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        return self.forward_attn(v, attn_scores, mask), new_cache
        

class RelPositionMultiHeadedAttention(MultiHeadedAttention):

    def __init__(
        self,
        n_head: int,
        n_feat : int,
        dropout_rate: float,
        key_bias: bool = True
    ) :
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias= False)


        self.pos_bias_u = nn.Parameter(Tensor(self.n_head, self.head_dim))
        self.pos_bias_v = nn.Parameter(Tensor(self.n_head, self.head_dim))

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def relative_shift(self, x: Tensor) -> Tensor:
        """
        tính toán relative positional encoding
        """
        zero_pad = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2], 1),
            device = x.device,
            dtype = x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim = -1)
        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        
        x = x_padded[:, : , 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the posi
        return x
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ):
        q, k ,v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.relative_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attn(v, scores, mask), new_cache
    
