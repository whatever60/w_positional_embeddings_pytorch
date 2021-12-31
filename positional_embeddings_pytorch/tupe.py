import math
from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .base import PositionalEmbedding


# this is from T5
def relative_position_bucket(
    relative_position,
    bidirectional: Optional[bool] = True,
    num_buckets: Optional[int] = 32,
    max_distance: Optional[int] = 128,
):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret = (n < 0).long() * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = (
        max_exact
        + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
    )
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)
    return nn.Parameter(ret, requires_grad=False)


# this is from TUPE
class TUPE(PositionalEmbedding):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout_p: float,
        has_cls_token: bool = False,
        rel_pos_embed: bool,
        num_buckets: int = 32,
        max_distance: int = 128,
        pos_scale_factor: Optional[int] = 1,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=None,
            max_length=max_length,
            padding_idx=None,
            dropout_p=dropout_p,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_length = max_length
        self.scaling = (embed_dim / num_heads * pos_scale_factor) ** -0.5

        assert has_cls_token == False, "CLS token is currently not supported"
        self.has_cls_token = has_cls_token
        if self.has_cls_token:
            # make room for [CLS]-to-others and others-to-[CLS]
            self.max_length += 2
        self.abs_pos_embed = nn.Parameter(torch.randn(self.max_length, self.embed_dim))
        self.ln = nn.LayerNorm(self.embed_dim)
        self.in_proj = nn.Linear(
            self.num_heads * self.head_dim, self.num_heads * self.head_dim * 2
        )

        # ==== relative positional embedding ====
        self.rel_pos_embed = rel_pos_embed
        if self.rel_pos_embed:
            assert num_buckets % 2 == 0
            self.num_buckets = num_buckets
            self.max_distance = max_distance
            self.rel_pos_embed = nn.Embedding(self.num_buckets + 1, self.num_heads)

    def compute_bias(
        self,
        relative_position,
        batch_size,
        q_len,
        k_len,
        cls_token_index: Optional[None],
    ) -> Tensor:
        assert cls_token_index is None, "CLS token is currently not supported"
        seq_len = max(q_len, k_len)
        # 0 is for others-to-[CLS] 1 is for [CLS]-to-others
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.has_cls_token:
            # only plus 1 here since because [CLS] already plused 1
            seq_len += 1
        # [seq_len, num_heads * head_dim]
        weight = self.ln(self.abs_pos_embed[:seq_len, :])
        # [num_heads, seq_len, head_dim]
        q, k = (
            self.in_proj(weight)
            .reshape(seq_len, 2, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )
        q, k = q[:q_len], k[:k_len]
        q = q * self.scaling
        # [num_heads, q_len, k_len]
        pos_embed = torch.bmm(q, k.transpose(1, 2))

        if self.has_cls_token:
            # p_0 \dot p_0 is [CLS]-to-others
            cls_2_others = pos_embed[:, 0, 0]
            # p_1 \dot p_1 is others-to-[CLS]
            others_2_cls = pos_embed[:, 1, 1]
            # offset
            pos_embed = pos_embed[:, 1:, 1:]
            # if [CLS] is not the first token
            if cls_token_index is not None:
                pos_embed = pos_embed.repeat(batch_size, 1, 1, 1)
                pos_embed[
                    torch.arange(batch_size), :, cls_token_index, :
                ] = cls_2_others.expand(batch_size, -1).unsqueeze(-1)
                pos_embed[
                    torch.arange(batch_size), :, :, cls_token_index
                ] = others_2_cls.expand(batch_size, -1).unsqueeze(-1)
            else:
                pos_embed[:, 0, :] = cls_2_others.unsqueeze(-1)
                pos_embed[:, :, 0] = others_2_cls.unsqueeze(-1)
            seq_len -= 1
        
        # ==== relative position embedding ====
        rel_pos_embed = torch.zeros_like(pos_embed)
        if self.rel_pos_embed:
            rel_pos_embed_bucket = self.rel_pos_embed_bucket[:seq_len, :seq_len]
            rel_pos_embed_bucket = relative_position_bucket(
                max_length=self.max_length,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            if self.has_cls_token:
                if cls_token_index is not None:
                    rel_pos_embed_bucket = rel_pos_embed_bucket.repeat(batch_size, 1, 1)
                    rel_pos_embed_bucket[
                        torch.arange(batch_size), cls_token_index, :
                    ] = (self.num_buckets // 2)
                    rel_pos_embed_bucket[
                        torch.arange(batch_size), :, cls_token_index
                    ] = self.num_buckets
                    rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(
                        0, 3, 1, 2
                    )
                else:
                    rel_pos_embed_bucket[0, :] = self.num_buckets // 2
                    rel_pos_embed_bucket[:, 0] = self.num_buckets
                    rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(
                        2, 0, 1
                    )
            else:
                rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(
                    2, 0, 1
                )
            pos_embed += rel_pos_embed

        pos_embed = (
            pos_embed.view(-1, *pos_embed.shape[2:])
            if cls_token_index is not None
            else pos_embed.repeat(batch_size, 1, 1, 1)
        )
        return self.positional_embedding_dropout(pos_embed)

    def forward(self, q, k):
        qlen, klen = q.shape[-2], k.shape[-2]
        context_position = torch.arange(
            qlen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]

        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        relative_position = memory_position - context_position  # shape (qlen, klen)
        # [batch_size, num_heads, qlen, klen]
        attn_bias = self.compute_bias(relative_position, q.shape[0], qlen, klen)
        return attn_bias + attn

    def forward_input(self, positions, input_):
        pass

    def forward_attn(self, q, k, positions_q, positions_k):
        return self(q, k)  # shape (1, num_heads, qlen, klen)
