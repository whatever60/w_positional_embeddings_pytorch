import math
from typing import Optional

import torch
from torch import nn
from torch import Tensor

from .base import PositionalEmbedding


# this is from T5
def relative_position_bucket(
    relative_position,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
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
    return ret


# this is from TUPE
class DountUnifiedPositionalEmbedding(PositionalEmbedding):
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        max_pos: int,
        dropout_p: float,
        has_cls_token: bool = False,
        rel_pos_embed: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
        pos_scale_factor: Optional[int] = 1,
    ) -> None:
        super().__init__(
            embed_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            max_pos=max_pos,
            padding_idx=None,
            dropout_p=dropout_p,
        )
        self.scaling = (head_dim * pos_scale_factor) ** -0.5

        assert has_cls_token == False, "CLS token is currently not supported"
        self.has_cls_token = has_cls_token
        if self.has_cls_token:
            # make room for [CLS]-to-others and others-to-[CLS]
            self.max_pos += 2
        self.ln = nn.LayerNorm(self.head_dim * self.num_heads)

        self.abs_pos_embed = nn.Parameter(
            torch.randn(self.max_pos, self.head_dim * self.num_heads)
        )
        self.in_proj = nn.Linear(
            self.num_heads * self.head_dim, self.num_heads * self.head_dim
        )

        # ==== relative positional embedding ====
        self.rel_pos_embed = rel_pos_embed
        if self.rel_pos_embed:
            assert num_buckets % 2 == 0
            self.num_buckets = num_buckets
            self.max_distance = max_distance
            self.rel_pos_embed = nn.Embedding(self.num_buckets + 1, self.num_heads)

            # context_pos = torch.arange(max_pos, dtype=torch.long)[:, None]
            # memory_pos = torch.arange(max_pos, dtype=torch.long)[None, :]
            # relative_pos = memory_pos - context_pos  # shape (qlen, klen)
            # self.rel_pos_embed_bucket = relative_position_bucket(
            #     relative_position=relative_pos,
            #     num_buckets=self.num_buckets,
            #     max_distance=self.max_pos,
            # )

    def compute_bias(
        self,
        position_q,
        position_k,
        cls_token_index: Optional[Tensor] = None,
    ) -> Tensor:
        assert cls_token_index is None, "CLS token is currently not supported"
        batch_size, num_ts, q_len = position_q.shape
        batch_size, num_ts, k_len = position_k.shape

        # [batch_size, num_ts, q_len + k_len, num_heads * head_dim]
        weight = self.ln(
            self.abs_pos_embed[torch.cat([position_q, position_k], dim=-1).long()]
        )
        # [batch_size, num_ts, q_len + k_len, num_heads, head_dim]
        # -> [batch_size, num_ts, num_heads, q_len + k_len, head_dim]
        q_k = (
            self.in_proj(weight)
            .view(batch_size, num_ts, q_len + k_len, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        q, k = q_k[:, :, :, :q_len], q_k[:, :, :, -k_len:]
        q = q * self.scaling

        pos_embed = torch.einsum("bphqd, bphkd -> bphqk", q, k)
        # ==== relative position embedding ====
        if self.rel_pos_embed:
            # [batch_size, num_ts, q_len, k_len]
            relative_position = position_k[:, :, None, :] - position_q[:, :, :, None]
            # rel_pos_embed_bucket = self.rel_pos_embed_bucket[:q_len, :k_len]
            rel_pos_embed_bucket = relative_position_bucket(
                relative_position=relative_position.long(),
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            # [batch_size, num_ts, q_len, k_len]
            # -> [batch_size, num_ts, q_len, k_len, num_heads]
            # -> [batch_size, num_ts, q_len, k_len, num_heads]
            # -> [batch_size, num_ts, num_heads, q_len, k_len]
            rel_pos_embed = self.rel_pos_embed(rel_pos_embed_bucket).permute(
                0, 1, 4, 2, 3
            )
            pos_embed += rel_pos_embed
        return self.positional_embedding_dropout(pos_embed)

    def forward(self, position_q, position_k):
        # [batch_size, num_heads, qlen, klen]
        attn_bias = self.compute_bias(position_q, position_k)
        return attn_bias

    def forward_input(self, input_, position):
        return input_

    def forward_attn(self, q, k, position_q, position_k):
        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        # [1, num_heads, qlen, klen]
        return attn + self(position_q, position_k).mean(dim=1)
