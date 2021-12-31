"""Binned relative positional embedding used in T5.
https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py
"""

import math
import torch
import torch.nn as nn

from .base import PositionalEmbedding


class T5PositionalEmbedding(PositionalEmbedding):
    def __init__(self, *, num_heads: int, num_buckets=32, max_distance=128):
        super().__init__(
            embed_dim=None,
            num_heads=num_heads,
            head_dim=None,
            max_length=None,
            padding_idx=None,
            dropout_p=None,
        )
        # self.bidirectional = bidirectional
        self.bidirectional = True
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(
                torch.long
            ) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, relative_position):
        """Compute binned relative position bias"""
        
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

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
        # [1, num_heads, qlen, klen]
        attn_bias = self.compute_bias(relative_position)
        return attn_bias + attn

    def forward_input(self, positions, input_):
        pass

    def forward_attn(self, q, k, positions_q, positions_k):
        return self(q, k)  # shape (1, num_heads, qlen, klen)
