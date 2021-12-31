"""Sinusoidal positional embedding in the original paper.
https://github.com/pytorch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
"""

import math
from typing import Optional

import torch
import torch.onnx.operators
from torch import nn

from .base import PositionalEmbedding


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, *, embed_dim, max_length, padding_idx):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=None,
            head_dim=None,
            max_length=max_length,
            padding_idx=padding_idx,
            dropout_p=None,
        )
        init_size = max_length + padding_idx + 1
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embed_dim, padding_idx
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(
        max_length: int, embed_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            max_length, -1
        )
        if embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, positions):
        # [batch_size, max_length]
        bsz, seq_len = positions.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embed_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    def forward_input(self, positions, input_):
        return self(positions)

    def forward_attn(self, positions, q, k):
        pass
