"""Learnable positional embedding using nn.Embedding.
https://github.com/pytorch/fairseq/blob/main/fairseq/modules/learned_positional_embedding.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PositionalEmbedding


class LearnedPositionalEmbedding(PositionalEmbedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, *, max_length: int, embed_dim: int, padding_idx: int):

        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust max_length appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        super().__init__(
            embed_dim=embed_dim,
            num_heads=None,
            head_dim=None,
            max_length=max_length,
            padding_idx=padding_idx,
            dropout_p=None,
        )
        
        max_length = max_length + 1
        self.emb = nn.Embedding(max_length, embed_dim, padding_idx=padding_idx)

        nn.init.normal_(self.weight, mean=0, std=embed_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)
        
    def __getattr__(self, name):
        """
        If the user tries to get a nonexistent attribute, assume it's part of the
        embedding layer and forward it to the embedding layer.
        """
        try:
            return getattr(self.emb, name)
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )

    def forward(self, positions):
        """Input is expected to be of size [bsz x seqlen]."""
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    
    def forward_input(self, positions, input_):
        return self(positions)

    def forward_attn(self, positions, q, k):
        pass
