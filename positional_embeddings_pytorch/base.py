from typing import Optional

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Positional embedding is responsible for:
    1) adding positional embeddings to the input
    2) calculate q k similarity given query and key (where positional embedding is blended)
    3) other things related to the calculation of positional embedding, such as applying
        layernorm, dropout or scaling of positional embedding

    It is not responsible for:
    1) scaling similarity matrix
    2) adding padding mask
    3) applying softmax to get attention weights
    4) applying dropout on attention
    5) applying layernorm
    """

    def __init__(
        self,
        *,
        embed_dim: Optional[int],
        num_heads: Optional[int],
        head_dim: Optional[int],
        max_length: Optional[int],
        padding_idx: Optional[int],
        dropout_p: Optional[float],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        # self.scale = 1 / (head_dim ** 0.5) if head_dim is not None else None
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.positional_embedding_dropout = nn.Dropout(dropout_p) if dropout_p else None

    def forward_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_q: torch.Tensor,
        positions_k: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # q: [batch_size, num_heads, q_len, head_dim]
        # k: [batch_size, num_heads, k_len, head_dim]
        # positions_q: [batch_size, q_len]
        # positions_k: [batch_size, k_len]

        # return: [batch_size, num_heads, q_len, k_len]
        raise NotImplementedError

    def forward_input(
        self, positions: torch.Tensor, input_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        # positions: [batch_size, seq_len], for absolute positional embedding
        # input_: [batch_size, seq_len, embed_dim]

        # return: [batch_size, seq_len, embed_dim]
        raise NotImplementedError
