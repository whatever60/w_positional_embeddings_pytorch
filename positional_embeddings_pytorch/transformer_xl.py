"""
https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""

import torch
from torch import nn

from .base import PositionalEmbedding


class TransformerXLPositionalEmbedding(PositionalEmbedding):
    def __init__(
        self, *, embed_dim: int, num_heads: int, head_dim: int, dropout_p: float
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            max_length=None,
            padding_idx=None,
            dropout_p=dropout_p,
        )

        self.r_net = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.pos_emb = _PositionalEmbedding(self.embed_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(1, self.num_heads, 1, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(1, self.num_heads, 1, self.head_dim))

    def forward(self, positions, q, k):
        # positions: [batch_size, k_len]
        # q, k: [batch_size, num_heads, (q/k)_len, head_dim]

        r = self.positional_embedding_dropout(self.pos_emb(positions))
        # [1, num_heads, k_len, head_dim]
        r_head_k = (
            self.r_net(r)
            .view(r.shape[0], self.num_heads, self.head_dim)
            .permute(1, 0, 2)
        ).unsqueeze(0)

        #### compute attention score
        rw_head_q = q + self.r_w_bias
        AC = torch.einsum("bhqd,bhkd->bhqk", rw_head_q, k)
        # AC = torch.einsum("bhqd,bhkd->bhqk", self.r_w_bias, k)

        rr_head_q = q + self.r_r_bias
        BD = torch.einsum("bhqd,bhkd->bhqk", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        # attn_score.mul_(self.scale)
        return attn_score

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward_input(self, positions, input_):
        pass

    def forward_attn(self, q, k, positions_q, positions_k):
        positions = torch.arange(
            k.shape[-2] - 1, -1, -1.0, device=k.device, dtype=k.dtype
        )
        return self(positions, q, k)


class _PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]
