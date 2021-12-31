from typing import Optional, Tuple
import math

import torch
from torch import Tensor
from torch import nn

from .base import PositionalEmbedding


class EnformerPositionalEmbedding(PositionalEmbedding):
    r"""
    bin_size: Bin sized used to partition the sequence. This can be used to
    compute features on the absolute scale relative to the genome.
    feature_functions: List of different feature functions to use. Each function
    will take as argument: positions, sequence length and number of features
    to compute.
    symmetric: If True, the resulting features will be symmetric across the
    relative position of 0 (i.e. only absolute value of positions will
    matter). If false, then both the symmetric and asymmetric version
    (symmetric multiplied by sign(positions)) of the features will be used.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        dropout_p: float,
    ) -> None:
        # assert embed_dim == num_heads * head_dim
        super().__init__(
            embed_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            max_length=None,
            padding_idx=None,
            dropout_p=dropout_p,
        )

        self.pos_embed = nn.Linear(self.num_heads * head_dim, self.num_heads * head_dim)
        self.content_bias = nn.Parameter(
            torch.zeros([1, self.num_heads, 1, self.head_dim])
        )
        self.pos_embed_bias = nn.Parameter(
            torch.zeros([self.num_heads, 1, self.head_dim])
        )
        self.pos_embed_functions = [
            "positional_features_exponential",
            "positional_features_central_mask",
            "positional_features_gamma",
            "positional_features_cosine",
            "positional_features_linear_masks",
            "positional_features_sin_cos",
        ]
        self.pos_embed_symmetric = False
        self.pos_embed_components = len(
            self.pos_embed_functions
        )  # 1 per each basis function
        if not self.pos_embed_symmetric:
            self.pos_embed_components = 2 * self.pos_embed_components
        # For now, we do not allow odd sized embeddings.
        # if self.embed_dim % num_components != 0:
        #     raise ValueError(
        #         f'embed_dim has to be divisible by {num_components}')
        self.feature_size = num_heads * head_dim // self.pos_embed_components

    def forward_input(self, positions, input_):
        pass

    def forward_attn(self, q, k, positions_q, positions_k):
        pe_index = torch.arange(k.shape[-2], device=k.device, dtype=torch.long)
        return self(pe_index, q, k)

    def forward(
        self,
        positions,
        q: Tensor,
        k: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # q *= self.scaling

        batch_size, num_heads, k_len, head_dim = k.shape
        q_len = q.shape[-2]
        # what is `seq_length` used for???
        pos_embed = self.positional_features_all(positions=positions, seq_length=q_len)
        pos_embed = self.positional_embedding_dropout(pos_embed)
        pos_embed = (
            self.pos_embed(pos_embed)
            .expand(batch_size, -1, -1)
            .reshape(batch_size, self.num_heads, k_len, self.head_dim)
        )

        content_logits = torch.einsum(
            "bhqd, bhkd", q + self.content_bias.repeat(batch_size, 1, q_len, 1), k
        )
        pos_embed_logits = torch.einsum(
            "bhqd, bhkd",
            q + self.pos_embed_bias.repeat(batch_size, 1, q_len, 1),
            pos_embed,
        )
        attn = content_logits + pos_embed_logits
        # attn = F.softmax(attn, dim=-1)
        return attn  # [batch_size * num_heads, q_len, k_len]

    def positional_features_all(
        self, positions: Tensor, seq_length: Optional[int] = 500
    ):
        """Compute relative positional encodings/features.
        Each positional feature function will compute/provide the same fraction of
        features, making up the total of embed_dim.
        Args:
            positions: Tensor of relative positions of arbitrary shape.
            seq_length: Sequence length denoting the characteristic length that
            the individual positional features can use. This is required since the
            parametrization of the input features should be independent of `positions`
            while it could still require to use the total number of features.
        Returns:
            Tensor of shape: `positions.shape + (self.embed_dim,)`.
        """
        embeddings = torch.cat(
            [
                getattr(self, f)(positions.abs(), seq_length=seq_length)
                for f in self.pos_embed_functions
            ],
            axis=-1,
        )
        if not self.pos_embed_symmetric:
            embeddings = torch.cat(
                [embeddings, positions.sign()[..., None] * embeddings], axis=-1
            )

        return embeddings

    @staticmethod
    def _prepend_dims(x, num_dims):
        return x.view([1] * num_dims + list(x.shape))

    @staticmethod
    def gamma_pdf(x, concentration, rate):
        """Gamma probability distribution function: p(x|concentration, rate)."""
        log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
        log_normalization = torch.lgamma(concentration) - concentration * torch.log(
            rate
        )
        return torch.exp(log_unnormalized_prob - log_normalization)

    def positional_features_exponential(
        self,
        positions: Tensor,
        seq_length: Optional[int] = None,
        min_half_life: Optional[float] = 3.0,
    ):
        """Create exponentially decaying positional weights.
        Args:
            positions: Position tensor (arbitrary shape).
            seq_length: Sequence length.
            min_half_life: Smallest exponential half life in the grid of half lives.
        Returns:
            A Tensor with shape [2 * seq_length - 1, self.feature_size].
        """
        if seq_length is None:
            seq_length = torch.max(positions.abs()) + 1
        # Grid of half lifes from [3, seq_length / 2] with self.feature_size
        # distributed on the log scale.
        max_range = math.log(seq_length) / math.log(2.0)
        # [TODO]: self.feature_size = 1 is atemp solution to match dim
        half_life = torch.pow(
            2.0,
            torch.linspace(min_half_life, max_range, self.feature_size + 1),
        ).to(positions.device)
        half_life = self._prepend_dims(half_life, len(positions.shape))
        positions = positions.abs()
        outputs = torch.exp(-math.log(2.0) / half_life * positions[..., None])
        return outputs

    def positional_features_central_mask(self, positions: Tensor, seq_length):
        """Positional features using a central mask (allow only central features)."""
        center_widths = torch.pow(
            2.0,
            torch.arange(
                self.feature_size, dtype=torch.float32, device=positions.device
            )
            + 1,
        )
        center_widths = center_widths - 1
        center_widths = self._prepend_dims(center_widths, len(positions.shape))
        outputs = (center_widths > positions.abs()[..., None]).float()
        return outputs

    def positional_features_gamma(
        self,
        positions: Tensor,
        seq_length: Optional[int] = None,
        stddev=None,
        start_mean=None,
    ):
        """Positional features computed using the gamma distributions."""
        if seq_length is None:
            seq_length = torch.max(positions.abs()) + 1
        if stddev is None:
            stddev = seq_length / (2 * self.feature_size)
        if start_mean is None:
            start_mean = seq_length / self.feature_size
        mean = torch.linspace(
            start_mean, seq_length, self.feature_size, device=positions.device
        )
        mean = self._prepend_dims(mean, len(positions.shape))
        concentration = (mean / stddev) ** 2
        rate = mean / stddev ** 2
        probabilities = self.gamma_pdf(
            positions.abs().float()[..., None], concentration, rate
        )
        probabilities += 1e-8  # To ensure numerical stability.
        outputs = probabilities / torch.max(probabilities)
        return outputs

    def positional_features_cosine(self, positions: Tensor, seq_length):
        """Cosine positional features."""
        periodicity = 1.25 * torch.pow(
            2.0,
            torch.arange(
                self.feature_size, dtype=torch.float32, device=positions.device
            ),
        )
        periodicity = self._prepend_dims(periodicity, len(positions.shape))

        outputs = torch.cos(2 * math.pi * positions[..., None] / periodicity)
        return outputs

    def positional_features_linear_masks(self, positions: Tensor, seq_length):
        """Exponentially increasing point focuses."""
        distances = torch.arange(
            self.feature_size, dtype=torch.float32, device=positions.device
        )
        distances = self._prepend_dims(distances, len(positions.shape))
        outputs = (distances == torch.abs(positions[..., None])).float()

        return outputs

    def positional_features_sin_cos(
        self,
        positions: Tensor,
        seq_length,
        max_time: Optional[int] = 10000.0,
    ):
        """Sine/cosine positional encodings."""
        # [TODO]: self.feature_size = 1 is atemp solution to match dim
        # if self.feature_size % 2 != 0:
        #     raise ValueError('self.feature_size needs to be divisible by 2.')
        i = torch.arange(
            0, self.feature_size + 1, 2, dtype=torch.float32, device=positions.device
        )
        i = self._prepend_dims(i, len(positions.shape))

        # Concat sines and cosines and return.
        outputs = torch.cat(
            [
                torch.sin(
                    positions[..., None] / max_time ** (i / self.feature_size + 1)
                ),
                torch.cos(
                    positions[..., None] / max_time ** (i / self.feature_size + 1)
                ),
            ],
            -1,
        )

        return outputs
