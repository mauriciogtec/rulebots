from math import gcd
from typing import Literal, Optional
import torch
from torch import nn


# class MultiHeadAttentionWeightsOnly(nn.Module):
#     def __init__(self, q_dim, k_dim, num_heads, embed_dim, dropout=0.0):
#         super(MultiHeadAttentionWeightsOnly, self).__init__()

#         assert (
#             embed_dim % num_heads == 0
#         ), "Projection dimension must be divisible by the number of heads."

#         self.q_dim = q_dim
#         self.k_dim = k_dim
#         self.num_heads = num_heads
#         self.proj_dim = embed_dim
#         self.head_dim = embed_dim // num_heads

#         # Linear layers for query and key transformations
#         self.q_proj = nn.Linear(q_dim, embed_dim, bias=False)
#         self.k_proj = nn.Linear(k_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, querys, keys, attn_mask=None, key_padding_mask=None):
#         # Input shapes:
#         #   query: (batch_size, query_len, q_dim)
#         #   key: (batch_size, key_len, k_dim)
#         #   attn_mask: (batch_size, key_len, key_len)
#         # Output shape:
#         #   attn_weights: (batch_size, query_len, key_len)

#         # check key and query have the same batch dimension
#         assert (
#             keys.dim() == querys.dim()
#         ), "Query and key must have the same batch dimension."

#         # check dimensions is 2D or 3D
#         assert keys.dim() in [2, 3], "Input key must be 2D or 3D tensor."

#         # Add batch dimension if it doesn't exist
#         is_batched = keys.dim() == 3
#         if not is_batched:
#             querys = querys.unsqueeze(0)
#             keys = keys.unsqueeze(0)

#         batch_size, query_len, q_dim = querys.size()
#         _, key_len, k_dim = keys.size()

#         assert q_dim == self.q_dim, "Query dimension must match the initialized q_dim."
#         assert k_dim == self.k_dim, "Key dimension must match the initialized k_dim."

#         # Linear transformations for query and key
#         Q = self.q_proj(querys)  # (batch_size, query_len, proj_dim)
#         K = self.k_proj(keys)  # (batch_size, key_len, proj_dim)

#         # Apply dropout after projections
#         Q = self.dropout(Q)
#         K = self.dropout(K)

#         # Reshape and transpose for multi-head computation
#         Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # Scaled dot-product attention weights (logits)
#         attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
#         # attn_weights shape: (batch_size, num_heads, query_len, key_len)

#         # Aggregate the attention weights across heads
#         attn_logits = attn_logits.mean(dim=1)  # Aggregate across the head dimension

#         # merge key padding and attention mask
#         mask = torch.ones(batch_size, query_len, key_len).to(attn_logits.device)

#         if key_padding_mask is not None:
#             mask *= key_padding_mask.unsqueeze(1).float()

#         if attn_mask is not None:
#             mask *= attn_mask.float()

#         # Apply the mask
#         attn_logits = attn_logits + (1.0 - mask) * -1e9

#         # Remove batch dimension if it was added for non-batch inputs
#         if not is_batched:
#             attn_logits = attn_logits.squeeze(0)

#         return attn_logits


class AttentionNetwork(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        output_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_attention_layers=2,
        normalize_inputs: bool = True,
        weights_only: bool = False,
        add_bias_kv: bool = True,
        initial_proj: bool = True,
    ):
        super().__init__()
        assert (
            num_attention_layers >= 1
        ), "Number of attention layers must be at least 1."
        assert (
            output_dim is not None or weights_only
        ), "Output dimension must be specified if not weights only."

        self.normalize_inputs = normalize_inputs

        if normalize_inputs:
            self.query_norm = nn.LayerNorm(q_dim)
            self.key_norm = nn.LayerNorm(k_dim)

        # initial projection layers
        self.proj = initial_proj
        if initial_proj:
            self.q_proj = nn.Sequential(
                nn.Linear(q_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            q_dim = hidden_dim

            self.k_proj = nn.Sequential(
                nn.Linear(k_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            k_dim = hidden_dim

        self.num_attention_layers = num_attention_layers
        embed_dim = hidden_dim

        self.attn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        for i in range(self.num_attention_layers):
            if i == 0:
                embed_dim = q_dim
            else:
                embed_dim = hidden_dim

            # num heads is the greatest common divisor of the input and output dimensions
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=gcd(num_heads, embed_dim),
                    dropout=dropout,
                    kdim=k_dim,
                    batch_first=True,
                    add_bias_kv=(i < num_attention_layers - 1) and add_bias_kv,
                )
            )
            if i < self.num_attention_layers - 1:
                self.act_layers.append(nn.SiLU())
                self.norm_layers.append(nn.LayerNorm(embed_dim))
            else:
                # identity layer
                self.act_layers.append(nn.Identity())
                self.norm_layers.append(nn.Identity())

        self.weights_only = weights_only
        if weights_only:
            head = self.attn_layers[-1]
            if hasattr(head, "v_proj_weight") and head.v_proj_weight is not None:
                head.v_proj_weight.requires_grad = False
            else:
                # TODO handle this case
                pass
            if hasattr(head, "bias_v") and head.bias_v is not None:
                head.bias_v.requires_grad = False

    def forward(self, queries, keys, attn_mask=None, key_padding_mask=None):
        if self.normalize_inputs:
            queries = self.query_norm(queries)
            keys = self.key_norm(keys)

        if hasattr(self, "q_proj"):
            queries = self.q_proj(queries)

        if hasattr(self, "k_proj"):
            keys = self.k_proj(keys)

        weights = None
        for i in range(self.num_attention_layers):
            attn_layer = self.attn_layers[i]
            act_layer = self.act_layers[i]
            norm_layer = self.norm_layers[i]

            output, weights = attn_layer(
                queries,
                keys,
                keys,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            if output.shape[-1] == queries.shape[-1]:
                queries = norm_layer(act_layer(queries + output))
            else:
                queries = norm_layer(act_layer(output))

        if self.weights_only:
            return weights
        else:
            return queries, weights


class AttentionActorCritic(nn.Module):
    """Actor where the output is the action logits and the critic is a state-only value function."""

    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        num_heads: int = 8,
        dropout: float = 0.0,
        proj: bool = True,
        num_attention_layers: int = 2,
        num_critic_layers: int = 2,
        normalize_inputs: bool = True,  # Input normalization flag
        sequential_critic: bool = False,
    ):
        super().__init__()
        assert (
            num_attention_layers >= 1
        ), "Number of attention layers must be at least 1."
        assert num_critic_layers >= 1, "Number of critic layers must be at least 1."

        self.actor = AttentionNetwork(
            q_dim=q_dim,
            k_dim=k_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_attention_layers=num_attention_layers,
            normalize_inputs=normalize_inputs,
            proj=proj,
            weights_only=True,
        )

        # state only critic, output dim should be 1
        self.sequential_critic = sequential_critic
        if not sequential_critic:
            self.critic = AttentionNetwork(
                q_dim=q_dim,
                k_dim=k_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_attention_layers=num_critic_layers,
                normalize_inputs=normalize_inputs,
                output_dim=1,
                proj=proj,
            )
        else:
            self.critic = nn.Sequential()
            for i in range(num_critic_layers):
                in_dim = self.actor.q_dim if i == 0 else hidden_dim
                out_dim = hidden_dim if i < num_critic_layers - 1 else 1
                self.critic.append(nn.Linear(in_dim, out_dim))
                if i < num_critic_layers - 1:
                    self.critic.append(nn.SiLU())
                    self.critic.append(nn.LayerNorm(out_dim))

    def forward(self, queries, keys, attn_mask=None, key_padding_mask=None):
        attn_logits = self.actor(queries, keys, attn_mask, key_padding_mask)

        if self.sequential_critic:
            value = self.critic(queries)
        else:
            value = self.critic(queries, keys, attn_mask, key_padding_mask)
        value = value.squeeze(-1)

        return attn_logits, value


# # Define the Attention-based Actor Network with Multi-Head Attention
# class AttentionActorCritic(nn.Module):
#     def __init__(
#         self,
#         q_dim,
#         k_dim,
#         hidden_dim,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         q_proj: bool = True,
#         k_proj: bool = True,
#         num_attention_layers: int = 2,
#         num_critic_layers: int = 2,
#         normalize_inputs: bool = True,  # Input normalization flag
#     ):
#         super().__init__()
#         assert (
#             num_attention_layers >= 1
#         ), "Number of attention layers must be at least 1."
#         assert num_critic_layers >= 1, "Number of critic layers must be at least 1."

#         self.actor = _build_attention_network(
#             q_dim,
#             k_dim,
#             hidden_dim,
#             num_heads,
#             dropout,
#             q_proj,
#             k_proj,
#             num_attention_layers,
#             normalize_inputs,
#         )

#         # state only state
#         self.critic = nn.Sequential()
#         for i in range(num_critic_layers):
#             in_dim = self.actor.q_dim if i == 0 else hidden_dim
#             out_dim = hidden_dim if i < num_critic_layers - 1 else 1
#             self.critic.append(nn.Linear(in_dim, out_dim))
#             if i < num_critic_layers - 1:
#                 self.critic.append(nn.SiLU())
#                 self.critic.append(nn.LayerNorm(out_dim))

#         for m in self.modules():
#             if hasattr(m, "weight"):
#                 nn.init.kaiming_normal_(m.weight)
#             if hasattr(m, "bias") and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, query, keys, attn_mask=None, key_padding_mask=None):
#         if self.actor.normalize_inputs:
#             query = self.actor.query_norm(query)
#             keys = self.actor.key_norm(keys)

#         if hasattr(self.actor, "q_proj"):
#             query = self.actor.q_proj(query)

#         if hasattr(self.actor, "k_proj"):
#             keys = self.actor.k_proj(keys)

#         value = self.critic(query).squeeze(-1)

#         for i in range(0, len(self.actor.attn_body), 3):
#             attn_layer = self.actor.attn_body[i]
#             act_layer = self.actor.attn_body[i + 1]
#             norm_layer = self.actor.attn_body[i + 2]

#             layer_output, _ = attn_layer(
#                 query,
#                 keys,
#                 keys,
#                 attn_mask=attn_mask,
#                 key_padding_mask=key_padding_mask,
#             )
#             query = norm_layer(act_layer(query + layer_output))

#         attn_logits = self.actor.attn_head(
#             query, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
#         )

#         return attn_logits, value


# # Define the Attention-based Actor Network with Multi-Head Attention
# class AttentionActor(nn.Module):
#     def __init__(
#         self,
#         q_dim,
#         k_dim,
#         hidden_dim,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         q_proj: bool = True,
#         k_proj: bool = True,
#         num_attention_layers: int = 2,
#         normalize_inputs: bool = True,  # Input normalization flag
#     ):
#         super().__init__()
#         assert (
#             num_attention_layers >= 1
#         ), "Number of attention layers must be at least 1."

#         self.normalize_inputs = normalize_inputs

#         if normalize_inputs:
#             self.query_norm = nn.LayerNorm(q_dim)
#             self.key_norm = nn.LayerNorm(k_dim)

#         if q_proj:
#             self.q_proj = nn.Sequential(
#                 nn.Linear(q_dim, hidden_dim),
#                 nn.SiLU(),
#                 nn.LayerNorm(hidden_dim),
#             )
#             self.q_dim = hidden_dim
#         else:
#             self.q_dim = q_dim

#         if k_proj:
#             self.k_proj = nn.Sequential(
#                 nn.Linear(k_dim, hidden_dim),
#                 nn.SiLU(),
#                 nn.LayerNorm(hidden_dim),
#             )
#             self.k_dim = hidden_dim
#         else:
#             self.k_dim = k_dim

#         if num_attention_layers == 1:
#             self.attn_body = []
#         else:
#             self.attn_body = nn.ModuleList()
#             for i in range(num_attention_layers - 1):
#                 embed_dim = self.q_dim if i == 0 else hidden_dim
#                 self.attn_body.append(
#                     nn.MultiheadAttention(
#                         embed_dim=embed_dim,
#                         num_heads=num_heads,
#                         dropout=dropout,
#                         kdim=self.k_dim,
#                         vdim=hidden_dim,
#                         batch_first=True,
#                         add_bias_kv=True,
#                     )
#                 )
#                 self.attn_body.append(nn.SiLU())
#                 self.attn_body.append(nn.LayerNorm(embed_dim))

#         self.attn_head = MultiHeadAttentionWeightsOnly(
#             q_dim=self.q_dim,
#             k_dim=self.k_dim,
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#         )

#         for m in self.modules():
#             if hasattr(m, "weight"):
#                 nn.init.kaiming_normal_(m.weight)
#             if hasattr(m, "bias") and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, query, keys, attn_mask=None, key_padding_mask=None):
#         if self.normalize_inputs:
#             query = self.query_norm(query)
#             keys = self.key_norm(keys)

#         if hasattr(self, "q_proj"):
#             query = self.q_proj(query)

#         if hasattr(self, "k_proj"):
#             keys = self.k_proj(keys)

#         for i in range(0, len(self.attn_body), 3):
#             attn_layer = self.attn_body[i]
#             act_layer = self.attn_body[i + 1]
#             norm_layer = self.attn_body[i + 2]

#             layer_output, _ = attn_layer(
#                 query,
#                 keys,
#                 keys,
#                 attn_mask=attn_mask,
#                 key_padding_mask=key_padding_mask,
#             )
#             query = norm_layer(act_layer(query + layer_output))

#         attn_logits = self.attn_head(
#             query, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
#         )

#         return attn_logits


if __name__ == "__main__":
    # Test the MultiHeadAttentionWeightsOnly module
    querys = torch.randn(2, 5, 8)
    keys = torch.randn(2, 7, 32)
    # atn mask must be a 2D tensor of
    attn_mask = torch.rand(5, 7).round().bool()
    key_padding_mask = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool
    )
    # Test the AttentionActorCritic module
    actor = AttentionActorCritic(
        q_dim=8,
        k_dim=32,
        hidden_dim=64,
        num_heads=8,
        dropout=0.25,
        q_proj=True,
        k_proj=True,
    )

    attn_logits, value = actor(querys, keys, attn_mask, key_padding_mask)

    print("Attention logits shape:", attn_logits.shape)
