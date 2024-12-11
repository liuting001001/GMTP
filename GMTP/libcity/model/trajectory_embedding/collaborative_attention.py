import math
from enum import Enum
import torch
import torch.nn as nn

import torch.nn.functional as F


class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class CollaborativeAttention(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_value_all: int,
        dim_key_query_all: int,
        dim_output: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: 0.,
        use_dense_layer: bool,
        use_layer_norm: bool,
        mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
        add_temporal_bias: bool = True,
        temporal_bias_dim: int = 64,
        use_mins_interval: bool = False,
        device: torch.device = torch.device('cpu'),
        add_cls: bool = True
    ):
        super().__init__()

        if dim_value_all % num_attention_heads != 0:
            raise ValueError(
                "Value dimension ({}) should be divisible by number of heads ({})".format(
                    dim_value_all, num_attention_heads
                )
            )

        if not use_dense_layer and dim_value_all != dim_output:
            raise ValueError(
                "Output dimension ({}) should be equal to value dimension ({}) if no dense layer is used".format(
                    dim_output, dim_value_all
                )
            )

        # save args
        self.dim_input = dim_input
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.dim_output = dim_output
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer
        self.use_layer_norm = use_layer_norm
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval
        self.device = device
        self.add_cls = add_cls

        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = (
            dim_key_query_all / num_attention_heads
        )  # does not have to be integer

        # intialize parameters
        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.content_bias = nn.Linear(dim_input, num_attention_heads, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)
        self.mixing = self.init_mixing_matrix()

        self.dense = (
            nn.Linear(dim_value_all, dim_output) if use_dense_layer else nn.Sequential()
        )

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(dim_value_all)
        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

        print("dim_input",dim_input)
        print("dim_value_all", dim_value_all)
        print("dim_key_query_all", dim_key_query_all)
        print("dim_output", dim_output)
        print("num_attention_heads", num_attention_heads)
        print("use_dense_layer",use_dense_layer)
        print("use_layer_norm",use_layer_norm)
        print("dim_value_per_head", self.dim_value_per_head)
        print("attention_head_size", self.attention_head_size)
        print("use_layer_norm", use_layer_norm)

    def forward(
        self,
        x,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        padding_masks=None,
        future_mask=True,
        output_attentions=False,
        batch_temporal_mat=None
    ):
        from_sequence = x
        to_sequence = x
        batch_size, seq_len, d_model = to_sequence.shape
        if encoder_hidden_states is not None:
            to_sequence = encoder_hidden_states
            attention_mask = encoder_attention_mask

        query_layer = self.query(from_sequence)
        key_layer = self.key(to_sequence)

        # point wise multiplication of the mixing coefficient per head with the shared query projection
        # (batch, from_seq, dim) x (head, dim) -> (batch, head, from_seq, dim)
        mixed_query = query_layer[..., None, :, :] * self.mixing[..., :, None, :]

        # broadcast the shared key for all the heads
        # (batch, 1, to_seq, dim)
        mixed_key = key_layer[..., None, :, :]
        print("mixed_query.shape",mixed_query.shape)
        print("mixed_key.shape", mixed_key.shape)

        # (batch, head, from_seq, to_seq)
        attention_scores = torch.matmul(mixed_query, mixed_key.transpose(-1, -2))

        # add the content bias term
        # (batch, to_seq, heads)
        if batch_temporal_mat is None:
            batch_temporal_mat = torch.zeros(batch_size, seq_len, seq_len, device=self.device)
        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            attention_scores += batch_temporal_mat  # (B, 1, T, T)

        content_bias = self.content_bias(to_sequence)
        # (batch, heads, 1, to_seq)
        broadcast_content_bias = content_bias.transpose(-1, -2).unsqueeze(-2)
        attention_scores += broadcast_content_bias
        if padding_masks is not None:
            attention_scores.masked_fill_(padding_masks == 0, float('-inf'))
        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            attention_scores.masked_fill_(mask_postion, float('-inf'))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = self.value(to_sequence)
        value_layer = self.transpose_for_scores(value_layer)
        print("value_layer.shape", value_layer.shape)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim_value_all,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.dense(context_layer)


        if self.use_layer_norm:
            context_layer = self.layer_norm(from_sequence + context_layer)
        print("context_layer.shape", context_layer.shape)
        if self.output_attentions:
            return (context_layer, attention_probs)
        else:
            return (context_layer,)


    def transpose_for_scores(self, y):
        new_y_shape = y.size()[:-1] + (self.num_attention_heads, -1)
        y = y.view(*new_y_shape)
        return y.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)

        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            # last head will be smaller if not equally divisible
            dim_head = int(math.ceil(self.dim_key_query_all / self.num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0

        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.one_()
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError(
                "Unknown mixing matrix initialization: {}".format(
                    self.mixing_initialization
                )
            )

        return nn.Parameter(mixing)