
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import math

from hyperparams import HyperParams as hp
from utils import init_weight


class FCBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 use_dropout=hp.use_dropout,
                 dp=hp.dp,
                 use_norm=hp.use_norm,
                 activation=None):
        super(FCBlock, self).__init__()
        if use_dropout:
            self.dropout = nn.Dropout(p=dp)
        else:
            self.dropout = None
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init_weight(self.linear.weight)
        if use_norm == "weight":
            self.linear = nn.utils.weight_norm(self.linear)
        self.activation = Activation(
                activation=activation,
                num_features=out_features)

    def forward(self, input):
        x = input
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return self.activation(x)

class Transpose(nn.Module):
    def __init__(self, dim1: int, dim2: int, inplace=False):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            return input.transpose_(self.dim1, self.dim2)
        else:
            return input.transpose(self.dim1, self.dim2)


class ConvTransBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 length_expand_rate=1,
                 dilation=1,
                 use_dropout=hp.use_dropout,
                 dp=hp.dp,
                 use_norm=hp.use_norm,
                 activation=None):
        assert kernel_size % 2 == 1, \
                "kernel_size must be odd"
        super(ConvTransBlock, self).__init__()
        output_padding = length_expand_rate - 1
        self.use_norm = use_norm
        self.use_dropout = use_dropout

        if use_norm == "batch":
            self.batch_norm = nn.BatchNorm1d(in_channels)
        if use_dropout:
            self.dropout = nn.Dropout(p=dp)

        pad = (kernel_size - 1) * dilation // 2 + \
                (1 + output_padding - length_expand_rate) // 2

        self.conv_trans = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=length_expand_rate,
                padding=pad,
                output_padding=output_padding)
        if self.use_norm == "weight":
            self.conv_trans = nn.utils.weight_norm(self.conv_trans)
        init_weight(self.conv_trans.weight)
        self.activation = Activation(
                activation=activation,
                num_features=out_channels)

    def forward(self, input):
        x = input
        if self.use_norm == "batch":
            x = self.batch_norm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv_trans(x)
        return self.activation(x)


class PoolBlock(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 stride: int,
                 causal: bool,
                 kind="max"):
        super(PoolBlock, self).__init__()
        # assert kernel_size % stride == 0, "kernel_size % stride must be 0"
        if not causal:
            assert (kernel_size - stride) % 2 == 0, \
                   "kernel_size - stride must be even, if non-causal"
        pad = kernel_size - stride
        if causal:
            padding = (pad, 0)
        else:
            padding = (pad // 2, pad // 2)
        if kind == "max":
            self.pool = nn.Sequential(
                nn.ReplicationPad1d(padding),
                nn.MaxPool1d(
                    kernel_size=kernel_size,
                    stride=stride))
        elif kind == "avg":
            self.pool = nn.Sequential(
                nn.ReplicationPad1d(padding),
                nn.AvgPool1d(
                    kernel_size=kernel_size,
                    stride=stride))
        elif kind == "lp":
            self.pool = nn.Sequential(
                nn.ReplicationPad1d(padding),
                nn.LPPool1d(
                    kernel_size=kernel_size,
                    stride=stride))
        else:
            assert False, "not supported kind"

    def forward(self, input):
        return self.pool(input)


class Conv1d(nn.Conv1d):
    def __init__(self,
                 causal: bool,
                 pad_kind: str = "constant",
                 expanded_rate: int = 1,
                 autoregressive=False,
                 *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        pad = (self.kernel_size[0] - 1) * self.dilation[0]
        if causal:
            padding = (pad, 0)
        else:
            padding = (pad // 2, pad // 2)
        self.pad_size = padding
        if pad_kind == "constant":
            self.pad = nn.ConstantPad1d(padding, 0.0)
        elif pad_kind == "replication":
            self.pad = nn.ReplicationPad1d(padding)
        else:
            assert False
        self.expanded_rate = expanded_rate
        assert not (self.stride[0] > 1 and autoregressive), \
            "stride > 1 , not supported autoregressive"
        self.autoregressive = autoregressive
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def forward(self, input):
        x = input
        if hp.mode == "train" or not self.autoregressive:
            x = self.pad(x)
            return super().forward(x)
        else:
            return self.autoregressive_forward(x)

    def autoregressive_forward(self, input):
        '''
        input: [batch_size=1, in_channels // groups, timestep]
        '''
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        weight = self.weight
        kernel_size = self.kernel_size[0]
        dilation = self.dilation[0]
        groups = self.groups
        batch_size = input.size(0)

        input = input.data
        r = hp.reduction_factor * self.expanded_rate
        if self.input_buffer is None:
            self.input_buffer = input.new(
                    batch_size,
                    # channels
                    self.in_channels,
                    # timestep
                    (kernel_size - 1) * dilation + r)
            self.input_buffer.zero_()
        else:
            self.input_buffer[:, :, :-r] = \
                    self.input_buffer[:, :, r:]
        self.input_buffer[:, :, -r:] = \
                input[:, :, -r:]
        input = self.input_buffer
        output = F.conv1d(input, weight, self.bias, groups=groups, dilation=dilation)
        return output

    def clear_buffer(self):
        self.input_buffer = None

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def _get_linearized_weight(self):
        '''
        Return:
            [out_channels, in_channels // groups, kernel_size]
        '''
        if self._linearized_weight is None:
            groups = self.groups
            kernel_size = self.kernel_size[0]
            assert self.weight.size() == \
                    (self.out_channels, self.in_channels // groups, kernel_size)
            self._linearized_weight = self.weight
        return self._linearized_weight


class ConvBlockE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,  # must be odd
                 causal: bool,
                 glu=True,
                 dilation=1,
                 residual=None,
                 use_dropout=hp.use_dropout,
                 dp=hp.dp,
                 use_norm=hp.use_norm,
                 groups=1,
                 length_expand_rate=None,
                 stride=1,
                 pad_kind=hp.pad_kind,
                 length_shrink_rate=None,
                 length_expanded_rate=1,
                 pool_kind="max",
                 pool_kernel_size=None,
                 autoregressive=False,
                 activation=None):
        super(ConvBlockE, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.glu = glu
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.length_expand_rate = length_expand_rate
        self.length_shrink_rate = length_shrink_rate
        self.length_expanded_rate = length_expanded_rate
        self.residual = out_channels == in_channels if residual is None else residual
        self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=2 * out_channels if glu else out_channels,
                kernel_size=kernel_size,
                pad_kind=pad_kind,
                stride=stride,
                groups=groups,
                dilation=dilation,
                expanded_rate=length_expanded_rate,
                autoregressive=autoregressive,
                causal=causal)
        self.autoregressive = autoregressive
        if autoregressive:
            assert causal, "if autoregressive convolution, must be causal"
        init_weight(self.conv.weight)
        if use_norm == "weight":
            self.conv = nn.utils.weight_norm(self.conv)
        if use_norm == "batch":
            self.batch_norm = nn.BatchNorm1d(in_channels)
        if length_expand_rate is not None:
            self.upsample = nn.Upsample(scale_factor=length_expand_rate)
        if length_shrink_rate is not None:
            self.pool = PoolBlock(
                kernel_size=pool_kernel_size,
                stride=length_shrink_rate,
                causal=causal,
                kind=pool_kind)
        if use_dropout:
            self.dropout = nn.Dropout(p=dp)
        self.activation = Activation(
                activation=activation,
                num_features=out_channels)

    def forward(self, input):
        if hp.debug:
            print("ConvBlockE",
                  torch.isnan(input).any().item(),
                  torch.isinf(input).any().item())
        x = input
        if self.use_norm == "batch":
            x = self.batch_norm(x)
        if self.residual:
            residual = x
        # elif self.residual:
        #     residual = F.interpolate(
        #         x.unsqueeze(0),
        #         scale_factor=(
        #             self.out_channels / self.in_channels, 1.0
        #             )).squeeze(0)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv(x)
        if self.glu:
            x = F.glu(x, dim=1)
        if self.residual:
            x += residual
        x = self.activation(x) * (0.5 ** 0.5)
        if self.length_expand_rate is not None:
            x = self.upsample(x)
        if self.length_shrink_rate is not None:
            x = self.pool(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, dim, w=None, auto=False, scaled=False):
        super(PositionalEncoding, self).__init__()
        self.auto = auto
        self.scaled = scaled
        if scaled:
            self.scale_param = nn.Parameter(data=torch.randn(1), requires_grad=True)
        if self.auto:
            self.pe = nn.Embedding(
                num_embeddings=max_length,
                embedding_dim=dim)
        else:
            self.w = w
            seed = torch.tensor([[
                i / (10000 ** (2 * (j // 2) / dim))
                for j in range(dim)]
                    for i in range(max_length)
                ])
            self.pe = nn.Parameter(data=seed, requires_grad=False)

    def forward(self, input, speaker_embedding=None, start=0):
        if self.auto:
            pos = torch.arange(input.size(1), device=input.device)
            if self.scaled:
                return input + self.scale_param * self.pe(pos)
            else:
                return input + self.pe(pos)
        else:
            pe = torch.zeros_like(self.pe)
            pe[0::2] = torch.sin(self.pe[0::2] * self.w)
            pe[1::2] = torch.cos(self.pe[1::2] * self.w)
            return input + pe[start: start + input.size(1)]


class AttentionBlock(nn.Module):
    def __init__(self,
                 query_size: int,
                 keys_size: int,
                 values_size: int,
                 hidden_size: int,
                 out_size: int,  # Charactor Embedding Size
                 num_head=1,
                 use_norm=hp.use_norm):
        super(AttentionBlock, self).__init__()
        assert num_head is not None and \
            query_size % num_head == 0 and \
            keys_size % num_head == 0 and \
            values_size % num_head == 0 and \
            hp.charactor_embedding_size % num_head == 0, \
            "channels and charactor_embedding_size must be \
             multiples of num_head"
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.query_size = query_size
        self.keys_size = keys_size
        self.values_size = values_size

        self.pe_query = PositionalEncoding(
                max_length=hp.max_timestep,
                dim=query_size,
                w=hp.w_query,
                auto=hp.position_encoding_auto,
                scaled=hp.position_encoding_scaled)
        self.pe_key = PositionalEncoding(
                max_length=hp.max_sentence_length,
                dim=keys_size,
                w=hp.w_key,
                auto=hp.position_encoding_auto,
                scaled=hp.position_encoding_scaled)

        if num_head == 1:
            self.fc_query = FCBlock(
                    in_features=query_size,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
            self.fc_keys = FCBlock(
                    in_features=keys_size,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
            self.fc_values = FCBlock(
                    in_features=values_size,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
            self.last = FCBlock(
                    in_features=hidden_size,
                    out_features=out_size,
                    activation="none",
                    use_norm=use_norm)
            self.softmax = nn.Sequential(
                    nn.Dropout(p=hp.attention_dp),
                    nn.Softmax(dim=2))
        else:
            self.fc_query = nn.ModuleList(
                [FCBlock(
                    in_features=query_size // num_head,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
                    for _ in range(num_head)])
            self.fc_keys = nn.ModuleList(
                [FCBlock(
                    in_features=keys_size // num_head,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
                    for _ in range(num_head)])
            self.fc_values = nn.ModuleList(
                [FCBlock(
                    in_features=values_size // num_head,
                    out_features=hidden_size,
                    activation="none",
                    use_norm=use_norm)
                    for _ in range(num_head)])
            self.last = FCBlock(
                    in_features=hidden_size * num_head,
                    out_features=out_size,
                    activation="none",
                    use_norm=use_norm)
            self.softmax = nn.ModuleList(
                    [nn.Sequential(
                        nn.Dropout(p=hp.attention_dp),
                        nn.Softmax(dim=2))
                        for _ in range(num_head)])

    def forward(self,
                # from decoder
                # [batch_size, frame_langth, mel_bands]
                query,
                # from encoder
                # [batch_size, sentence_length, charactor_embedding_size]
                keys, values,
                timestep=0,
                target_mask=None,
                src_mask=None,
                speaker_embedding=None):

        if self.num_head == 1:
            v = self.fc_values(values)
        else:
            v = torch.split(values, self.values_size // self.num_head, 2)
            v = [self.fc_values[i](v[i]) for i in range(self.num_head)]

        # if target_mask is not None:
        #     pe_query = pe_query.repeat([query.size(0), 1, 1]) \
        #             .masked_fill_(target_mask, 0.0)
        query = self.pe_query(query, start=timestep)
        if self.num_head == 1:
            query = self.fc_query(query)
        else:
            query = torch.split(query, self.query_size // self.num_head, 2)
            query = [self.fc_query[i](query[i]) for i in range(self.num_head)]

        # if src_mask is not None:
        #     pe_key = pe_key.repeat([keys.size(0), 1, 1]) \
        #             .masked_fill_(src_mask, 0.0)
        keys = self.pe_key(keys)
        if self.num_head == 1:
            keys = self.fc_keys(keys)
        else:
            keys = torch.split(keys, self.keys_size // self.num_head, 2)
            keys = [self.fc_keys[i](keys[i]) for i in range(self.num_head)]

        if  self.num_head == 1:
            x = torch.bmm(query, keys.transpose_(1, 2))
        else:
            x = [torch.bmm(query[i], keys[i].transpose(1, 2))
                 for i in range(self.num_head)]
        if target_mask is not None:
            if self.num_head == 1:
                x.masked_fill_(target_mask, 0.0)
            else:
                for x_ in x:
                    x_.masked_fill_(target_mask, 0.0)
        if self.num_head == 1:
            attn = self.softmax(x)
            # print(attn[0].max(-1)[1])
        else:
            attn = [self.softmax[i](x[i]) for i in range(self.num_head)]

        sentence_length = values.size(1)
        norm_param = np.sqrt(sentence_length)
        if self.num_head == 1:
            x = torch.bmm(attn, v) / norm_param
            x = self.last(x)
        else:
            x = [torch.bmm(attn[i], v[i]) / norm_param
                 for i in range(self.num_head)]
            x = self.last(torch.cat(x, dim=2))
        return x, attn


class DecoderBlock(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 channels: int,
                 keys_values_size: int,
                 hidden_size: int,
                 conv_layers=1,
                 dilation=1,
                 groups=1,
                 autoregressive=True,
                 num_head=1,
                 use_norm=hp.use_norm):
        super(DecoderBlock, self).__init__()
        self.num_head = num_head
        self.causal_conv = nn.Sequential(
            *[ConvBlockE(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                use_norm=use_norm,
                dp=hp.decoder_dp,
                groups=groups,
                autoregressive=autoregressive,
                activation="none",
                causal=True,
                dilation=dilation ** (i % 4))
                for i in range(conv_layers)])

        self.attention = AttentionBlock(
            query_size=channels,
            keys_size=keys_values_size,
            values_size=keys_values_size,
            hidden_size=hidden_size,
            out_size=channels,
            num_head=num_head,
            use_norm=use_norm)

    def forward(self,
                # [batch_size, decoder_affine_size, reduction_factor]
                input,
                # [batch_size, sentence_length, charactor_embedding_size]
                keys, values,
                timestep,
                speaker_embedding=None,
                target_mask=None,
                src_mask=None):
        x = input.transpose(1, 2)
        x = self.causal_conv(x)
        x.transpose_(1, 2)

        residual = x
        x, attn = self.attention(
                query=x,
                keys=keys, values=values,
                timestep=timestep,
                speaker_embedding=speaker_embedding,
                target_mask=target_mask,
                src_mask=src_mask)

        x = (residual + x) * (0.5 ** 0.5)
        if type(attn) is list:
            return x, torch.stack(attn).mean(dim=0)
        else:
            return x, attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 keys_values_size: int,
                 dp: float,
                 use_norm=None):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dp)
        self.dropout1 = nn.Dropout(p=dp)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                kdim=keys_values_size,
                vdim=keys_values_size,
                dropout=dp)
        self.dropout2 = nn.Dropout(p=dp)
        self.linear = nn.Sequential(
                nn.LayerNorm(d_model),
                FCBlock(
                    in_features=d_model,
                    out_features=dim_feedforward,
                    activation="relu",
                    use_norm=use_norm,
                    dp=dp),
                FCBlock(
                    in_features=dim_feedforward,
                    out_features=d_model,
                    activation="none",
                    use_norm=use_norm,
                    dp=dp))

    def forward(self, query, keys, values,
                tgt_mask=None,
                mem_mask=None,
                tgt_key_padding_mask=None,
                mem_key_padding_mask=None):
        residual = query
        x = self.layer_norm1(query)
        x, _ = self.self_attn(x, x, x,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        x = self.dropout1(x)
        x += residual

        residual = x
        x, attn = self.multihead_attn(query=x, key=keys, value=values,
                                      attn_mask=mem_mask,
                                      key_padding_mask=mem_key_padding_mask)
        x = self.dropout2(x)
        x += residual

        x = self.linear(x) + x

        return x, attn


class Entropy(nn.Module):
    def __init__(self, method="mean"):
        super(Entropy, self).__init__()
        self.method = method

    def forward(self, input):
        eps = 1e-8
        if self.method == "mean":
            x = -torch.mean(input * torch.log2(input + eps))
        elif self.method == "sum":
            x = -torch.sum(input * torch.log2(input + eps))
        else:
            assert False, "method must be `mean` or `sum`"
        return x


class GuidedAttentionLoss(nn.Module):
    def __init__(self, method="mean"):
        super(GuidedAttentionLoss, self).__init__()
        self.method = method

    def forward(self, attn):
        N = attn.size(1)
        T = attn.size(2)
        t = torch.arange(1., T + 1.) \
                .unsqueeze(0) \
                .expand_as(attn).to(attn.device)
        n = torch.arange(1., N + 1.) \
                .unsqueeze(1) \
                .expand_as(attn).to(attn.device)
        ones = torch.ones((1, N, T)).to(attn.device)
        W = ones - torch.exp(
            (-(n / N - t / T) ** 2.)
                / (2. * np.power(hp.guided_attention_g, 2.)))
        if self.method == "sum":
            return (attn * W).sum()
        elif self.method == "mean":
            return (attn * W).mean()
        else:
            return (attn * W).mean()


class Sparsemax(nn.Module):
    def __init__(self, dim: int):
        super(Sparsemax, self).__init__()
        self.dim = dim
        self.output = None

    def forward(self, input):
        K = input.size(self.dim)
        # sort z as z(1) >= ... >= z(K)
        zs = input.transpose(self.dim, 0)
        sorted, idx = torch.sort(zs, dim=0, descending=True)
        ks = torch.arange(1.0, K + 1) \
                  .view(-1, *([1] * (len(zs.size()) - 1))) \
                  .expand_as(zs).to(input.device)
        left = 1 + ks * zs
        right = torch.cumsum(zs, dim=0)
        mask = left <= right
        ks.masked_fill_(mask, 0)
        k, _ = torch.max(ks, dim=0)
        # tau
        tmp = right.masked_fill(mask, -float("inf"))
        tmp, _ = torch.max(tmp, dim=0)
        tau = (tmp - 1.0) / k
        tau.unsqueeze_(0).transpose_(self.dim, 0)
        self.output = torch.relu(input - tau)
        return self.output

    def backward(self, grad_output):
        non_zeros = torch.ne(self.output, 0.0)
        sum = torch.sum(grad_output * non_zeros, dim=self.dim)
        return non_zeros * (grad_output - sum.expand_as(grad_output))


class Eve(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999, 0.999),
                 eps=1e-8,
                 k=0.1,
                 K=10.0,
                 weight_decay=0,
                 amsgrad=False):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            k=k,
            K=K,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        if lr <= 0.0:
            raise ValueError
        if not (0.0 <= betas[0] < 1.0) and \
           not (0.0 <= betas[1] < 1.0) and \
           not (0.0 <= betas[2] < 1.0):
            raise ValueError
        if eps <= 0.0:
            raise ValueError
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        loss = None
        f = None
        if closure is not None:
            loss = closure()
            f = loss.data.item()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                lr = group["lr"]
                beta1, beta2, beta3 = group["betas"]
                k = group["k"]
                K = group["K"]
                eps = group["eps"]
                amsgrad = group["amsgrad"]

                state = self.state[p]

                if len(state) == 0:  # init
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["d"] = 1.0
                    if amsgrad:
                        state["max_v"] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format)
                    state["f_hat"] = 0.0

                step = state["step"] + 1
                state["step"] = step

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if group["weight_decay"] != 0:
                    g.add_(group["weight_decay"], p.data)

                m = state["m"]
                m.mul_(beta1).add_(1 - beta1, g)

                v = state["v"]
                v.mul_(beta2).addcmul_(1 - beta2, g, g)

                d = None
                if step > 1:
                    if f < state["f_hat"]:
                        delta = k + 1
                        Delta = K + 1
                    else:
                        delta = 1.0 / (K + 1)
                        Delta = 1.0 / (k + 1)
                    c = min(max(delta, f / state["f_hat"]), Delta)
                    next = c * state["f_hat"]
                    if next > state["f_hat"]:
                        r = next / state["f_hat"] - 1.0
                    else:
                        r = 1.0 - state["f_hat"] / next
                    d = beta3 * state["d"] + (1 - beta3) * r
                    state["d"] = d
                    state["f_hat"] = next
                else:
                    state["f_hat"] = f
                    state["d"] = 1.0
                    d = 1.0

                if amsgrad:
                    max_v = state["max_v"]
                    torch.max(state["max_v"], v, out=max_v)
                    denom = (max_v.sqrt() /
                             math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / (bias_correction1 * d)

                p.data.addcdiv_(-step_size, m, denom)
        return loss


class SantaE(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 sigma=0.999,
                 lambda_=1e-8,
                 beta_a=1.0,
                 beta_b=0.0,
                 beta_c=2.0,
                 c=100,
                 burnin=5000):
        defaults = dict(
            lr=lr,
            sigma=sigma,
            lambda_=lambda_,
            beta_a=beta_a,
            beta_b=beta_b,
            beta_c=beta_c,
            c=c,
            burnin=burnin)
        super(SantaE, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SantaE, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                lr = group["lr"]
                sigma = group["sigma"]
                lambda_ = group["lambda_"]

                def one_over_betas(t):
                    a = group["beta_a"]
                    b = group["beta_b"]
                    c = group["beta_c"]
                    return (1 / a) * (t + b) ** (-c)
                c = group["c"]
                burnin = group["burnin"]

                if len(state) == 0:  # init
                    sq_lr = np.sqrt(lr)
                    state["step"] = 0
                    state["u"] = sq_lr * torch.randn_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["alpha"] = torch.full_like(
                        p.data,
                        sq_lr * c,
                        memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["g"] = torch.full_like(
                        p.data,
                        np.sqrt(lambda_),
                        memory_format=torch.preserve_format)

                step = state["step"] + 1
                state["step"] = step

                v = state["v"]
                v.mul_(sigma).addcmul_(1.0 - sigma, grad, grad)

                g_prev = state["g"]
                g = (1 / (lambda_ + v.sqrt())).sqrt()
                state["g"] = g

                u = state["u"]
                u_prev = u.clone()
                alpha = state["alpha"]

                if step < burnin:
                    one_over_beta = one_over_betas(step)

                    alpha.addcmul_(u_prev, u_prev).add_(-lr * one_over_beta)

                    zeta = torch.randn_like(
                        p.data,
                        memory_format=torch.preserve_format)

                    u.pow_(-1).mul_(lr * one_over_beta).mul_(1 - g_prev / g) \
                     .add_(torch.sqrt(2 * lr * one_over_beta * g) * zeta)

                else:
                    u = torch.zeros_like(
                        state["u"],
                        memory_format=torch.preserve_format)
                    state["u"] = u

                u.addcmul_((1 - alpha), u_prev).addcmul_(-lr, grad, g)
                p.data.addcmul_(u, g)

        return loss


class SantaSSS(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 sigma=0.999,
                 lambda_=1e-8,
                 betas=lambda t: np.power(t, 2.0),
                 c=100,
                 burnin=5000):
        defaults = dict(
            lr=lr,
            sigma=sigma,
            lambda_=lambda_,
            betas=betas,
            c=c,
            burnin=burnin)
        super(SantaSSS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SantaSSS, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                lr = group["lr"]
                sigma = group["sigma"]
                lambda_ = group["lambda_"]
                betas = group["betas"]
                c = group["c"]
                burnin = group["burnin"]

                if len(state) == 0:  # init
                    sq_lr = np.sqrt(lr)
                    state["step"] = 0
                    state["u"] = sq_lr * torch.randn_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["alpha"] = torch.full_like(
                        p.data,
                        sq_lr * c,
                        memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["g"] = torch.full_like(
                        p.data,
                        1.0 / np.sqrt(lambda_),
                        memory_format=torch.preserve_format)

                step = state["step"] + 1
                state["step"] = step

                v = sigma * state["v"] + (1.0 - sigma) * (grad * grad)
                state["v"] = v

                g_prev = state["g"]
                g = 1.0 / torch.sqrt(lambda_ + torch.sqrt(v))
                state["g"] = g

                u_prev = state["u"]

                p.data.addcmul_(0.5, g, u_prev)

                if step < burnin:
                    beta = betas(step)
                    lr_over_beta = lr / beta

                    alpha = state["alpha"] + \
                            0.5 * (u_prev * u_prev - lr_over_beta)

                    exp_minus_half_alpha = torch.exp(-0.5 * alpha)
                    zeta = torch.randn_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    u = exp_minus_half_alpha * u_prev
                    u += -lr * g * grad + \
                        torch.sqrt(2.0 * g_prev * lr_over_beta) * zeta + \
                        lr_over_beta * (1.0 - g_prev / g) / u_prev
                    u *= exp_minus_half_alpha

                    alpha += 0.5 * (u * u - lr_over_beta)

                    state["alpha"] = alpha
                    state["u"] = u
                else:
                    exp_minus_half_alpha = torch.exp(-0.5 * state["alpha"])
                    u = exp_minus_half_alpha * u_prev
                    u += -lr * g * grad
                    u *= exp_minus_half_alpha
                    state["u"] = u

                p.data.addcmul_(0.5, g, u)

        return loss


class Activation(nn.Module):
    def __init__(self, activation=None, num_features=None):
        super(Activation, self).__init__()
        if activation is None:
            activation = hp.activation

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "maxout":
            assert num_features is not None, "num_features must not be None"
            self.activation = Maxout(
                    num_features,
                    transpose=True)
        elif activation == "siren":
            assert num_features is not None, "num_features must not be None"
            self.activation = Siren(num_features)
        elif activation == "none":
            self.activation = nn.Identity()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "mish":
            self.activation = Mish()
        elif activation == "tanhshrink":
            self.activation = nn.Tanhshrink()
        elif activation == "sin":
            self.activation = Sine()
        else:
            self.activation = nn.LeakyReLU()

    def forward(self, input):
        return self.activation(input)


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, input):
        return torch.sin(input)


class Siren(nn.Module):
    def __init__(self, num_features: int, omega=1):
        super(Siren, self).__init__()
        self.num_features = num_features
        self.omega = omega
        self.linear = nn.Linear(num_features, num_features)
        init_weight(self.linear.weight)

    def forward(self, input):
        x = input.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        out = torch.sin(self.omega * x)
        return out


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class Maxout(nn.Module):
    def __init__(self,
                 num_features,
                 n=3,
                 dropout=hp.use_dropout,
                 transpose=False):
        super(Maxout, self).__init__()
        self.dropout = None
        self.transpose = transpose
        if dropout:
            self.dropout = nn.Dropout(hp.dp)
        self.linear_list = nn.ModuleList([
            nn.Linear(num_features, num_features)
            for _ in range(n)])

    def forward(self, input):
        x = None
        if self.transpose:
            x = input.transpose(1, 2)
        else:
            x = input
        out = []
        for layer in self.linear_list:
            out.append(layer(x))
        means = map(lambda x: torch.mean(x).cpu().numpy(), out)
        idx = np.argmax(means)
        x = out[idx]
        if self.dropout is not None:
            x = self.dropout(x)
        if self.transpose:
            x.transpose_(1, 2)
        return x


class WaveNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = ConvBlockE(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=False,
            glu=True)
        self.conv = ConvBlockE(
            in_channels=in_channels,
            out_channels=out_channels,
            causal=False,
            kernel_size=1)

    def forward(self, input):
        residual = input
        x = self.dilated_conv(input)
        x, y = torch.split(x, dim=1)
        x = torch.tanh(x) * torch.sigmoid(y)
        x = self.conv(x) + residual
        return x


class WORLDBlock(nn.Module):
    def __init__(self):
        # TODO
        self.fc_voiced = FCBlock()
        self.fc_spectral_envelope = FCBlock()
        self.fc_aperiodicity = FCBlock()
        self.fc_f0 = FCBlock()

    def forward(self, input):
        voiced = torch.sigmoid(self.fc_voiced(input))
        spectral_envelope = self.fc_spectral_envelope(input)
        aperiodicity = self.fc_aperiodicity(input)
        f0 = self.fc_f0(input)
        return voiced, spectral_envelope, aperiodicity, f0
