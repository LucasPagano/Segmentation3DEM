import torch

import PyTorch_Models.Attentions.NonLocalBlock as NonLocalBlock

from torch import nn
from torch.nn import functional as f
from torch.nn import Module, Conv2d, Parameter, Softmax


# ------------------------------------------------------ Linear Attention ------------------------------------------------------
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PositionLinearAttention(Module):
    """ Position linear attention.
        Linear Attention Mechanism: An Efficient Attention for Semantic Segmentation: https://arxiv.org/pdf/2007.14902v2.pdf
        https://github.com/lironui/Linear-Attention-Mechanism
    """

    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class ChannelLinearAttention(Module):
    """ Channel linear attention.
        Linear Attention Mechanism: An Efficient Attention for Semantic Segmentation: https://arxiv.org/pdf/2007.14902v2.pdf
        https://github.com/lironui/Linear-Attention-Mechanism
    """

    def __init__(self, eps=1e-6):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = x.view(batch_size, chnnels, -1)
        K = x.view(batch_size, chnnels, -1)
        V = x.view(batch_size, chnnels, -1)

        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)

        # tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, t))

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)

        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class LAMblock(nn.Module):
    """ This class builds a Linear Attention (PositionLinearAttention+ChannelLinearAttention) block.
    """

    def __init__(self, in_places, eps=1e-6):
        super(LAMblock, self).__init__()
        self.PLA = PositionLinearAttention(in_places, eps=eps)
        self.CLA = ChannelLinearAttention(eps=eps)

    def forward(self, x):
        return self.PLA(x) + self.CLA(x)


# ------------------------------------------------------ Efficient Attention ------------------------------------------------------
class EfficientAttention(Module):
    """ Efficient Attention: Attention with Linear Complexities. https://arxiv.org/pdf/1812.01243.pdf
        https://github.com/cmsflash/efficient-attention
    """

    def __init__(self, in_channels, key_channels, head_count, value_channels, normalization, initialization):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        if initialization is not None:
            if initialization == "0":
                torch.nn.init.zeros_(self.keys.weight)
                torch.nn.init.zeros_(self.queries.weight)
                torch.nn.init.zeros_(self.values.weight)
                torch.nn.init.zeros_(self.reprojection.weight)
            else:
                raise Exception("Unsupported initialization: '" + str(initialization) + "'.")

        if normalization is None:
            self.normalization = None
        elif normalization == "BN":
            self.normalization = nn.BatchNorm2d(in_channels)
        elif normalization == "IN":
            self.normalization = nn.InstanceNorm2d(in_channels)
        else:
            raise Exception("Unknown normalization, got '" + str(normalization) + "', expected among {None, BN, IN}.")

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = f.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        if self.normalization is not None:
            attention = self.normalization(attention)

        return attention


class EAblock(Module):
    def __init__(self, in_channels: int, parameters: str):
        super().__init__()

        params = parameters.split('_')

        if params[0] != "EA":
            raise Exception("Efficient Attention parameters error: the first block is not 'EA'.")

        if len(params) != 6:
            raise Exception("Five delimiters '_' expected in Efficient Attention parameters.")

        for i in range(1, 4):
            if isinstance(params[i], int):  # type(params[i]) != int:
                raise Exception("Efficient Attention parameter %d is not an integer: %s" % (i, params[i]))

        param4 = None if params[4] == "None" else params[4]
        param5 = None if params[5] == "None" else params[5]

        self.EA = EfficientAttention(in_channels, int(params[1]), int(params[2]), int(params[3]), param4, param5)

    def forward(self, x):
        return self.EA(x)


# ------------------------------------------------------ Non Local Block ------------------------------------------------------
class NLblock(Module):
    def __init__(self, in_channels: int, parameters: str):
        super().__init__()

        params = parameters.split('_')

        if not params[0].startswith("NL"):
            raise Exception("Non Local Block Attention parameters error: the first block does not start with 'NL'.")

        if len(params) != 5:
            raise Exception("Four delimiters '_' expected in Non Local Block Attention parameters.")

        param1 = None if params[1] == "None" else params[1]

        if isinstance(params[2], int):
            raise Exception("Non Local Block Attention parameter 2 is not an integer: %s." % params[2])
        elif int(params[2]) not in {1, 2, 3}:
            raise Exception("Non Local Block Attention parameter 2 is the dimension and must be among {1, 2, 3}.")

        for i in range(3, 4):
            if isinstance(params[i], bool):
                raise Exception("Non Local Block Attention parameter %d is not an boolean: %s" % (i, params[i]))

        if params[0] == "NLC":
            self.NL = NonLocalBlock.Concatenation(in_channels, param1, int(params[2]), bool(params[3]), bool(params[4]))
        elif params[0] == "NLP":
            self.NL = NonLocalBlock.Product(in_channels, param1, int(params[2]), bool(params[3]), bool(params[4]))
        elif params[0] == "NLE":
            self.NL = NonLocalBlock.EmbeddedGaussian(in_channels, param1, int(params[2]), bool(params[3]),
                                                     bool(params[4]))
        elif params[0] == "NLG":
            self.NL = NonLocalBlock.Gaussian(in_channels, param1, int(params[2]), bool(params[3]), bool(params[4]))
        else:
            raise Exception("Unknown non local block attention type: '" + str(
                params[0]) + "'. Expected: NLC (concatenation), NLP (product), " +
                            "NLE (embedded gaussian), NLG (gaussian).")

    def forward(self, x):
        return self.NL(x)
