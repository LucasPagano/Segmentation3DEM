import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorch_Models.CEECNET.utils import get_norm


class SigmoidCrisp(nn.Module):
    def __init__(self, smooth=1.e-2):
        super().__init__()

        self.smooth = smooth
        self.gamma = nn.Parameter(torch.ones(size=(1,)))

    def forward(self, x):
        out = self.smooth + torch.sigmoid(self.gamma)
        out = torch.reciprocal(out)

        out = torch.mul(x, out)
        out = torch.sigmoid(out)
        return out


class Conv2DNormed(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, strides=(1, 1),
                 padding=(0, 0), dilation=(1, 1), _norm_type='BatchNorm', norm_groups=None, groups=1,):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, channels, kernel_size, strides, padding, dilation, groups)
        self.norm = get_norm(_norm_type, channels, norm_groups)

    def forward(self, x):
        return self.norm(self.conv2d(x))


class ExpandLayer(nn.Module):
    def __init__(self, in_channels, nfilters, _norm_type='BatchNorm', norm_groups=None, ngroups=1):
        super().__init__()

        self.conv1 = Conv2DNormed(in_channels=in_channels, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type, norm_groups=norm_groups)
        self.bilinear_resize_2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = Conv2DNormed(in_channels=nfilters, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type, norm_groups=norm_groups)

    def forward(self, x):
        out = self.bilinear_resize_2d(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        return out


class ExpandNCombine(nn.Module):
    def __init__(self, in_channels, nfilters, _norm_type='BatchNorm', norm_groups=None, ngroups=1, **kwards):
        super().__init__()

        self.conv1 = Conv2DNormed(in_channels=in_channels, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type, norm_groups=norm_groups, **kwards)
        self.bilinear_resize_2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = Conv2DNormed(in_channels=nfilters * 2, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type, norm_groups=norm_groups, **kwards)

    def forward(self, input1, input2):
        out = self.bilinear_resize_2d(input1)
        out = self.conv1(out)
        out = F.relu(out)
        out2 = self.conv2(torch.cat((out, input2), dim=1))
        out2 = F.relu(out2)

        return out2


class ExpandNCombine_V3(nn.Module):
    def __init__(self, in_channels, nfilters, _norm_type='BatchNorm', norm_groups=None, ngroups=1, ftdepth=5):
        super().__init__()
        self.bilinear_resize_2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = Conv2DNormed(in_channels=in_channels, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type, norm_groups=norm_groups)  # restore help
        self.conv2 = Conv2DNormed(in_channels=nfilters, channels=nfilters, kernel_size=3, padding=1, groups=ngroups,
                                  _norm_type=_norm_type,
                                  norm_groups=norm_groups)  # restore help
        self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding=1, nheads=ngroups, norm=_norm_type,
                            norm_groups=norm_groups, ftdepth=ftdepth)  # process

    def forward(self, input1, input2):
        out = self.bilinear_resize_2d(input1)
        out = self.conv1(out)
        out1 = F.relu(out)

        out2 = self.conv2(input2)
        out2 = F.relu(out2)

        outf = self.conv3(out1, out2)
        outf = F.relu(outf)

        return outf


class RelFTAttention2D(nn.Module):
    def __init__(self, in_channels, nkeys, kernel_size=3, padding=1, nheads=1, norm='BatchNorm', norm_groups=None,
                 ftdepth=5):
        super().__init__()

        self.query = Conv2DNormed(in_channels=in_channels, channels=nkeys, kernel_size=kernel_size, padding=padding,
                                  _norm_type=norm,
                                  norm_groups=norm_groups, groups=nheads)
        self.key = Conv2DNormed(in_channels=in_channels, channels=nkeys, kernel_size=kernel_size, padding=padding,
                                _norm_type=norm,
                                norm_groups=norm_groups, groups=nheads)
        self.value = Conv2DNormed(in_channels=in_channels, channels=nkeys, kernel_size=kernel_size, padding=padding,
                                  _norm_type=norm,
                                  norm_groups=norm_groups, groups=nheads)

        self.metric_channel = FTanimoto(depth=ftdepth, axis=[2, 3])
        self.metric_space = FTanimoto(depth=ftdepth, axis=1)

        self.norm = get_norm(name=norm, channels=nkeys, norm_groups=norm_groups)

    def forward(self, input1, input2, input3):
        # These should work with ReLU as well
        q = torch.sigmoid(self.query(input1))
        k = torch.sigmoid(self.key(input2))  # B,C,H,W
        v = torch.sigmoid(self.value(input3))  # B,C,H,W

        att_spat = self.metric_space(q, k)  # B,1,H,W
        v_spat = torch.mul(att_spat, v)  # emphasize spatial features

        att_chan = self.metric_channel(q, k)  # B,C,1,1
        v_chan = torch.mul(att_chan, v)  # emphasize spatial features

        v_cspat = 0.5 * torch.mul(v_chan, v_spat)  # emphasize spatial features
        v_cspat = self.norm(v_cspat)

        return v_cspat


class FTAttention2D(nn.Module):
    def __init__(self, in_channels, nkeys, kernel_size=3, padding=1, nheads=1, norm='BatchNorm', norm_groups=None,
                 ftdepth=5):
        super().__init__()

        self.att = RelFTAttention2D(in_channels=in_channels, nkeys=nkeys, kernel_size=kernel_size, padding=padding,
                                    nheads=nheads, norm=norm,
                                    norm_groups=norm_groups, ftdepth=ftdepth)

    def forward(self, x):
        return self.att(x, x, x)


class FTanimoto(nn.Module):
    """
    This is the average fractal Tanimoto set similarity with complement.
    """

    def __init__(self, depth=5, smooth=1.0e-5, axis=[2, 3]):
        super().__init__()

        assert depth >= 0, "Expecting depth >= 0, aborting ..."

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1. / depth

        self.smooth = smooth
        self.axis = axis

    def inner_prod(self, prob, label):
        prod = torch.mul(prob, label)
        prod = torch.sum(prod, dim=self.axis, keepdim=True)

        return prod

    def tnmt_base(self, preds, labels):

        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)

        num = tpl + self.smooth
        denum = 0.0

        for d in range(self.depth):
            a = 2. ** d
            b = -(2. * a - 1.)

            denum = denum + torch.reciprocal(torch.add(a * (tpp + tll), b * tpl) + self.smooth)

        return torch.mul(num, denum) * self.scale

    def forward(self, preds, labels):
        l12 = self.tnmt_base(preds, labels)
        l12 = l12 + self.tnmt_base(1. - preds, 1. - labels)

        return 0.5 * l12


class DownSample(nn.Module):
    def __init__(self, in_channels, nfilters, factor=2, _norm_type='BatchNorm', norm_groups=None):
        super().__init__()

        # Double the size of filters, since you downscale by 2.
        self.factor = factor
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3, 3)
        self.strides = (factor, factor)
        self.pad = (1, 1)

        self.convdn = Conv2DNormed(in_channels=in_channels, channels=self.nfilters,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding=self.pad,
                                   _norm_type=_norm_type,
                                   norm_groups=norm_groups)

    def forward(self, x):
        return self.convdn(x)


class UpSample(nn.Module):
    def __init__(self, nfilters, factor=2, _norm_type='BatchNorm', norm_groups=None):
        super().__init__()
        self.factor = factor
        self.nfilters = nfilters // self.factor

        self.convup_normed = Conv2DNormed(in_channels=nfilters, channels=self.nfilters,
                                          kernel_size=(1, 1),
                                          _norm_type=_norm_type,
                                          norm_groups=norm_groups)
        self.uplsampling = nn.Upsample(scale_factor=self.factor, mode="nearest")

    def forward(self, x):
        x = self.uplsampling(x)
        x = self.convup_normed(x)

        return x


class PSPPooling(nn.Module):
    def __init__(self, in_channels, nfilters, depth=4, _norm_type='BatchNorm', norm_groups=None, mob=False):

        super().__init__()
        self.nfilters = nfilters
        self.depth = depth

        convs = []
        for _ in range(depth):
            convs.append(
                Conv2DNormed(in_channels=in_channels, channels=self.nfilters, kernel_size=(1, 1), padding=(0, 0),
                             _norm_type=_norm_type,
                             norm_groups=norm_groups))
        self.convs = nn.Sequential(*convs)
        self.conv_norm_final = Conv2DNormed(in_channels=nfilters * self.depth + in_channels, channels=self.nfilters,
                                            kernel_size=(1, 1),
                                            padding=(0, 0),
                                            _norm_type=_norm_type,
                                            norm_groups=norm_groups)

    # ******** Utilities functions to avoid calling infer_shape ****************
    def half_split(self, _a):
        """
        Returns a list of half split arrays. Usefull for HalfPoolling
        """
        size_b = _a.size(2) // 2
        b = torch.split(_a, size_b, 2)  # Split First dimension
        size_c = b[0].size(3) // 2
        c1 = torch.split(b[0], size_c, 3)  # Split 2nd dimension
        c2 = torch.split(b[1], size_c, 3)  # Split 2nd dimension

        d11 = c1[0]
        d12 = c1[1]

        d21 = c2[0]
        d22 = c2[1]

        return [d11, d12, d21, d22]

    def quarter_stitch(self, _dss):
        """
        INPUT:
            A list of [d11,d12,d21,d22] block matrices.
        OUTPUT:
            A single matrix joined of these submatrices
        """

        temp1 = torch.cat((_dss[0], _dss[1]), dim=-1)
        temp2 = torch.cat((_dss[2], _dss[3]), dim=-1)
        result = torch.cat((temp1, temp2), dim=2)

        return result

    def half_pooling(self, _a):
        Ds = self.half_split(_a)
        Dss = []
        for x in Ds:
            # kernel size = image size is for global pooling
            Dss += [torch.mul(torch.ones_like(x), F.max_pool2d(x, kernel_size=x.size()[2:]))]

        return self.quarter_stitch(Dss)

    def split_pooling(self, _a, depth):
        """
        A recursive function that produces the Pooling you want - in particular depth (powers of 2)
        """
        if depth == 1:
            return self.half_pooling(_a)
        else:
            D = self.half_split(_a)
            return self.quarter_stitch([self.split_pooling(d, depth - 1) for d in D])

    # ***********************************************************************************

    def forward(self, _input):

        p = [_input]
        # 1st:: Global Max Pooling .
        p += [self.convs[0](torch.mul(torch.ones_like(_input), F.max_pool2d(_input, kernel_size=_input.size()[2:])))]
        p += [self.convs[d](self.split_pooling(_input, d)) for d in range(1, self.depth)]
        out = torch.cat(p, dim=1)
        out = self.conv_norm_final(out)

        return out


class CombineLayers(nn.Module):
    def __init__(self, in_channels, _nfilters, _norm_type='BatchNorm', norm_groups=None):
        super().__init__()
        # This performs convolution, no BatchNormalization. No need for bias.
        self.up = UpSample(nfilters=in_channels, _norm_type=_norm_type, norm_groups=norm_groups)

        self.conv_normed = Conv2DNormed(in_channels=in_channels, channels=_nfilters,
                                        kernel_size=(1, 1),
                                        padding=(0, 0),
                                        _norm_type=_norm_type,
                                        norm_groups=norm_groups)

    def forward(self, _layer_lo, _layer_hi):
        up = self.up(_layer_lo)
        up = F.relu(up)
        x = torch.cat((up, _layer_hi), dim=1)
        x = self.conv_normed(x)

        return x


class Fusion(nn.Module):
    def __init__(self, nfilters, kernel_size=3, padding=1, nheads=1, norm='BatchNorm', norm_groups=None, ftdepth=5):
        super().__init__()

        self.fuse = Conv2DNormed(in_channels=nfilters * 2
                                 , channels=nfilters, kernel_size=kernel_size, padding=padding,
                                 _norm_type=norm,
                                 norm_groups=norm_groups, groups=nheads)
        self.relatt12 = RelFTAttention2D(in_channels=nfilters, nkeys=nfilters, kernel_size=kernel_size,
                                         padding=padding, nheads=nheads,
                                         norm=norm, norm_groups=norm_groups, ftdepth=ftdepth)
        self.relatt21 = RelFTAttention2D(in_channels=nfilters, nkeys=nfilters, kernel_size=kernel_size,
                                         padding=padding, nheads=nheads,
                                         norm=norm, norm_groups=norm_groups, ftdepth=ftdepth)

        self.gamma1 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma2 = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, input_t1, input_t2):
        # These inputs must have the same dimensionality , t1, t2
        relatt12 = torch.mul(self.gamma1, self.relatt12(input_t1, input_t2, input_t2))
        relatt21 = torch.mul(self.gamma2, self.relatt21(input_t2, input_t1, input_t1))

        ones = torch.ones_like(input_t1)

        # Enhanced output of 1, based on memory of 2
        out12 = torch.mul(input_t1, ones + relatt12)
        # Enhanced output of 2, based on memory of 1
        out21 = torch.mul(input_t2, ones + relatt21)

        fuse = self.fuse(torch.cat((out12, out21), dim=1))
        fuse = F.relu(fuse)

        return fuse


class CATFusion(nn.Module):
    """
    Alternative to concatenation followed by normed convolution: improves performance.
    """

    def __init__(self, nfilters_out, nfilters_in, kernel_size=3, padding=1, nheads=1, norm='BatchNorm',
                 norm_groups=None, ftdepth=5):
        super().__init__()

        self.fuse = Conv2DNormed(nfilters_in * 2, nfilters_out, kernel_size=kernel_size, padding=padding,
                                 _norm_type=norm,
                                 norm_groups=norm_groups, groups=nheads)
        self.relatt12 = RelFTAttention2D(in_channels=nfilters_in, nkeys=nfilters_in, kernel_size=kernel_size,
                                         padding=padding, nheads=nheads,
                                         norm=norm, norm_groups=norm_groups, ftdepth=ftdepth)
        self.relatt21 = RelFTAttention2D(in_channels=nfilters_in,
                                         nkeys=nfilters_in, kernel_size=kernel_size, padding=padding, nheads=nheads,
                                         norm=norm, norm_groups=norm_groups, ftdepth=ftdepth)

        self.gamma1 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma2 = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, input_t1, input_t2):
        # These inputs must have the same dimensionality , t1, t2
        relatt12 = torch.mul(self.gamma1, self.relatt12(input_t1, input_t2, input_t2))
        relatt21 = torch.mul(self.gamma2, self.relatt21(input_t2, input_t1, input_t1))

        ones = torch.ones_like(input_t1)

        # Enhanced output of 1, based on memory of 2
        out12 = torch.mul(input_t1, ones + relatt12)
        # Enhanced output of 2, based on memory of 1
        out21 = torch.mul(input_t2, ones + relatt21)

        fuse = self.fuse(torch.cat((out12, out21), dim=1))
        fuse = F.relu(fuse)

        return fuse


class CombineLayersWithFusion(nn.Module):
    def __init__(self, nfilters, nheads=1, _norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__()

        self.conv1 = Conv2DNormed(channels=nfilters, kernel_size=3, padding=1, groups=nheads, _norm_type=_norm_type,
                                  norm_groups=norm_groups, **kwards)  # restore help
        self.bilinear_resize_2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding=1, nheads=nheads, norm=_norm_type,
                            norm_groups=norm_groups, ftdepth=ftdepth)  # process

    def forward(self, _layer_lo, _layer_hi):
        up = self.bilinear_resize_2d(_layer_lo)
        up = self.conv1(up)
        up = F.relu(up)
        x = self.conv3(up, _layer_hi)

        return x


class ResNetV2Block(nn.Module):
    """
    ResNet v2 building block. It is built upon the assumption of ODD kernel
    """

    def __init__(self, in_channels, _nfilters, _kernel_size=(3, 3), _dilation_rate=(1, 1),
                 _norm_type='BatchNorm', norm_groups=None, ngroups=1):
        super().__init__()

        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate

        # Ensures padding = 'SAME' for ODD kernel selection
        p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1) / 2
        p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1) / 2
        p = (int(p0), int(p1))

        self.BN1 = get_norm(_norm_type, in_channels, norm_groups=norm_groups)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.nfilters, kernel_size=self.kernel_size,
                               padding=p,
                               dilation=self.dilation_rate, bias=False, groups=ngroups)
        self.BN2 = get_norm(_norm_type, self.nfilters, norm_groups=norm_groups)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=self.nfilters, kernel_size=self.kernel_size,
                               padding=p,
                               dilation=self.dilation_rate, bias=True, groups=ngroups)

    def forward(self, _input_layer):
        x = self.BN1(_input_layer)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x


class FracTALResNetUnit(nn.Module):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3, 3), dilation_rate=(1, 1),
                 norm_type='BatchNorm',
                 norm_groups=None, ftdepth=5):
        super().__init__()
        in_channels = nfilters
        self.block1 = ResNetV2Block(in_channels, nfilters, kernel_size, dilation_rate, _norm_type=norm_type,
                                    norm_groups=norm_groups, ngroups=ngroups)
        self.attn = FTAttention2D(in_channels=nfilters, nkeys=nfilters, nheads=nheads, kernel_size=kernel_size,
                                  norm=norm_type,
                                  norm_groups=norm_groups, ftdepth=ftdepth)

        self.gamma = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        out1 = self.block1(x)

        att = self.attn(x)
        att = torch.mul(self.gamma, att)

        out = torch.mul((x + out1), torch.ones_like(out1) + att)
        return out
