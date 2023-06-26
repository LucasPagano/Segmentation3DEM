import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorch_Models.CEECNET.blocks import Conv2DNormed, ExpandNCombine, ExpandLayer, FTAttention2D, RelFTAttention2D, DownSample, \
    PSPPooling, CombineLayers, CombineLayersWithFusion, ExpandNCombine_V3, Fusion, CATFusion, FracTALResNetUnit
from PyTorch_Models.CEECNET.head import HeadCMTSKBC


class CEEC_unit_v1(nn.Module):
    def __init__(self, nfilters, nheads=1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__()
        nfilters_init = nfilters // 2
        self.conv_init_1 = Conv2DNormed(in_channels=nfilters, channels=nfilters_init, kernel_size=3, padding=1,
                                        strides=1, groups=ngroups,
                                        _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr11 = Conv2DNormed(in_channels=nfilters_init, channels=nfilters_init * 2, kernel_size=3, padding=1,
                                    strides=2, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr12 = Conv2DNormed(in_channels=nfilters_init * 2, channels=nfilters_init * 2, kernel_size=3, padding=1,
                                    strides=1, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.expand1 = ExpandNCombine(in_channels=nfilters_init * 2, nfilters=nfilters_init, _norm_type=norm_type,
                                      norm_groups=norm_groups, ngroups=ngroups)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        self.conv_init_2 = Conv2DNormed(in_channels=nfilters, channels=nfilters_init, kernel_size=3, padding=1,
                                        strides=1, groups=ngroups,
                                        _norm_type=norm_type, norm_groups=norm_groups, **kwards)  # half size

        self.expand2 = ExpandLayer(in_channels=nfilters_init, nfilters=nfilters_init // 2, _norm_type=norm_type,
                                   norm_groups=norm_groups,
                                   ngroups=ngroups)
        self.compr21 = Conv2DNormed(in_channels=nfilters_init // 2, channels=nfilters_init, kernel_size=3, padding=1,
                                    strides=2, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr22 = Conv2DNormed(in_channels=nfilters_init * 2, channels=nfilters_init, kernel_size=3, padding=1,
                                    strides=1, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)

        # Will join with master input with concatenation  -- IMPORTANT: ngroups = 1 !!!!
        self.collect = Conv2DNormed(in_channels=nfilters, channels=nfilters, kernel_size=3, padding=1, strides=1,
                                    groups=1,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)

        self.att = FTAttention2D(in_channels=nfilters, nkeys=nfilters, nheads=nheads, norm=norm_type,
                                 norm_groups=norm_groups,
                                 ftdepth=ftdepth)
        self.ratt122 = RelFTAttention2D(in_channels=nfilters_init, nkeys=nfilters_init, nheads=nheads, norm=norm_type,
                                        norm_groups=norm_groups,
                                        ftdepth=ftdepth)
        self.ratt211 = RelFTAttention2D(in_channels=nfilters_init, nkeys=nfilters_init, nheads=nheads, norm=norm_type,
                                        norm_groups=norm_groups,
                                        ftdepth=ftdepth)

        self.gamma1 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma2 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma3 = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        # =========== UNet branch ===========
        out10 = self.conv_init_1(x)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        out1 = self.expand1(out1, out10)
        out1 = F.relu(out1)

        # =========== \capNet branch ===========

        out20 = self.conv_init_2(x)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(torch.cat((out2, out20), dim=1))
        out2 = F.relu(out2)

        att = torch.mul(self.gamma1, self.att(x))
        ratt122 = torch.mul(self.gamma2, self.ratt122(out1, out2, out2))
        ratt211 = torch.mul(self.gamma3, self.ratt211(out2, out1, out1))

        ones1 = torch.ones_like(out10)
        ones2 = torch.ones_like(x)

        # Enhanced output of 1, based on memory of 2
        out122 = torch.mul(out1, ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1
        out211 = torch.mul(out2, ones1 + ratt211)

        out12 = F.relu(self.collect(torch.cat((out122, out211), dim=1)))

        # Emphasize residual output from memory on input
        out_res = torch.mul(x + out12, ones2 + att)
        return out_res


class CEEC_unit_v2(nn.Module):
    def __init__(self, nfilters, nheads=1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5,
                 **kwards):
        super().__init__()

        nfilters_init = nfilters // 2
        self.conv_init_1 = Conv2DNormed(in_channels=nfilters, channels=nfilters_init, kernel_size=3, padding=1,
                                        strides=1, groups=ngroups,
                                        _norm_type=norm_type, norm_groups=norm_groups)  # half size
        self.compr11 = Conv2DNormed(in_channels=nfilters_init, channels=nfilters_init * 2, kernel_size=3, padding=1,
                                    strides=2, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)  # half size
        self.compr12 = Conv2DNormed(in_channels=nfilters_init * 2, channels=nfilters_init * 2, kernel_size=3, padding=1,
                                    strides=1, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)  # process
        self.expand1 = ExpandNCombine_V3(in_channels=nfilters_init * 2, nfilters=nfilters_init, _norm_type=norm_type,
                                         norm_groups=norm_groups,
                                         ngroups=ngroups, ftdepth=ftdepth)  # restore original size + process

        self.conv_init_2 = Conv2DNormed(in_channels=nfilters, channels=nfilters_init, kernel_size=3, padding=1,
                                        strides=1, groups=ngroups,
                                        _norm_type=norm_type, norm_groups=norm_groups, **kwards)  # half size
        self.expand2 = ExpandLayer(in_channels=nfilters_init, nfilters=nfilters_init // 2, _norm_type=norm_type,
                                   norm_groups=norm_groups,
                                   ngroups=ngroups)
        self.compr21 = Conv2DNormed(in_channels=nfilters_init // 2, channels=nfilters_init, kernel_size=3, padding=1,
                                    strides=2, groups=ngroups,
                                    _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr22 = Fusion(nfilters=nfilters_init, kernel_size=3, padding=1, nheads=ngroups, norm=norm_type,
                              norm_groups=norm_groups, ftdepth=ftdepth)

        self.collect = CATFusion(nfilters_out=nfilters, nfilters_in=nfilters_init, kernel_size=3, padding=1,
                                 nheads=1, norm=norm_type, norm_groups=norm_groups, ftdepth=ftdepth)

        self.att = FTAttention2D(in_channels=nfilters, nkeys=nfilters, nheads=nheads, norm=norm_type,
                                 norm_groups=norm_groups,
                                 ftdepth=ftdepth)
        self.ratt122 = RelFTAttention2D(in_channels=nfilters_init, nkeys=nfilters_init, nheads=nheads, norm=norm_type,
                                        norm_groups=norm_groups,
                                        ftdepth=ftdepth)
        self.ratt211 = RelFTAttention2D(in_channels=nfilters_init, nkeys=nfilters_init, nheads=nheads, norm=norm_type,
                                        norm_groups=norm_groups,
                                        ftdepth=ftdepth)

        self.gamma1 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma2 = nn.Parameter(torch.zeros(size=(1,)))
        self.gamma3 = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        # =========== UNet branch ===========
        out10 = self.conv_init_1(x)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        out1 = self.expand1(out1, out10)
        out1 = F.relu(out1)

        # =========== \capNet branch ===========
        out20 = self.conv_init_2(x)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(out2, out20)

        att = torch.mul(self.gamma1, self.att(x))
        ratt122 = torch.mul(self.gamma2, self.ratt122(out1, out2, out2))
        ratt211 = torch.mul(self.gamma3, self.ratt211(out2, out1, out1))

        ones1 = torch.ones_like(out10)
        ones2 = torch.ones_like(x)

        # Enhanced output of 1, based on memory of 2
        out122 = torch.mul(out1, ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1
        out211 = torch.mul(out2, ones1 + ratt211)

        out12 = self.collect(out122, out211)  # includes relu, it's for fusion

        out_res = torch.mul(x + out12, ones2 + att)
        return out_res


class XDNFeatures(nn.Module):
    def __init__(self, in_channels, nfilters_init, depth, widths=[1], psp_depth=4, verbose=True, norm_type='BatchNorm',
                 norm_groups=None, nheads_start=8, model='CEECNetV1', upFuse=False, ftdepth=5):
        super().__init__()

        self.depth = depth

        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = Conv2DNormed(in_channels, nfilters_init, kernel_size=(1, 1), _norm_type=norm_type,
                                       norm_groups=norm_groups)

        self.convs_dn = []
        self.pools = []

        for idx in range(depth):
            nheads = nheads_start * 2 ** idx  #
            nfilters = nfilters_init * 2 ** idx
            if verbose:
                print("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(idx, nfilters, nheads, widths[idx]))
            tnet = []
            for _ in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.append(CEEC_unit_v1(nfilters=nfilters, nheads=nheads, ngroups=nheads, norm_type=norm_type,
                                             norm_groups=norm_groups, ftdepth=ftdepth))
                elif model == 'CEECNetV2':
                    tnet.append(CEEC_unit_v2(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                             norm_type=norm_type,
                                             norm_groups=norm_groups, ftdepth=ftdepth))
                elif model == 'FracTALResNet':
                    tnet.append(
                        FracTALResNetUnit(nfilters=nfilters, nheads=nheads, ngroups=nheads, norm_type=norm_type,
                                          norm_groups=norm_groups, ftdepth=ftdepth))
                else:
                    raise ValueError(
                        "I don't know requested model, available options: CEECNetV1, CEECNetV2, FracTALResNet - Given model::{}, aborting ...".format(
                            model))
            self.convs_dn.extend(tnet)

            if idx < depth - 1:
                self.pools.append(
                    DownSample(in_channels=nfilters, nfilters=nfilters, _norm_type=norm_type, norm_groups=norm_groups))
        self.convs_dn = nn.Sequential(*self.convs_dn)
        self.pools = nn.Sequential(*self.pools)
        print("MIDDLE")
        # Middle pooling operator
        self.middle = PSPPooling(in_channels=nfilters, nfilters=nfilters, depth=psp_depth, _norm_type=norm_type,
                                 norm_groups=norm_groups)

        self.convs_up = []  # 1 argument
        self.UpCombs = []  # 2 arguments
        for idx in range(depth - 1, 0, -1):
            nheads = nheads_start * 2 ** (idx - 1)
            nfilters = nfilters_init * 2 ** (idx - 1)
            in_channels = nfilters_init * 2 ** idx
            if verbose:
                print(
                    "depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(2 * depth - idx - 1, nfilters, nheads,
                                                                                  widths[idx]))

            tnet = []
            for _ in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.append(CEEC_unit_v1(nfilters=nfilters, nheads=nheads, ngroups=nheads, norm_type=norm_type,
                                             norm_groups=norm_groups, ftdepth=ftdepth))
                elif model == 'CEECNetV2':
                    tnet.append(CEEC_unit_v2(nfilters=nfilters, nheads=nheads, ngroups=nheads, norm_type=norm_type,
                                             norm_groups=norm_groups, ftdepth=ftdepth))
                elif model == 'FracTALResNet':
                    tnet.append(
                        FracTALResNetUnit(nfilters=nfilters, nheads=nheads, ngroups=nheads, norm_type=norm_type,
                                          norm_groups=norm_groups, ftdepth=ftdepth))
                else:
                    raise ValueError(
                        "I don't know requested model, available options: CEECNetV1, CEECNetV2, FracTALResNet - Given model::{}, aborting ...".format(
                            model))
            self.convs_up.extend(tnet)

            if upFuse is True:
                self.UpCombs.append(CombineLayersWithFusion(in_channels=in_channels, nfilters=nfilters, nheads=nheads,
                                                            _norm_type=norm_type,
                                                            norm_groups=norm_groups, ftdepth=ftdepth))
            else:
                self.UpCombs.append(CombineLayers(in_channels=in_channels, _nfilters=nfilters, _norm_type=norm_type,
                                                  norm_groups=norm_groups))

        self.convs_up = nn.Sequential(*self.convs_up)
        self.UpCombs = nn.Sequential(*self.UpCombs)

    def forward(self, x):

        conv1_first = self.conv_first(x)

        # ******** Going down ***************
        fusions = []
        pools = conv1_first

        for idx in range(self.depth):
            conv1 = self.convs_dn[idx](pools)
            if idx < self.depth - 1:
                # Evaluate fusions
                fusions = fusions + [conv1]
                # Evaluate pools
                pools = self.pools[idx](conv1)

        # Middle psppooling
        middle = self.middle(conv1)
        # Activation of middle layer
        middle = F.relu(middle)
        fusions = fusions + [middle]

        # ******* Coming up ****************
        convs_up = middle
        for idx in range(self.depth - 1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx - 2])
            convs_up = self.convs_up[idx](convs_up)

        return convs_up, conv1_first


class XNetSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = XDNFeatures(in_channels=config.in_channels, nfilters_init=config.nfilters_init,
                                    depth=config.depth, widths=config.widths,
                                    psp_depth=config.psp_depth,
                                    verbose=config.verbose, norm_type=config.norm_type, norm_groups=config.norm_groups,
                                    nheads_start=config.nheads_start, model=config.model, upFuse=config.upFuse,
                                    ftdepth=config.ftdepth)
        self.head = HeadCMTSKBC(config.nfilters_init, config.nclasses, norm_type=config.norm_type,
                                norm_groups=config.norm_groups)

    def forward(self, x):
        out1, out2 = self.features(x)
        return self.head(out1, out2)

    def get_gammas(self):
        gammas = []
        for module in self.features.modules():
            if isinstance(module, CEEC_unit_v1) or isinstance(module, CEEC_unit_v2):
                gammas.append(module.gamma1.item())
            elif isinstance(module, FracTALResNetUnit):
                gammas.append(module.gamma.item())
        gammas = {"depth_{}".format(i): g for i, g in enumerate(gammas)}
        return gammas


if __name__ == "__main__":
    from utils import Dotdict

    HPP_default = Dotdict(dict(
        in_channels=1,
        nfilters_init=32,
        depth=3,
        nclasses=3,
        widths=[1],
        psp_depth=4,
        verbose=True,
        norm_type="BatchNorm",
        norm_groups=4,
        nheads_start=4,
        model="CEECNetV2",
        upFuse=False,
        ftdepth=5,
    ))

    ceecnet = XNetSegmentation(HPP_default)
    x = torch.randn(5, 1, 32, 32)
    out = ceecnet(x)

    print("In : {}\n Final seg : {} (should be [B, nclasses, H, W])\n Bound : {}\n Dist : {}".format(x.size(),
                                                                                                     out[0].size(),
                                                                                                     out[1].size(),
                                                                                                     out[2].size()))
