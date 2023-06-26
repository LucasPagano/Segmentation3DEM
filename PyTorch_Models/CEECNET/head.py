import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorch_Models.CEECNET.blocks import Conv2DNormed, PSPPooling, SigmoidCrisp


class HeadSingle(nn.Module):
    # Helper classification head, for a single layer output
    def __init__(self, in_channels, nfilters, n_classes, depth=2, norm_type='BatchNorm', norm_groups=None):
        super().__init__()
        logits = []
        for _ in range(depth):
            logits.append(
                Conv2DNormed(in_channels=in_channels, channels=nfilters, kernel_size=(3, 3), padding=(1, 1),
                             _norm_type=norm_type,
                             norm_groups=norm_groups))
            logits.append(nn.ReLU())
            # only first convolution needs tp project from in_channels -> nfilters
            in_channels = nfilters
        logits.append(nn.Conv2d(in_channels=nfilters, out_channels=n_classes, kernel_size=1, padding=0))
        self.logits = nn.Sequential(*logits)

    def forward(self, x):
        return self.logits(x)  # [B, nfilters, H, W] -> [B, nclasses, H, W]


class HeadCMTSKBC(nn.Module):
    # BC: Balanced (features) Crisp (boundaries)
    def __init__(self, _nfilters_init, n_classes, norm_type='BatchNorm', norm_groups=None):
        super().__init__()

        self.model_name = "Head_CMTSK_BC"

        self.nfilters = _nfilters_init  # Initial number of filters
        self.nclasses = n_classes

        self.psp_2ndlast = PSPPooling(in_channels=self.nfilters * 2, nfilters=self.nfilters, _norm_type=norm_type,
                                      norm_groups=norm_groups)

        # bound logits
        self.bound_logits = HeadSingle(in_channels=self.nfilters * 2, nfilters=self.nfilters, n_classes=self.nclasses,
                                       norm_type=norm_type, norm_groups=norm_groups)
        self.bound_Equalizer = Conv2DNormed(in_channels=self.nclasses, channels=self.nfilters, kernel_size=1,
                                            _norm_type=norm_type,
                                            norm_groups=norm_groups)

        # distance logits -- deeper for better reconstruction
        self.distance_logits = HeadSingle(in_channels=self.nfilters * 2, nfilters=self.nfilters,
                                          n_classes=self.nclasses, norm_type=norm_type,
                                          norm_groups=norm_groups)
        self.dist_Equalizer = Conv2DNormed(in_channels=self.nclasses, channels=self.nfilters, kernel_size=1,
                                           _norm_type=norm_type,
                                           norm_groups=norm_groups)

        self.Comb_bound_dist = Conv2DNormed(in_channels=self.nfilters * 2, channels=self.nfilters, kernel_size=1,
                                            _norm_type=norm_type,
                                            norm_groups=norm_groups)

        # Segmenetation logits -- deeper for better reconstruction
        self.final_segm_logits = HeadSingle(in_channels=self.nfilters * 2, nfilters=self.nfilters, n_classes=self.nclasses,
                                            norm_type=norm_type,
                                            norm_groups=norm_groups)

        self.CrispSigm = SigmoidCrisp()

        # Last activation, customization for binary results
        if self.nclasses == 1:
            self.ChannelAct = nn.Sigmoid()
        else:
            self.ChannelAct = nn.Softmax(dim=1)

    def forward(self, upconv4, conv1):

        # second last layer
        convl = torch.cat((conv1, upconv4), dim=1)
        conv = self.psp_2ndlast(convl)
        conv = F.relu(conv)

        # logits
        # 1st find distance map, skeleton like, topology info
        dist = self.distance_logits(convl)  # do not use max pooling for distance
        dist = self.ChannelAct(dist)
        # makes nfilters equals to conv and convl
        dist_eq = F.relu(self.dist_Equalizer(dist))  # [B, nfilters, H, W]

        # Then find boundaries
        bound = torch.cat((conv, dist_eq), dim=1)
        bound = self.bound_logits(bound)
        bound = self.CrispSigm(bound)  # Boundaries are not mutually exclusive
        bound_eq = F.relu(self.bound_Equalizer(bound))  # [B, nfilters, H, W]

        # Now combine all predictions in a final segmentation mask
        # Balance first boundary and distance transform, with the features
        comb_bd = self.Comb_bound_dist(torch.cat((bound_eq, dist_eq), dim=1))
        comb_bd = F.relu(comb_bd)

        all_layers = torch.cat((comb_bd, conv), dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)

        return final_segm, bound, dist
