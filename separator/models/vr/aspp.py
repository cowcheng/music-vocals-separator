import torch
import torch.nn.functional as F
from torch import nn

from separator.models.vr.conv2dbn import Conv2DBNActiv


class ASPPModule(nn.Module):

    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
