import torch
import torch.nn.functional as F
from torch import nn

from separator.models.vr import utils
from separator.models.vr.conv2dbn import Conv2DBNActiv


class Decoder(nn.Module):

    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            skip = utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv1(x)
        # h = self.conv2(h)

        if self.dropout is not None:
            h = self.dropout(h)

        return h
