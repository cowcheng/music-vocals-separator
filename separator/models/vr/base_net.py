import torch
from torch import nn

from separator.models.vr.aspp import ASPPModule
from separator.models.vr.conv2dbn import Conv2DBNActiv
from separator.models.vr.decoder import Decoder
from separator.models.vr.encoder import Encoder
from separator.models.vr.lstm import LSTMModule


class BaseNet(nn.Module):

    def __init__(
        self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))
    ):
        super(BaseNet, self).__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h
