from torch import nn

from separator.models.vr.conv2dbn import Conv2DBNActiv


class LSTMModule(nn.Module):

    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        h = h.reshape(nframes, N, 1, nbins)
        h = h.permute(1, 2, 3, 0)

        return h
