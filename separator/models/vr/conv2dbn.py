from torch import nn


class Conv2DBNActiv(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)
