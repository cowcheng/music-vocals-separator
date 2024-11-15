from torch import nn


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.conv(x)
