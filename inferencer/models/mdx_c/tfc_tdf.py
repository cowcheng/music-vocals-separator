from torch import nn


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x
