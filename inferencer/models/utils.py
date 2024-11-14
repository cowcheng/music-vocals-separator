from functools import partial

from torch import nn


def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == "BatchNorm":
            return nn.BatchNorm2d(c)
        elif norm_type == "InstanceNorm":
            return nn.InstanceNorm2d(c, affine=True)
        elif "GroupNorm" in norm_type:
            g = int(norm_type.replace("GroupNorm", ""))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


def get_act(act_type):
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type[:3] == "elu":
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    else:
        raise Exception
