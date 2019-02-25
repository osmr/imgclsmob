""" Fractal Model - per sample drop path """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu_(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class FractalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_columns,
                 p_ldrop,
                 dropout_rate):
        """ Fractal block
        Args:
            - n_columns: # of columns
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
        """
        super(FractalBlock, self).__init__()

        self.n_columns = n_columns
        self.p_ldrop = p_ldrop

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns - 1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    first_block = (i + 1 == dist)  # first block in this column
                    if first_block:
                        cur_in_channels = in_channels
                    else:
                        cur_in_channels = out_channels

                    module = ConvBlock(
                        in_channels=cur_in_channels,
                        out_channels=out_channels,
                        dropout_rate=dropout_rate)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

        pass

    def drop_mask(self, B, global_cols, n_cols):
        """ Generate drop mask; [n_cols, B].
        1) generate global masks
        2) generate local masks
        3) resurrect random path in all-dead column
        4) concat global and local masks

        Args:
            - B: batch_size
            - global_cols: global columns which to alive [GB]
            - n_cols: the number of columns of mask
        """
        # global drop mask
        GB = global_cols.shape[0]
        # calc gdrop cols / samples
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]
        # gen gdrop mask
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.

        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.-self.p_ldrop, [n_cols, LB]).astype(np.float32)
        alive_count = ldrop_mask.sum(axis=0)
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices] = 1.

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    def join(self, outs, global_cols):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        n_cols = len(outs)
        out = torch.stack(outs) # [n_cols, B, C, H, W]

        if self.training:
            mask = self.drop_mask(out.size(1), global_cols, n_cols).to(out.device) # [n_cols, B]
            n_cols, B = mask.size()
            mask = mask.view(n_cols, B, 1, 1, 1)  # unsqueeze to [n_cols, B, 1, 1, 1]
            n_alive = mask.sum(dim=0) # [B, 1, 1, 1]
            masked_out = out * mask # [n_cols, B, C, H, W]
            n_alive[n_alive == 0.] = 1. # all-dead cases
            out = masked_out.sum(dim=0) / n_alive # [B, C, H, W] / [B, 1, 1, 1]
        else:
            out = out.mean(dim=0)  # no drop

        return out

    def forward(self, x, global_cols):
        """
        global_cols works only in training mode.
        """
        out = x
        outs = [out] * self.n_columns
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = []  # outs of current depth

            for c in range(st, self.n_columns):
                cur_in = outs[c]  # current input
                cur_module = self.columns[c][i]  # current module
                cur_outs.append(cur_module(cur_in))

            # join
            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined

        return outs[-1]  # for deepest case


class FractalNet(nn.Module):
    def __init__(self,
                 channels,
                 dropout_probs,
                 n_columns,
                 p_ldrop,
                 gdrop_ratio,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        """ FractalNet
        Args:
            - n_columns: the number of columns
            - init_block_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - doubling: if True, doubling by 1x1 conv in front of the block.
        """
        super(FractalNet, self).__init__()
        self.gdrop_ratio = gdrop_ratio
        self.n_columns = n_columns

        layers = nn.ModuleList()
        for i, out_channels in enumerate(channels):
            p_dropout = dropout_probs[i]
            layers.append(FractalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                n_columns=n_columns,
                p_ldrop=p_ldrop,
                dropout_rate=p_dropout))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        layers.append(Flatten())
        layers.append(nn.Linear(out_channels, num_classes))  # fc layer

        self.layers = layers

    def forward(self, x):
        GB = int(x.size(0) * self.gdrop_ratio)
        global_cols = np.random.randint(0, self.n_columns, size=[GB])

        out = x
        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                out = layer(out, global_cols)
            else:
                out = layer(out)

        return out


def oth_fractalnet_cifar10(pretrained=False, **kwargs):

    dropout_probs = (0.0, 0.1, 0.2, 0.3, 0.4)
    channels = [64 * (2 ** (i if i != len(dropout_probs) - 1 else i - 1)) for i in range(len(dropout_probs))]

    model = FractalNet(
        channels=channels,
        dropout_probs=dropout_probs,
        n_columns=3,
        p_ldrop=0.15,
        gdrop_ratio=0.5,
        num_classes=10)
    return model


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_fractalnet_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_fractalnet_cifar10 or weight_count == 33724618)

        x = Variable(torch.randn(14, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (14, 10))


if __name__ == "__main__":
    _test()
