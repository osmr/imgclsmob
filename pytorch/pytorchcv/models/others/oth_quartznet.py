__all__ = ['oth_quartznet5x5_en_ls', 'oth_quartznet15x5_en', 'oth_quartznet15x5_en_nr', 'oth_quartznet15x5_fr',
           'oth_quartznet15x5_de', 'oth_quartznet15x5_ru', 'oth_jasperdr10x5_en', 'oth_jasperdr10x5_en_nr']

import torch.nn as nn


class QuartzNet(nn.Module):
    def __init__(self,
                 raw_net,
                 num_classes):
        super(QuartzNet, self).__init__()
        self.in_size = None
        self.num_classes = num_classes

        # self.preprocessor = raw_net.preprocessor
        self.encoder = raw_net.encoder
        self.decoder = raw_net.decoder

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, lens):
        from nemo.core import typecheck
        with typecheck.disable_checks():
            x, lens = self.encoder(x, lens)
            x = self.decoder(x)
        return x, lens


path_pref = "../../../../../imgclsmob_data/nemo/"
# path_pref = "../imgclsmob_data/nemo/"


def oth_quartznet5x5_en_ls(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet5x5LS-En_08ecf82a.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_quartznet15x5_en(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet15x5Base-En_3dbcc2ff.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_quartznet15x5_en_nr(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet15x5NR-En_b05e34f3.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_quartznet15x5_fr(pretrained=False, num_classes=43, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_fr_quartznet15x5_a3fdb084.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_quartznet15x5_de(pretrained=False, num_classes=32, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_de_quartznet15x5_6ae5d87d.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_quartznet15x5_ru(pretrained=False, num_classes=35, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_ru_quartznet15x5_88a3e5aa.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_jasperdr10x5_en(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "Jasper10x5Dr-En_2b94c9d1.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def oth_jasperdr10x5_en_nr(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_en_jasper10x5dr_0d5ebc6c.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

    pretrained = False
    audio_features = 64

    models = [
        # oth_quartznet5x5_en_ls,
        # oth_quartznet15x5_en,
        # oth_quartznet15x5_en_nr,
        # oth_quartznet15x5_fr,
        # oth_quartznet15x5_de,
        # oth_quartznet15x5_ru,
        oth_jasperdr10x5_en,
        # oth_jasperdr10x5_en_nr,
    ]

    for model in models:

        net = model(
            pretrained=pretrained)
        num_classes = net.num_classes

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_quartznet5x5_en_ls or weight_count == 6713181)
        assert (model != oth_quartznet15x5_en or weight_count == 18924381)
        assert (model != oth_quartznet15x5_en_nr or weight_count == 18924381)
        assert (model != oth_quartznet15x5_fr or weight_count == 18938731)
        assert (model != oth_quartznet15x5_de or weight_count == 18927456)
        assert (model != oth_quartznet15x5_ru or weight_count == 18930531)
        assert (model != oth_jasperdr10x5_en or weight_count == 332632349)
        assert (model != oth_jasperdr10x5_en_nr or weight_count == 332632349)

        batch = 3

        seq_len = np.random.randint(60, 150)
        # seq_len = 90
        x = torch.randn(batch, audio_features, seq_len)
        len = torch.tensor(seq_len - 2, dtype=torch.long, device=x.device).unsqueeze(dim=0)
        # len = torch.full((batch, 1), seq_len - 2).to(dtype=torch.long, device=x.device)

        y, y_len = net(x, len)
        # y.sum().backward()
        assert (y.size()[0] == batch)
        assert (y.size()[1] in [seq_len // 2, seq_len // 2 + 1])
        assert (y.size()[2] == num_classes)


if __name__ == "__main__":
    _test()
