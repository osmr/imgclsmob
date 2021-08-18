__all__ = ['oth_quartznet5x5_en_ls', 'oth_quartznet15x5_en', 'oth_quartznet15x5_en_nr', 'oth_quartznet15x5_fr',
           'oth_quartznet15x5_de', 'oth_quartznet15x5_it', 'oth_quartznet15x5_es', 'oth_quartznet15x5_ca',
           'oth_quartznet15x5_pl', 'oth_quartznet15x5_ru', 'oth_jasperdr10x5_en', 'oth_jasperdr10x5_en_nr',
           'oth_quartznet15x5_ru34']

import torch.nn as nn
# import torch.nn.functional as F

# import editdistance


class CtcDecoder(object):
    """
    CTC decoder (to decode a sequence of labels to words).

    Parameters:
    ----------
    vocabulary : list of str
        Vocabulary of the dataset.
    """
    def __init__(self,
                 vocabulary):
        super().__init__()
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

    def __call__(self,
                 predictions):
        """
        Decode a sequence of labels to words.

        Parameters:
        ----------
        predictions : np.array of int or list of list of int
            Tensor with predicted labels.

        Returns:
        -------
        list of str
            Words.
        """
        hypotheses = []
        for prediction in predictions:
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = "".join([self.labels_map[c] for c in decoded_prediction])
            hypotheses.append(hypothesis)
        return hypotheses


# class WER(object):
#     """
#     Word Error Rate (WER).
#
#     Parameters:
#     ----------
#     vocabulary : list of str
#         Vocabulary of the dataset.
#     """
#     def __init__(self,
#                  vocabulary):
#         super().__init__()
#         self.blank_id = len(vocabulary)
#         self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
#
#         self.scores = 0
#         self.words = 0
#
#     def update(self,
#                hypotheses,
#                references):
#         words = 0.0
#         scores = 0.0
#
#         for h, r in zip(hypotheses, references):
#             h_list = h.split()
#             r_list = r.split()
#             words += len(r_list)
#             scores += editdistance.eval(h_list, r_list)
#
#         self.scores += scores
#         self.words += words
#
#     def compute(self):
#         return float(self.scores) / self.words


class QuartzNet(nn.Module):
    def __init__(self,
                 raw_net,
                 num_classes):
        super(QuartzNet, self).__init__()
        self.in_size = None
        self.num_classes = num_classes

        self.preprocessor = raw_net.preprocessor
        self.encoder = raw_net.encoder
        self.decoder = raw_net.decoder

        # self.vocabulary = raw_net.cfg.decoder.params.vocabulary

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


# path_pref = "../../../../../imgclsmob_data/nemo/"
path_pref = "../imgclsmob_data/nemo/"


def oth_quartznet5x5_en_ls(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet5x5LS-En_08ecf82a.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_en(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet15x5Base-En_3dbcc2ff.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_en_nr(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet15x5NR-En_b05e34f3.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_fr(pretrained=False, num_classes=43, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_fr_quartznet15x5_a3fdb084.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_de(pretrained=False, num_classes=32, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_de_quartznet15x5_6ae5d87d.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_it(pretrained=False, num_classes=39, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_it_quartznet15x5_0f6e4537.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_es(pretrained=False, num_classes=36, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_es_quartznet15x5_f2083912.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_ca(pretrained=False, num_classes=39, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_ca_quartznet15x5_b1a4fa3c.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_pl(pretrained=False, num_classes=34, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_pl_quartznet15x5_9dd685f7.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_ru(pretrained=False, num_classes=35, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_ru_quartznet15x5_88a3e5aa.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_jasperdr10x5_en(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "Jasper10x5Dr-En_2b94c9d1.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_jasperdr10x5_en_nr(pretrained=False, num_classes=29, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "stt_en_jasper10x5dr_0d5ebc6c.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


def oth_quartznet15x5_ru34(pretrained=False, num_classes=34, **kwargs):
    from nemo.collections.asr.models import EncDecCTCModel
    quartznet_nemo_path = path_pref + "QuartzNet15x5_golos_1a63a2d8.nemo"
    raw_net = EncDecCTCModel.restore_from(quartznet_nemo_path)
    net = QuartzNet(raw_net=raw_net, num_classes=num_classes)
    net = net.cpu()
    return net#, raw_net


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

    pretrained = True
    audio_features = 64

    models = [
        # oth_quartznet5x5_en_ls,
        # oth_quartznet15x5_en,
        # oth_quartznet15x5_en_nr,
        # oth_quartznet15x5_fr,
        # oth_quartznet15x5_de,
        # oth_quartznet15x5_it,
        # oth_quartznet15x5_es,
        # oth_quartznet15x5_ca,
        # oth_quartznet15x5_pl,
        # oth_quartznet15x5_ru,
        # oth_jasperdr10x5_en,
        # oth_jasperdr10x5_en_nr,
        oth_quartznet15x5_ru34,
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
        assert (model != oth_quartznet15x5_it or weight_count == 18934631)
        assert (model != oth_quartznet15x5_es or weight_count == 18931556)
        assert (model != oth_quartznet15x5_ca or weight_count == 18934631)
        assert (model != oth_quartznet15x5_pl or weight_count == 18929506)
        assert (model != oth_quartznet15x5_ru or weight_count == 18930531)
        assert (model != oth_jasperdr10x5_en or weight_count == 332632349)
        assert (model != oth_jasperdr10x5_en_nr or weight_count == 332632349)
        assert (model != oth_quartznet15x5_ru34 or weight_count == 18929506)

        batch = 3
        seq_len = np.random.randint(60, 150, batch)
        seq_len_max = seq_len.max() + 2
        x = torch.randn(batch, audio_features, seq_len_max)
        x_len = torch.tensor(seq_len, dtype=torch.long, device=x.device)
        # x_len = torch.full((batch, 1), seq_len - 2).to(dtype=torch.long, device=x.device)

        y, y_len = net(x, x_len)
        # y.sum().backward()
        assert (y.size()[0] == batch)
        assert (y.size()[1] in [seq_len_max // 2, seq_len_max // 2 + 1])
        assert (y.size()[2] == num_classes)


if __name__ == "__main__":
    _test()
