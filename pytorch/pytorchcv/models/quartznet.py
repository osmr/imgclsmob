"""
    QuartzNet for ASR, implemented in PyTorch.
    Original paper: 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.
"""

__all__ = ['quartznet5x5_en_ls', 'quartznet15x5_en', 'quartznet15x5_en_nr', 'quartznet15x5_fr', 'quartznet15x5_de',
           'quartznet15x5_ru']

import os
from .jasper import Jasper


def get_quartznet(version,
                  bn_eps=1e-3,
                  dropout_rate=0.0,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create QuartzNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Model version.
    bn_eps : float, default 1e-3
        Small float added to variance in Batch norm.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    import numpy as np

    blocks, repeat = tuple(map(int, version.split("x")))
    main_stage_repeat = blocks // 5

    channels_per_stage = [256, 256, 256, 512, 512, 512, 512, 1024]
    kernel_sizes_per_stage = [33, 33, 39, 51, 63, 75, 87, 1]
    dropout_rates_per_stage = [dropout_rate] * 8
    stage_repeat = np.full((8,), 1)
    stage_repeat[1:-2] *= main_stage_repeat
    channels = sum([[a] * r for (a, r) in zip(channels_per_stage, stage_repeat)], [])
    kernel_sizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])
    use_dw = True

    net = Jasper(
        channels=channels,
        kernel_sizes=kernel_sizes,
        bn_eps=bn_eps,
        dropout_rates=dropout_rates,
        repeat=repeat,
        use_dw=use_dw,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def quartznet5x5_en_ls(num_classes=29, **kwargs):
    """
    QuartzNet 15x5 model for English language (trained on LibriSpeech dataset) from 'QuartzNet: Deep Automatic Speech
    Recognition with 1D Time-Channel Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="5x5", model_name="quartznet5x5_en_ls", **kwargs)


def quartznet15x5_en(num_classes=29, **kwargs):
    """
    QuartzNet 15x5 model for English language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="15x5", model_name="quartznet15x5_en", **kwargs)


def quartznet15x5_en_nr(num_classes=29, **kwargs):
    """
    QuartzNet 15x5 model for English language (with presence of noise) from 'QuartzNet: Deep Automatic Speech
    Recognition with 1D Time-Channel Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="15x5", model_name="quartznet15x5_en_nr", **kwargs)


def quartznet15x5_fr(num_classes=43, **kwargs):
    """
    QuartzNet 15x5 model for French language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="15x5", model_name="quartznet15x5_fr", **kwargs)


def quartznet15x5_de(num_classes=32, **kwargs):
    """
    QuartzNet 15x5 model for German language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="15x5", model_name="quartznet15x5_de", **kwargs)


def quartznet15x5_ru(num_classes=35, **kwargs):
    """
    QuartzNet 15x5 model for Russian language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 35
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_quartznet(num_classes=num_classes, version="15x5", model_name="quartznet15x5_ru", **kwargs)


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
        quartznet5x5_en_ls,
        quartznet15x5_en,
        quartznet15x5_en_nr,
        quartznet15x5_fr,
        quartznet15x5_de,
        quartznet15x5_ru,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != quartznet5x5_en_ls or weight_count == 6713181)
        assert (model != quartznet15x5_en or weight_count == 18924381)
        assert (model != quartznet15x5_en_nr or weight_count == 18924381)
        assert (model != quartznet15x5_fr or weight_count == 18938731)
        assert (model != quartznet15x5_de or weight_count == 18927456)
        assert (model != quartznet15x5_ru or weight_count == 18930531)

        batch = 1
        seq_len = np.random.randint(60, 150)
        seq_len = 90
        x = torch.randn(batch, audio_features, seq_len)
        x_len = torch.tensor(seq_len - 2, dtype=torch.long, device=x.device).unsqueeze(dim=0)

        # net.load_state_dict(torch.load("/home/osemery/projects/imgclsmob_data/nemo/quartznet15x5_ru.pth"))

        # x = torch.from_numpy(np.load("/home/osemery/projects/imgclsmob_data/test/x_qn.npy"))
        # x_len = torch.from_numpy(np.load("/home/osemery/projects/imgclsmob_data/test/xl_qn.npy"))
        # y1 = torch.from_numpy(np.load("/home/osemery/projects/imgclsmob_data/test/y_qn.npy"))
        # y1_len = torch.from_numpy(np.load("/home/osemery/projects/imgclsmob_data/test/yl_qn.npy"))

        y, y_len = net(x, x_len)
        # y.sum().backward()
        assert (tuple(y.size())[:2] == (batch, net.num_classes))
        assert (y.size()[2] in [seq_len // 2, seq_len // 2 + 1])


if __name__ == "__main__":
    _test()
