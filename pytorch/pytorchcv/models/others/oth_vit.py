from functools import partial
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(
            in_features=dim,
            out_features=(dim * 3),
            bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(
            in_features=dim,
            out_features=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q.matmul(k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self,
                 channels,
                 mid_channels,
                 dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(channels, mid_channels)
        self.activ = nn.GELU()
        self.fc2 = nn.Linear(mid_channels, channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 qkv_bias,
                 qk_scale,
                 dropout_rate,
                 att_dropout_rate,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.att = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=att_dropout_rate,
            proj_drop=dropout_rate)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            channels=dim,
            mid_channels=mlp_hidden_dim,
            dropout_rate=dropout_rate)

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImagePatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 embedding_dim,
                 patch_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of classes for classification head
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        dropout_rate (float): dropout rate
        att_dropout_rate (float): attention dropout rate
        norm_layer: (nn.Module): normalization layer
    """

    def __init__(self,
                 in_size=(224, 224),
                 patch_size=(16, 16),
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout_rate=0.,
                 att_dropout_rate=0.,
                 norm_layer=None):
        super().__init__()
        # assert (representation_size is None)

        self.num_classes = num_classes
        self.num_features = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ImagePatchEmbedding(
            in_channels=in_channels,
            embedding_dim=embed_dim,
            patch_size=patch_size)
        num_patches = (in_size[1] // patch_size[1]) * (in_size[0] // patch_size[0])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module("block{}".format(i + 1), Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout_rate=dropout_rate,
                att_dropout_rate=att_dropout_rate,
                norm_layer=norm_layer))

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(
            in_features=self.num_features,
            out_features=num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x)[:, 0]

        x = self.head(x)
        return x


def _create_vision_transformer(variant,
                               pretrained=False,
                               **kwargs):
    net = VisionTransformer(**kwargs)
    return net


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=3."""
    model_kwargs = dict(
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3.,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        **kwargs)
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    in_size = (224, 224)
    classes = 1000

    models = [
        vit_small_patch16_224,
        vit_base_patch16_224,
        vit_large_patch16_224,
        vit_deit_tiny_patch16_224,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != vit_small_patch16_224 or weight_count == 48754408)
        assert (model != vit_base_patch16_224 or weight_count == 86567656)
        assert (model != vit_large_patch16_224 or weight_count == 304326632)
        assert (model != vit_deit_tiny_patch16_224 or weight_count == 5717416)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (batch, classes))


if __name__ == "__main__":
    _test()
