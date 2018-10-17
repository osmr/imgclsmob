import numpy as np
import mxnet as mx
import torch
from torch.autograd import Variable


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = mx.gluon.nn.Conv2D(
                channels=64,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=True,
                in_channels=3)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


class PytorchModel(torch.nn.Module):

    def __init__(self):
        super(PytorchModel, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


def main():

    success = True
    for i in range(10):
        # w = np.random.randint(10, size=(64, 3, 7, 7)).astype(np.float32)
        # x = np.random.randint(10, size=(1, 3, 224, 224)).astype(np.float32)
        w = np.random.randn(64, 3, 7, 7).astype(np.float32)
        b = np.random.randn(64, ).astype(np.float32)
        x = np.random.randn(10, 3, 224, 224).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_params['conv.weight']._load_init(mx.nd.array(w, ctx), ctx)
        gl_params['conv.bias']._load_init(mx.nd.array(b, ctx), ctx)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        pt_model = PytorchModel()
        pt_model.eval()

        pt_params = pt_model.state_dict()
        pt_params['conv.weight'] = torch.from_numpy(w)
        pt_params['conv.bias'] = torch.from_numpy(b)
        pt_model.load_state_dict(pt_params)

        pt_model = pt_model.cuda()

        pt_x = Variable(torch.from_numpy(x)).cuda()
        pt_y = pt_model(pt_x).detach().cpu().numpy()

        dist = np.sum(np.abs(gl_y - pt_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
