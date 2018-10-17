import numpy as np
import mxnet as mx
import torch
from torch.autograd import Variable

LENGTH = 64


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.bn = mx.gluon.nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                in_channels=LENGTH,
                use_global_stats=False)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        return x


class PytorchModel(torch.nn.Module):

    def __init__(self):
        super(PytorchModel, self).__init__()

        self.bn = torch.nn.BatchNorm2d(
            num_features=LENGTH,
            eps=1e-5,
            momentum=0.9)

    def forward(self, x):
        x = self.bn(x)
        return x


def main():

    success = True
    for i in range(10):
        g = np.random.randn(LENGTH, ).astype(np.float32)
        b = np.random.randn(LENGTH, ).astype(np.float32)
        m = np.random.randn(LENGTH, ).astype(np.float32)
        v = np.random.randn(LENGTH, ).astype(np.float32)
        b = b - b.min() + 1.0
        v = v - v.min() + 1.0

        IMG_SIZE = 224
        x = np.random.randn(1, LENGTH, IMG_SIZE, IMG_SIZE).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_params['bn.gamma']._load_init(mx.nd.array(g, ctx), ctx)
        gl_params['bn.beta']._load_init(mx.nd.array(b, ctx), ctx)
        gl_params['bn.running_mean']._load_init(mx.nd.array(m, ctx), ctx)
        gl_params['bn.running_var']._load_init(mx.nd.array(v, ctx), ctx)
        # gl_model.initialize()

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        pt_model = PytorchModel()
        pt_model.eval()

        pt_params = pt_model.state_dict()
        pt_params['bn.weight'] = torch.from_numpy(g)
        pt_params['bn.bias'] = torch.from_numpy(b)
        pt_params['bn.running_mean'] = torch.from_numpy(m)
        pt_params['bn.running_var'] = torch.from_numpy(v)
        pt_model.load_state_dict(pt_params)

        pt_model = pt_model.cuda()

        pt_x = Variable(torch.from_numpy(x)).cuda()
        pt_y = pt_model(pt_x).detach().cpu().numpy()

        diff = np.abs(gl_y - pt_y)
        dist = np.sum(diff)
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(pt_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
