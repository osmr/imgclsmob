import numpy as np
import mxnet as mx
import torch
from torch.autograd import Variable


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.dense = mx.gluon.nn.Dense(
                units=1000,
                use_bias=False,
                in_units=1024)

    def hybrid_forward(self, F, x):
        x = self.dense(x)
        return x


class PytorchModel(torch.nn.Module):

    def __init__(self):
        super(PytorchModel, self).__init__()

        self.dense = torch.nn.Linear(
            in_features=1024,
            out_features=1000,
            bias=False)

    def forward(self, x):
        x = self.dense(x)
        return x


def main():

    success = True
    for i in range(10):
        w = np.random.randn(1000, 1024).astype(np.float32)
        # b = np.random.randn(1000, ).astype(np.float32)
        x = np.random.randn(1, 1024).astype(np.float32)

        gl_model = GluonModel()

        ctx = mx.cpu()
        gl_params = gl_model._collect_params_with_prefix()
        gl_params['dense.weight']._load_init(mx.nd.array(w, ctx), ctx)
        # gl_params['dense.bias']._load_init(mx.nd.array(b, ctx), ctx)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        pt_model = PytorchModel()

        pt_params = pt_model.state_dict()
        pt_params['dense.weight'] = torch.from_numpy(w)
        # pt_params['dense.bias'] = torch.from_numpy(b)
        pt_model.load_state_dict(pt_params)

        pt_x = Variable(torch.from_numpy(x))
        pt_y = pt_model(pt_x).detach().numpy()

        dist = np.sum(np.abs(gl_y - pt_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(pt_y)
            y = np.matmul(w.astype(np.float64), x[0].astype(np.float64))
            # y = np.dot(w, x[0])
            gl_dist = np.sum(np.abs(gl_y - y))
            pt_dist = np.sum(np.abs(pt_y - y))
            print("i={}, gl_dist={}".format(i, gl_dist))
            print("i={}, pt_dist={}".format(i, pt_dist))

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
