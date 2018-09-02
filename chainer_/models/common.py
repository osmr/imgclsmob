from chainer import Chain

__all__ = ['SimpleSequential']


class SimpleSequential(Chain):

    def __init__(self):
        super(SimpleSequential, self).__init__()
        self.layer_names = []

    def __setattr__(self, name, value):
        super(SimpleSequential, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        super(SimpleSequential, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    def __call__(self, x):
        for name in self.layer_names:
            x = self[name](x)
        return x

