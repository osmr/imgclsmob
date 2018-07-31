from math import pi, cos
from mxnet import lr_scheduler


class LRScheduler(lr_scheduler.LRScheduler):
    r"""Learning Rate Scheduler

    For mode='step', we multiply lr with `step_factor` at each epoch in `step`.

    For mode='poly'::

        lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power

    For mode='cosine'::

        lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2

    If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.

    For warmup_mode='linear'::

        lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter

    For warmup_mode='constant'::

        lr = warmup_lr

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'step', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    n_iters : int
        Number of iterations in each epoch.
    n_epochs : int
        Number of training epochs.
    step : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    target_lr : float
        Target learning rate for poly and cosine, as the ending learning rate.
    power : float
        Power of poly function.
    warmup_epochs : int
        Number of epochs for the warmup stage.
    warmup_lr : float
        The base learning rate for the warmup stage.
    warmup_mode : str
        Modes for the warmup stage.
        Currently it supports 'linear' and 'constant'.
    """
    def __init__(self,
                 mode,
                 base_lr,
                 n_iters,
                 n_epochs,
                 step=(30, 60, 90),
                 step_factor=0.1,
                 target_lr=0,
                 power=0.9,
                 warmup_epochs=0,
                 warmup_lr=0,
                 warmup_mode='linear'):
        super(LRScheduler, self).__init__(base_lr=base_lr)
        assert(mode in ['step', 'poly', 'cosine'])
        assert(warmup_mode in ['constant', 'linear', 'poly'])

        self.mode = mode
        #self.base_lr = base_lr
        self.learning_rate = self.base_lr
        self.n_iters = n_iters

        self.step = step
        self.step_factor = step_factor
        self.target_lr = target_lr
        self.power = power

        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.warmup_mode = warmup_mode

        self.N = n_epochs * n_iters
        self.warmup_N = warmup_epochs * n_iters

    def __call__(self, num_update):
        return self.learning_rate

    def update(self, i, epoch):
        T = epoch * self.n_iters + i
        assert (T >= 0) and (T <= self.N)

        T = float(T)

        if self.warmup_epochs > epoch:
            # Warm-up Stage
            if self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            elif self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * T / self.warmup_N
            elif self.warmup_mode == 'poly':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                                     pow(T / self.warmup_N, self.power)
            else:
                raise NotImplementedError
        else:
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * pow(self.step_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * \
                                     pow(1 - (T - self.warmup_N) / (self.N - self.warmup_N), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * \
                                     (1 + cos(pi * (T - self.warmup_N) / (self.N - self.warmup_N))) / 2
            else:
                raise NotImplementedError

