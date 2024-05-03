"""
    Loss functions.
"""

__all__ = ['SegSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyLoss']

from mxnet.gluon.loss import Loss, _reshape_like


class SegSoftmaxCrossEntropyLoss(Loss):
    """
    SoftmaxCrossEntropyLoss with ignore labels (for segmentation task).

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    """
    def __init__(self,
                 sparse_label=True,
                 batch_axis=0,
                 ignore_label=-1,
                 size_average=True,
                 **kwargs):
        super(SegSoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        """
        Compute loss.
        """
        softmaxout = F.SoftmaxOutput(
            pred,
            label.astype(pred.dtype),
            ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True,
            normalization=("valid" if self._size_average else "null"))
        if self._sparse_label:
            loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(F.log(softmaxout) * label, axis=-1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label, F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class MixSoftmaxCrossEntropyLoss(SegSoftmaxCrossEntropyLoss):
    """
    SegSoftmaxCrossEntropyLoss with auxiliary loss support.

    Parameters
    ----------
    aux : bool, default True
        Whether to use auxiliary loss.
    aux_weight : float, default 0.2
        The weight for aux loss.
    ignore_label : int, default -1
        The label to ignore.
    """
    def __init__(self,
                 aux=True,
                 aux_weight=0.2,
                 ignore_label=-1,
                 **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, F, pred1, pred2, label):
        """
        Compute loss including auxiliary output.
        """
        loss1 = super(MixSoftmaxCrossEntropyLoss, self).hybrid_forward(F, pred1, label)
        loss2 = super(MixSoftmaxCrossEntropyLoss, self). hybrid_forward(F, pred2, label)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward(self, F, preds, label, **kwargs):
        """
        Compute loss.
        """
        if self.aux:
            return self._aux_forward(F, preds[0], preds[1], label)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).hybrid_forward(F, preds, label)
