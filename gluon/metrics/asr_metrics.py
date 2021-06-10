"""
Evaluation Metrics for Automatic Speech Recognition (ASR).
"""

import mxnet as mx

__all__ = ['WER']


class WER(mx.metric.EvalMetric):
    """
    Computes Word Error Rate (WER) for Automatic Speech Recognition (ASR).

    Parameters:
    ----------
    vocabulary : list of str
        Vocabulary of the dataset.
    name : str, default 'wer'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 vocabulary,
                 name="wer",
                 output_names=None,
                 label_names=None):
        super(WER, self).__init__(
            name=name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.vocabulary = vocabulary
        self.ctc_decoder = CtcDecoder(vocabulary=vocabulary)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        import editdistance

        for labels_i, preds_i in zip(labels, preds):
            labels_code = labels_i.asnumpy()
            labels_i = []
            for label_code in labels_code:
                label_text = "".join([self.ctc_decoder.labels_map[c] for c in label_code])
                labels_i.append(label_text)

            preds_i = preds_i[0]
            greedy_predictions = preds_i.swapaxes(1, 2).log_softmax(axis=-1).argmax(axis=-1, keepdims=False).asnumpy()
            preds_i = self.ctc_decoder(greedy_predictions)

            assert (len(labels_i) == len(preds_i))
            for pred, label in zip(preds_i, labels_i):
                pred = pred.split()
                label = label.split()

                word_error_count = editdistance.eval(label, pred)
                word_count = max(len(label), len(pred))

                assert (word_error_count <= word_count)

                self.sum_metric += word_error_count
                self.global_sum_metric += word_error_count
                self.num_inst += word_count
                self.global_num_inst += word_count


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
