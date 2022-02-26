"""
Contains `span_micro_f1` used to compute the competition metric.

References
----------
https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline
"""

# System imports.
import numpy as np
import tensorflow as tf
from tf.keras.metrics import Metric
from sklearn.metrics import f1_score



def micro_f1(preds, truths):
    """Micro f1 on binary arrays.

    Arguments
    ---------
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns
    -------
        float
            f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """Converts spans to a binary array indicating whether each character is in
    the span.

    Arguments
    ---------
        spans : list of lists of two ints
            Spans.

    Returns
    -------
        binary : ndarray
            Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """Micro f1 on spans.

    Arguments
    ---------
        preds : list of lists of two ints
            Prediction spans.
        truths : list of lists of two ints
            Ground truth spans.

    Returns
    -------
        float
            f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(
            np.max(pred) if len(pred) else 0,
            np.max(truth) if len(truth) else 0
        )
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)


class F1Micro(Metric):

    def __init__(self, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives
