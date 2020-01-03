import tensorflow as tf
import logging

import tensorflow as tf
import tensorflow_addons as tfa


def masked_categorical_crossentropy(y_true,
                                    y_pred,
                                    y_mask,
                                    from_logits=False,
                                    label_smoothing=0):
    with tf.name_scope("masked_categorical_crossentropy"):
        batch_size = tf.shape(y_true)[0]
        if y_mask.dtype != tf.float32:
            y_mask = tf.cast(y_mask, tf.float32)

        if label_smoothing > 0:
            num_masked = tf.reduce_sum(y_mask, axis=1)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = tf.expand_dims(label_smoothing / (num_masked + tf.keras.backend.epsilon()), axis=-1)
            onehot_answer = y_true * smooth_positives + smooth_negatives
            masked_onehot_answer = onehot_answer * y_mask
        else:
            masked_onehot_answer = y_true * y_mask

        loss = tf.keras.losses.categorical_crossentropy(
            masked_onehot_answer, y_pred, from_logits, label_smoothing=0)

    return loss


def softmax_sequence_reconstruction_error(decoder_softmax, answer, answer_length,
                                          average=True, average_batch=False,
                                          smoothing_rate=0.0, vocab_size=30000):
    """
    Sequential cross-entropy loss for softmax output.
    Only use this if you cannot get logits, otherwise use `tfa.seq2seq.SequenceLoss`
    (It is more optimized and numerically stable)
    """
    with tf.name_scope("softmax_sequence_reconstruction_error"):
        onehot_answer = tf.one_hot(answer, vocab_size)
        if smoothing_rate > 0:
            smooth_positives = 1.0 - smoothing_rate
            smooth_negatives = smoothing_rate / vocab_size
            onehot_answer = onehot_answer * smooth_positives + smooth_negatives
        xentropy = tf.keras.losses.categorical_crossentropy(onehot_answer, decoder_softmax)
        answer_mask = tf.sequence_mask(answer_length, dtype=decoder_softmax.dtype)
        if average:
            if average_batch:
                reconstruction_error = tf.reduce_sum(xentropy * answer_mask) / tf.cast(tf.reduce_sum(answer_mask), tf.float32)
            else:
                reconstruction_error = tf.reduce_sum(xentropy * answer_mask, axis=1) / tf.cast(answer_length, tf.float32)
        else:
            reconstruction_error = tf.reduce_sum(xentropy * answer_mask, axis=1)
            if average_batch:
                reconstruction_error = tf.reduce_mean(reconstruction_error)
    return reconstruction_error


def softmax_kl_divergence(prior, posterior, masking):
    # XXX : use softplus? rather than epsilon?
    with tf.name_scope("softmax_kl_divergence"):
        if masking.dtype == tf.bool:
            masking = tf.cast(masking, tf.float32)
        kld = posterior * tf.math.log((posterior + tf.keras.backend.epsilon()) / (prior + tf.keras.backend.epsilon()))
        kld = tf.reduce_sum(kld * masking, axis=1)
    return kld


def sequence_loss_ls(logits,
                     targets,
                     weights,
                     average_across_timesteps=True,
                     average_across_batch=True,
                     sum_over_timesteps=False,
                     sum_over_batch=False,
                     label_smoothing=0.,
                     softmax_loss_function=None,
                     name=None):
    """Label smoothed sequence_loss of tfa.seq2seq.sequence_loss

    Most of this code is from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/seq2seq/loss.py#L24
    """
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                         "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError(
            "Targets must be a [batch_size x sequence_length] tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError(
            "Weights must be a [batch_size x sequence_length] tensor")
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError(
            "average_across_timesteps and sum_over_timesteps cannot "
            "be set to True at same time.")
    if average_across_batch and sum_over_batch:
        raise ValueError(
            "average_across_batch and sum_over_batch cannot be set "
            "to True at same time.")
    if average_across_batch and sum_over_timesteps:
        raise ValueError(
            "average_across_batch and sum_over_timesteps cannot be set "
            "to True at same time because of ambiguous order.")
    if sum_over_batch and average_across_timesteps:
        raise ValueError(
            "sum_over_batch and average_across_timesteps cannot be set "
            "to True at same time because of ambiguous order.")
    with tf.name_scope(name or "sequence_loss"):
        num_classes = tf.shape(input=logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        if softmax_loss_function is None:
            if label_smoothing > 0:
                onehot_targets = tf.one_hot(targets, num_classes)
                crossent = tf.keras.losses.categorical_crossentropy(
                    onehot_targets, logits_flat, from_logits=True,
                    label_smoothing=label_smoothing)
            else:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_flat)
        else:
            crossent = softmax_loss_function(
                labels=targets, logits=logits_flat)
        crossent *= tf.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_size = tf.reduce_sum(input_tensor=weights)
            crossent = tf.math.divide_no_nan(crossent, total_size)
        elif sum_over_timesteps and sum_over_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_count = tf.cast(
                tf.math.count_nonzero(weights), crossent.dtype)
            crossent = tf.math.divide_no_nan(crossent, total_count)
        else:
            crossent = tf.reshape(crossent, tf.shape(input=logits)[0:2])
            if average_across_timesteps or average_across_batch:
                reduce_axis = [0] if average_across_batch else [1]
                crossent = tf.reduce_sum(
                    input_tensor=crossent, axis=reduce_axis)
                total_size = tf.reduce_sum(
                    input_tensor=weights, axis=reduce_axis)
                crossent = tf.math.divide_no_nan(crossent, total_size)
            elif sum_over_timesteps or sum_over_batch:
                reduce_axis = [0] if sum_over_batch else [1]
                crossent = tf.reduce_sum(
                    input_tensor=crossent, axis=reduce_axis)
                total_count = tf.cast(
                    tf.math.count_nonzero(weights, axis=reduce_axis),
                    dtype=crossent.dtype)
                crossent = tf.math.divide_no_nan(crossent, total_count)
        return crossent


class SequenceLossLS(tfa.seq2seq.SequenceLoss):
    def __init__(self,
                 *args,
                 label_smoothing=0.,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def __call__(self, y_true, y_pred, sample_weight=None):
        return sequence_loss_ls(
            y_pred,
            y_true,
            sample_weight,
            average_across_timesteps=self.average_across_timesteps,
            average_across_batch=self.average_across_batch,
            sum_over_timesteps=self.sum_over_timesteps,
            sum_over_batch=self.sum_over_batch,
            label_smoothing=self.label_smoothing,
            softmax_loss_function=self.softmax_loss_function,
            name=self.name)
