import jax
from jax import numpy as jnp


def softmax_cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    axis=None,
):
    """Computes the cross entropy loss between input logits and target labels.

    This function mimics the functionality of torch.nn.functional.cross_entropy,
    combining softmax and cross entropy in a numerically stable way.

    Args:
        input (jnp.ndarray): Input tensor containing raw, unnormalized logits.
            Shape: (*), where * means any number of dimensions. The axis dimension
            should contain class logits.
        target (jnp.ndarray or int): Target tensor containing class indices or
            class probabilities. If integer indices, shape should be (*) where
            each value is in [0, C) where C is the number of classes. If
            probabilities, shape should match input.
        weight (jnp.ndarray, optional): Manual rescaling weight given to each
            class. If provided, should be a 1D tensor of size C (number of classes).
            Default: None.
        ignore_index (int): Specifies a target value that is ignored and does not
            contribute to the input gradient. Default: -100.
        reduction (str): Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the weighted mean of the output is taken
            'sum': the output will be summed
            Default: 'mean'.
        label_smoothing (float): A float in [0.0, 1.0]. Specifies the amount of
            smoothing when computing the loss, where 0.0 means no smoothing.
            Default: 0.0.
        axis (int, optional): Dimension along which softmax is computed. If None,
            defaults to the last dimension. Default: None.

    Returns:
        jnp.ndarray: The computed cross entropy loss. If reduction is 'none',
            returns a tensor of the same shape as target. Otherwise returns
            a scalar tensor.

    Note:
        This function is numerically stable and avoids computing the full softmax
        by using the log-sum-exp trick and only computing probabilities for the
        target classes when possible.

        When target contains class indices, the function supports:
        - Ignoring specific indices via ignore_index
        - Label smoothing
        - Per-class weighting

        When target contains class probabilities, the function computes the
        cross entropy between the softmax of input and the target distribution.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            f"reduction must be one of 'none', 'mean', or 'sum'; got {reduction!r}"
        )
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(
            f"label_smoothing must be between 0 and 1; got {label_smoothing}"
        )

    if axis is None:
        axis = input.ndim - 1
    if axis < 0:
        axis = input.ndim + axis

    num_classes = input.shape[axis]

    class_weight = None if weight is None else jnp.asarray(weight)
    if class_weight is not None:
        weight_shape = (
            (1,) * axis + (input.shape[axis],) + (1,) * (input.ndim - axis - 1)
        )
        weight = class_weight.reshape(weight_shape)

    target = jnp.asarray(target)

    if target.ndim != input.ndim:
        no_ignore = jax.lax.stop_gradient(target != ignore_index)
        logits_max = jnp.max(input, axis=axis, keepdims=True)
        logits = input - jax.lax.stop_gradient(logits_max)
        log_normalizers = jax.nn.logsumexp(logits, axis=axis)

        labels_no_ignore = jnp.where(no_ignore, target, 0)
        label_logits = jnp.take_along_axis(
            logits,
            jnp.expand_dims(labels_no_ignore, axis=axis),
            axis=axis,
        ).squeeze(axis=axis)
        negative_log_likelihood = log_normalizers - label_logits

        if class_weight is None:
            if label_smoothing == 0.0:
                losses = negative_log_likelihood
            else:
                smooth_losses = log_normalizers - jnp.mean(logits, axis=axis)
                losses = (1.0 - label_smoothing) * negative_log_likelihood
                losses = losses + label_smoothing * smooth_losses
        else:
            target_weights = jnp.take(class_weight, labels_no_ignore)
            if label_smoothing == 0.0:
                losses = target_weights * negative_log_likelihood
            else:
                smooth_losses = (
                    log_normalizers * jnp.sum(class_weight)
                    - jnp.sum(logits * weight, axis=axis)
                ) / num_classes
                losses = (
                    (1.0 - label_smoothing) * target_weights * negative_log_likelihood
                    + label_smoothing * smooth_losses
                )
            target_normalizer = jnp.sum(
                target_weights,
                where=no_ignore,
            )

        losses = jnp.where(no_ignore, losses, 0.0)
    else:
        target_probs = (
            target * (1.0 - label_smoothing)
            + jnp.ones_like(target) / num_classes * label_smoothing
        )

        logits_max = jnp.max(input, axis=axis, keepdims=True)
        logits = input - jax.lax.stop_gradient(logits_max)

        log_normalizers = jax.nn.logsumexp(logits, axis=axis)

        if class_weight is not None:
            target_probs = target_probs * weight

        losses = log_normalizers * jnp.sum(target_probs, axis=axis) - jnp.sum(
            target_probs * logits, axis=axis
        )

        no_ignore = None

    if reduction == "none":
        return losses
    if reduction == "mean":
        if target.ndim == input.ndim:
            return jnp.mean(losses)
        if class_weight is None:
            return jnp.mean(losses, where=no_ignore)
        return jnp.sum(losses, where=no_ignore) / target_normalizer
    return jnp.sum(losses, where=no_ignore)
