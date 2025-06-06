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

    if axis is None:
        axis = input.ndim - 1
    if axis < 0:
        axis = input.ndim + axis

    C = input.shape[axis]

    if weight is not None:
        weight_shape = (
            (1,) * axis + (input.shape[axis],) + (1,) * (input.ndim - axis - 1)
        )
        weight = weight.reshape(weight_shape)

    if isinstance(target, int) or target.ndim != input.ndim:
        no_ignore = jax.lax.stop_gradient(target != ignore_index)
        logits_max = jnp.max(
            input, axis=axis, keepdims=True
        )  # , where=no_ignore, initial=-jnp.inf)
        logits = input - jax.lax.stop_gradient(logits_max)

        broadcast_shape = logits.shape[:axis] + (1,) + logits.shape[axis + 1 :]

        log_normalizers = jax.nn.logsumexp(
            logits, b=no_ignore.reshape(broadcast_shape), axis=axis
        )

        labels_no_ignore = jnp.where(no_ignore, target, 0)

        label_logits = jnp.take_along_axis(
            logits, labels_no_ignore[..., None], axis=axis
        )[..., 0]

        if label_smoothing != 0 or weight is not None:
            one_hot_labels = jax.nn.one_hot(labels_no_ignore, num_classes=C, axis=axis)
            target_probs = (
                one_hot_labels * (1.0 - label_smoothing)
                + jnp.ones_like(one_hot_labels) / C * label_smoothing
            )

            if weight is not None:
                target_probs = target_probs * weight
                log_normalizers = log_normalizers * jnp.sum(target_probs, axis=axis)
                target_normalizer = jnp.sum(
                    target_probs, where=no_ignore.reshape(broadcast_shape)
                )

            losses = -(
                jnp.sum(
                    target_probs * logits,
                    where=no_ignore.reshape(broadcast_shape),
                    axis=axis,
                )
                - log_normalizers
            )
        else:
            label_logits = jnp.take_along_axis(
                logits, labels_no_ignore.reshape(broadcast_shape), axis=axis
            )
            label_logits = label_logits.reshape(labels_no_ignore.shape)
            losses = log_normalizers - label_logits

        losses = jnp.where(no_ignore, losses, 0.0)
    else:
        target_probs = (
            target * (1.0 - label_smoothing)
            + jnp.ones_like(target) / C * label_smoothing
        )

        logits_max = jnp.max(input, axis=axis, keepdims=True)
        logits = input - jax.lax.stop_gradient(logits_max)

        log_normalizers = jax.nn.logsumexp(logits, axis=axis)

        if weight is not None:
            target_probs = target_probs * weight
            log_normalizers = log_normalizers * jnp.sum(
                target_probs * weight, axis=axis
            )
            target_normalizer = jnp.sum(target_probs)

        losses = -(jnp.sum(target_probs * logits, axis=axis) - log_normalizers)

        no_ignore = None

    if reduction == "none":
        return losses
    if reduction == "mean":
        if weight is None:
            return jnp.mean(losses, where=no_ignore)
        else:
            return jnp.sum(losses, where=no_ignore) / target_normalizer
    if reduction == "sum":
        return jnp.sum(losses, where=no_ignore)
