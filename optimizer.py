from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools

import six

from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


class MyOptimizerV2(optimizer_v2.OptimizerV2):
    def __init__(self, name, **kwargs):
        super(MyOptimizerV2, self).__init__(name, **kwargs)

    def _distributed_apply(self, distribution, grads_and_vars, name):
        """`apply_gradients` using a `DistributionStrategy`."""
        reduced_grads = distribution.extended.batch_reduce_to(
            ds_reduce_util.ReduceOp.SUM, grads_and_vars)
        var_list = [v for _, v in grads_and_vars]

        # grads_and_vars = zip(reduced_grads, var_list)
        grads_and_vars = zip([g + (v * WEIGHT_DECAY) for g, v in grads_and_vars], var_list)

        def apply_grad_to_update_var(var, grad):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable.")
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices)
            update_op = self._resource_apply_dense(grad, var)
            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        update_ops = []
        with backend.name_scope(name or self._name):
            for grad, var in grads_and_vars:
                scope_name = ("" if ops.executing_eagerly_outside_functions() else
                              "_" + var.op.name)
                with backend.name_scope("update" + scope_name):
                    update_ops.extend(
                        distribution.extended.update(
                            var, apply_grad_to_update_var, args=(grad,), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1).op

            return self._iterations.assign_add(1)


class SGD(MyOptimizerV2):
    """Stochastic gradient descent and momentum optimizer.

    Computes:
    ```
    theta(t+1) = theta(t) - learning_rate * gradient
    gradient is evaluated at theta(t).
    ```

    or Computes (if `nesterov = False`):
    ```
    v(t+1) = momentum * v(t) - learning_rate * gradient
    theta(t+1) = theta(t) + v(t+1)
    if `nesterov` is False, gradient is evaluated at theta(t).
    if `nesterov` is True, gradient is evaluated at theta(t) + momentum * v(t),
      and the variables always store theta + m v instead of theta
    ```

    Some of the args below are hyperparameters, where a hyperparameter is
    defined as a scalar Tensor, a regular Python value, or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    @compatibility(eager)
    When eager execution is enabled, learning_rate can be a callable that takes
    no arguments and returns the actual value to use. This can be useful for
    changing these values across different invocations of optimizer functions.
    @end_compatibility

    # References
        nesterov = True, See [Sutskever et al., 2013](
          http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
    """

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 name="SGD",
                 **kwargs):
        """Construct a new Stochastic Gradient Descent or Momentum optimizer.

        Arguments:
          learning_rate: float hyperparameter >= 0. Learning rate.
          momentum: float hyperparameter >= 0 that accelerates SGD in the relevant
            direction and dampens oscillations.
          nesterov: boolean. Whether to apply Nesterov momentum.
          name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'SGD'.
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """
        super(SGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return training_ops.resource_apply_keras_momentum(
                var.handle,
                momentum_var.handle,
                lr_t,
                grad,
                self._get_hyper("momentum", var_dtype),
                use_locking=self._use_locking,
                use_nesterov=self.nesterov)
        else:
            return training_ops.resource_apply_gradient_descent(
                var.handle, lr_t, grad, use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        if self._momentum:
            return super(SGD, self)._resource_apply_sparse_duplicate_indices(
                grad, var, indices)
        else:
            var_dtype = var.dtype.base_dtype
            lr_t = self._decayed_lr(var_dtype)
            return resource_variable_ops.resource_scatter_add(var.handle, indices,
                                                              -grad * lr_t)

    def _resource_apply_sparse(self, grad, var, indices):
        # This method is only needed for momentum optimization.
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum_var = self.get_slot(var, "momentum")
        return training_ops.resource_sparse_apply_keras_momentum(
            var.handle,
            momentum_var.handle,
            lr_t,
            grad,
            indices,
            self._get_hyper("momentum", var_dtype),
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(SGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config
