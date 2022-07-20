# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras wrapper layer for Haiku modules"""

import collections
import inspect

import jax
from tensorflow.python.util.tf_export import keras_export

from keras.layers.jax.jax_layer import JaxLayer


@keras_export("keras.layers.experimental.FlaxLayer")
class FlaxLayer(JaxLayer):
    """Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.

    This layer enables the use of Flax components in the form of
    [`flax.linen.Module`](
            https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    instances within Keras. It is based on the experimental `jax2tf` and is
    therefore subject to the same limitations. Details are covered in the
    [jax2tf documentation](
      https://github.com/google/jax/tree/main/jax/experimental/jax2tf).

    The module method must take the following arguments with these exact names:

    - `self` if the method is bound to the module, which is the case for the
      default of `__call__`, and `module` otherwise to pass the module.
    - `inputs`: the inputs to the model, a JAX `DeviceArray` or a `PyTree` of
      `DeviceArray`s.
    - `training` *(optional)*: an argument specifying if we're in training mode
      or inference mode, `True` is passed in training mode.

    `FlaxLayer` handles the non-trainable state of your model and required RNGs
    automatically. Note that the `mutable` parameter of
    [`flax.linen.Module.apply()`](
      https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply)
    is set to `DenyList(["params"])`, therefore making the assumptions that all
    values outside of the "params" collection are non-trainable weights.

    This example shows how to create a `FlaxLayer` from a Flax `Module` with
    the default `__call__` method and no training argument:

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, inputs):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(flax_module)
    ```

    This example shows how to wrap the module method to conform to the required
    signature. This allows having multiple input arguments and a training
    argument that has a different name and values. This additionally shows how
    to use a function that is not bound to the module.

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def forward(self, input1, input1, deterministic):
            ...
            return outputs

    def my_flax_module_wrapper(module, inputs, training):
        input1, input2 = inputs
        return module.forward(input1, input2, not training)

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(flax_module)
        module=flax_module,
        method=my_flax_module_wrapper,
    )
    ```

    Args:
      module: An instance of `flax.linen.Module` or subclass.
      method: The method to call the model. This is generally a method in the
        `Module`. If not provided, the `__call__` method is used instead.
        `method` can also be a function not defined in the `Module`, in which
        case it must take the `Module` as the first argument. It is used for
        both `Module.init` and `Module.apply`. Details are documented in the
        `method` argument of [`flax.linen.Module.apply()`](
          https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply).
      variables: A `dict` containing all the variables of the module in the
        same format as what is returned by [`flax.linen.Module.init()`](
          https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init).
        It should contain a "params" key and, if applicable, other keys for
        collections of variables for non-trainable state. This allows passing
        trained parameters and learned non-trainable state or controlling the
        initialization. If `None` is passed, the module's `init` function is
        called at build time to initialize the variables of the model.
    """

    def __init__(
        self,
        module,
        method=None,
        variables=None,
        **kwargs,
    ):
        self.module = module
        self.method = method

        # Late import to prevent circular dependency between flax and tf.
        from flax.core import scope as flax_scope

        apply_mutable = flax_scope.DenyList(["params"])

        def apply_with_training(params, state, rng, inputs, training):
            return self.module.apply(
                self._params_and_state_to_variables(params, state),
                inputs,
                rngs=self._get_apply_rng(rng),
                method=self.method,
                mutable=apply_mutable,
                training=training,
            )

        def apply_without_training(params, state, rng, inputs):
            return self.module.apply(
                self._params_and_state_to_variables(params, state),
                inputs,
                rngs=self._get_apply_rng(rng),
                method=self.method,
                mutable=apply_mutable,
            )

        def init_with_training(rng, inputs, training):
            return self._variables_to_params_and_state(
                self.module.init(
                    self._get_init_rng(rng),
                    inputs,
                    method=self.method,
                    training=training,
                )
            )

        def init_without_training(rng, inputs):
            return self._variables_to_params_and_state(
                self.module.init(
                    self._get_init_rng(rng),
                    inputs,
                    method=self.method,
                )
            )

        if (
            "training"
            in inspect.signature(method or module.__call__).parameters
        ):
            apply_fn, init_fn = apply_with_training, init_with_training
        else:
            apply_fn, init_fn = apply_without_training, init_without_training

        params, state = self._variables_to_params_and_state(variables)

        super().__init__(
            apply_fn=apply_fn,
            init_fn=init_fn,
            params=params,
            state=state,
            **kwargs,
        )

    def _params_and_state_to_variables(self, params, state):
        if params:
            if state:
                return {**params, **state}
            else:
                return params
        elif state:
            return state
        return {}

    def _variables_to_params_and_state(self, variables):
        # neither params nor state
        if variables is None:
            return None, None
        # state only
        if "params" not in variables:
            return None, variables
        # params only
        if len(variables) == 1:
            return variables, None
        # both, we need to split
        params = {"params": variables["params"]}
        state = {k: v for k, v in variables.items() if k != "params"}
        return params, state

    def _get_apply_rng(self, rng):
        return {"dropout": rng}

    def _get_init_rng(self, rng):
        rng1, rng2 = jax.random.split(rng)
        return {"params": rng1, "dropout": rng2}

    def _create_variables(self, values, trainable):
        # Turn Flax's FrozenDicts to regular dicts because they break load/save.
        def recreate_dicts(structure):
            if isinstance(structure, collections.abc.Mapping):
                return {k: recreate_dicts(v) for k, v in structure.items()}
            else:
                return structure

        return recreate_dicts(super()._create_variables(values, trainable))

    def get_config(self):
        config_method = self.method
        if (
            hasattr(self.method, "__self__")
            and self.method.__self__ == self.module
        ):
            # A method bound to the module is serialized by name.
            config_method = self.method.__name__
        config = {
            "module": self.module,
            "method": config_method,
        }
        base_config = super().get_config()
        # apply_fn and init_fn come from module, do not save them.
        base_config.pop("apply_fn")
        base_config.pop("init_fn")
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["method"], str):
            # Deserialize bound method from the module.
            module = config["module"]
            method = config["method"]
            config["method"] = getattr(module, method)
        return super(FlaxLayer, cls).from_config(config)
