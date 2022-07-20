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
"""Keras wrapper layer for JAX models"""


import collections
import functools
import inspect
import itertools
import re
import string

import jax
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import tf_utils


@keras_export("keras.layers.experimental.JaxLayer")
class JaxLayer(Layer):
    """Keras Layer that wraps a [JAX](https://jax.readthedocs.io) model.

    This layer enables the use of JAX components within Keras. It is based
    on the experimental `jax2tf` and is therefore subject to the same
    limitations. Details are covered in the [jax2tf documentation](
      https://github.com/google/jax/tree/main/jax/experimental/jax2tf).

    ## Model function

    This layer accepts JAX models in the form of a function, `apply_fn`, which
    must take the following arguments with these exact names:

    - `params`: trainable parameters of the model.
    - `state` (*optional*): non-trainable state of the model. Can be omitted if
      the model has no non-trainable state.
    - `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if the
      model does not need RNGs, neither during training nor during inference.
    - `inputs`: inputs to the model, a JAX `DeviceArray` or a `PyTree`s of
      `DeviceArray`s.
    - `training` (*optional*): an argument specifying if we're in training mode
      or inference mode, `True` is passed in training mode. Can be omitted if
      the model behaves the same in training mode and inference mode.

    The `inputs` argument is mandatory. Inputs to the model must be provided via
    a single argument. If the JAX model takes multiple inputs as separate
    arguments, they must be combined into a single structure, for instance in a
    `tuple` or a `dict`.

    ## Model weights initialization

    The initialization of the `params` and `state` of the model can be handled
    by this layer, in which case the `init_fn` argument must be provided. This
    allows the model to be initialized dynamically with the right shape.
    Alternatively, and if the shape is known, the `params` argument and
    optionally the `state` arguments can be used to create an already
    initialized model.

    The `init_fn` function, if provided, must take the following arguments with
    these exact names:

    - `rng`: a `jax.random.PRNGKey` instance.
    - `inputs`: a JAX `DeviceArray` or a `PyTree`s of `DeviceArray`s with dummy
      values to provide the shape of the inputs.
    - `training` (*optional*): an argument specifying if we're in training mode
      or inference mode. `True` is always passed to `init_fn`. Can be omitted
      regardless of whether `apply_fn` has a `training` argument.

    ## Models with non-trainable state

    For JAX models that have non-trainable state:

    - `apply_fn` must have a `state` argument
    - `apply_fn` must return a tuple containing the outputs of the model and the
      new non-trainable state of the model
    - `init_fn` must return a tuple containing the initial trainable params of
      the model and the initial non-trainable state of the model.

    This code shows a possible combination of `apply_fn` and `init_fn`
    signatures for a model with non-trainable state. In this example, the model
    has a `training` argument and has an `rng` argument in `apply_fn`.

    ```python
    def stateful_apply(params, state, rng, inputs, training):
        outputs = ...
        new_state = ...
        return outputs, new_state

    def stateful_init(rng, inputs):
        initial_params = ...
        initial_state = ...
        return initial_params, initial_state
    ```

    ## Models without non-trainable state

    For JAX models with no non-trainable state:

    - `apply_fn` must not have a `state` argument
    - `apply_fn` must return only the outputs of the model
    - `init_fn` must return only the initial trainable params of the model.

    This code shows a possible combination of `apply_fn` and `init_fn`
    signatures for a model without non-trainable state. In this example, the
    model does not have a `training` argument and does not have an `rng`
    argument in `apply_fn`.

    ```python
    def stateless_apply(params, inputs):
        outputs = ...
        return outputs

    def stateless_init(rng, inputs):
        initial_params = ...
        return initial_params
    ```

    ## Conforming to the required signature

    If a model has a different signature than the one required by `JaxLayer`,
    one can easily write a wrapper method to adapt the arguments. This example
    shows a model that has multiple inputs as separate arguments, expects
    multiple RNGs in a `dict`, and has a `deterministic` argument with the
    opposite meaning as `training`. To conform, the inputs are combined in a
    single structure using a `tuple`, the RNG is split and used the populate the
    expected `dict`:

    ```python
    def my_model_fn(params, rngs, input1, input2, deterministic):
        ...
        if not deterministic:
            dropout_rng = rngs["dropout"]
            keep = jax.random.bernoulli(dropout_rng, dropout_rate, x.shape)
            x = jax.numpy.where(keep, x / dropout_rate, 0)
            ...
        ...
        return outputs

    def my_model_wrapper_fn(params, rng, inputs, training):
        input1, input2 = inputs
        rng1, rng2 = jax.random.split(rng)
        rngs = {"dropout": rng1, "preprocessing": rng2}
        deterministic = not training
        return my_model_fn(params, rngs, input1, input2, deterministic)

    keras_layer = JaxLayer(my_model_wrapper_fn, params=initial_params)
    ```

    ## Usage with Haiku modules

    `JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
    components in the form of
    [`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
    This is achieved by transforming the module per the Haiku pattern and then
    passing `module.apply` in the `apply_fn` parameter and `module.init` in the
    `init_fn` parameter if needed.

    If the model has non-trainable state, it should be transformed with
    [`haiku.transform_with_state`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
    If the model has no non-trainable state, it should be transformed with
    [`haiku.transform`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
    Additionally, and optionally, if the module does not use RNGs in "apply", it
    can be tranformed with
    [`haiku.without_apply_rng`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).

    The following example shows how to create a `JaxLayer` from a Haiku module
    that uses random number generators via `hk.next_rng_key()` and takes a
    training positional argument:

    ```python
    class MyHaikuModule(hk.Module):
    def __call__(self, x, training):
        x = hk.Conv2D(32, (3, 3))(x)
        x = jax.nn.relu(x)
        x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)
        x = hk.Flatten()(x)
        x = hk.Linear(200)(x)
        if training:
            x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        x = jax.nn.softmax(x)
        return x

    def my_haiku_module_fn(inputs, training):
        module = MyHaikuModule()
        return module(inputs, training)

    transformed_module = hk.transform(my_haiku_module_fn)

    keras_layer = HaikuLayer(
        apply_fn=transformed_module.apply,
        init_fn=transformed_module.init,
    )
    ```

    Args:
      apply_fn: The function to call the model. See description above for the
        list of arguments it takes and the outputs it returns.
      init_fn: the function to call to initialize the model. See description
        above for the list of arguments it takes and the ouputs it returns. If
        `None`, then `params` and/or `state` must be provided.
      params: A `PyTree` containing all the model trainable parameters. This
        allows passing trained parameters or controlling the initialization.
        If both `params` and `state` are `None`, `init_fn` is called at build
        time to initialize the trainable parameters of the model.
      state: A `PyTree` containing all the model non-trainable state. This
        allows passing learned state or controlling the initialization. If both
        `params` and `state` are `None`, and `apply_fn` takes a `state`
        argument, then `init_fn` is called at build time to initialize the
        non-trainable state of the model.
    """

    def __init__(
        self,
        apply_fn,
        init_fn=None,
        params=None,
        state=None,
        **kwargs,
    ):
        if init_fn is None and params is None and state is None:
            raise ValueError(
                "`init_fn`, `params` and `state` cannot all be `None`."
            )

        super().__init__(**kwargs)
        self.apply_fn = apply_fn
        self.init_fn = init_fn
        self.params = self._create_variables(params, trainable=True)
        self.state = self._create_variables(state, trainable=False)
        self.rng = jax.random.PRNGKey(tf_utils.get_random_seed())

        self.apply_fn_arguments = self._validate_signature(
            apply_fn,
            "apply_fn",
            {"params", "state", "rng", "inputs", "training"},
            {"inputs"},
        )
        self.has_state = "state" in self.apply_fn_arguments

        if init_fn:
            self.init_fn_arguments = self._validate_signature(
                init_fn, "init_fn", {"rng", "inputs", "training"}, {"inputs"}
            )

    def _validate_signature(self, fn, fn_name, allowed, required):
        fn_parameters = inspect.signature(fn).parameters
        for parameter_name in required:
            if parameter_name not in fn_parameters:
                raise ValueError(
                    f"Missing required argument in `{fn_name}`: "
                    f"`{parameter_name}`"
                )

        parameter_names = []
        for parameter in fn_parameters.values():
            if parameter.name not in allowed:
                raise ValueError(
                    f"Unsupported argument in `{fn_name}`: `{parameter.name}`, "
                    f"supported arguments are `{'`, `'.join(allowed)}`"
                )
            parameter_names.append(parameter.name)

        return parameter_names

    def _get_jax2tf_input_shape(self, input_shape):
        """Convert input shape in a format suitable for `jax2tf`.

        `jax2tf` expects a letter for each unknown dimension, which allows
        correlated dimensions. Since correlated dimensions are not supported by
        Keras, we simply use 'a', 'b', 'c'..., for each unknown dimension. We
        however use 'batch' for dimension 0 if not defined to correlate the
        batch size across inputs.

        Example (spaces added for readability):
        ```
        input_shape:  (None , 4   , None, None, 5   )
        result:      "(batch, 4   , a   , b   , 5   )"
        ```

        Args:
          input_shape: a single shape or a structure of shapes for the inputs.
        Returns:
          the shape or shapes structure in the `jax2tf` format as strings.
        """
        dim_names = itertools.chain(
            string.ascii_lowercase,  # a, b, ... z
            itertools.starmap(  # aa, ab, ... az, ba, bb, ... zz
                lambda a, b: a + b,
                itertools.product(string.ascii_lowercase, repeat=2),
            ),
        )

        def get_single_jax2tf_shape(shape):
            jax2tf_shape = []

            for index, dim in enumerate(shape.as_list()):
                if dim is not None:
                    jax2tf_shape.append(str(dim))
                elif index == 0:
                    jax2tf_shape.append("batch")
                else:
                    jax2tf_shape.append(next(dim_names))

            return "(" + ", ".join(jax2tf_shape) + ")"

        return tf.nest.map_structure(get_single_jax2tf_shape, input_shape)

    def _jax2tf_convert(self, fn, polymorphic_shapes):
        # Late import to prevent circular dependency between jax2tf and tf.
        from jax.experimental import jax2tf

        converted_fn = jax2tf.convert(fn, polymorphic_shapes=polymorphic_shapes)
        # Autograph won't work with the output of jax2tf.
        converted_fn = tf.autograph.experimental.do_not_convert(converted_fn)
        return converted_fn

    def _create_variables(self, values, trainable):
        """Create a structure of variables from a structure of JAX arrays.

        `values` is traversed via JAX's `tree_map`. When a leaf is a JAX array
        or a tensor-like object, a corresponding variable is created with it as
        the initial value. The resulting structure of variables is returned.
        Note that leaf objects that are not JAX arrays and not tensor-like are
        left intact as they are assumed to be configuration used by the model.

        Args:
          values: the structure of values to traverse.
          trainable: whether to create trainable variables.
        Returns:
          the same structure as `values` with variables initialized with the
          `values` tensors as leaves.
        """

        def create_variable(value):
            if tf.is_tensor(value) or isinstance(
                value,
                (jax.Array, np.ndarray, np.generic, int, float),
            ):
                return tf.Variable(value, trainable=trainable)
            else:
                return value

        # Use jax's tree_map as it understands pytrees and registered classes.
        return jax.tree_util.tree_map(create_variable, values)

    def _get_variables_updates(self, variables, values, path=""):
        """Return callables assigning values to a structure of variables.

        Both the `variables` and the `values` structures are traversed at the
        same time via recursion. When a tensor-like object is encountered in
        `values` and the corresponding variable exists in `variables`, a new
        update callable, which does the variable assignment, is added to the
        list of updates. This list of all updates is what is returned.
        When a sub-structure of `values` has no corresponding sub-structure in
        `variables`, new variables are created using `_create_variables`.

        Args:
          variables: the structure of variables to assign new values to.
            `self.state` should be passed at the top level.
          values: the structure of values to assign.
          path: string representation of the position within the structure to be
            used for scope names and logging.
        Returns:
          a list of callables for the variables updates.
        """

        def sanitize_scope_name(name):
            if getattr(sanitize_scope_name, "invalid_char", None) is None:
                sanitize_scope_name.invalid_char = re.compile(
                    "[^A-Za-z0-9_.\\/>-]"
                )
            return sanitize_scope_name.invalid_char.sub("_", name)

        def assign_fn(variable, value, name):
            name = sanitize_scope_name(name)
            with backend.name_scope("AssignNewValue" + name):
                return variable.assign(value)

        updates = []

        if variables is None and not path:
            self.state = self._create_variables(values, False)
        elif tf.is_tensor(values) or isinstance(
            values,
            (jax.Array, np.ndarray, np.generic, int, float),
        ):
            if not isinstance(variables, tf.Variable):
                raise ValueError(
                    f"Type mismatch at {path if path else '/'}: "
                    f"`values` is a tensor-like of type {type(values)} but "
                    f"`variables` of type {type(variables)} is not a variable"
                )
            updates.append(
                functools.partial(assign_fn, variables, values, path)
            )
        elif isinstance(values, collections.abc.Mapping):
            if not isinstance(variables, collections.abc.Mapping):
                raise ValueError(
                    f"Structure type mismatch at {path if path else '/'}: "
                    f"`values` is a mapping of type {type(values)} but "
                    f"`variables` of type {type(variables)} is not a mapping"
                )
            for key, value in values.items():
                if key in variables:
                    updates.extend(
                        self._get_variables_updates(
                            variables[key], value, path + "/" + key
                        )
                    )
                else:
                    variables[key] = self._create_variables(value, False)
        elif isinstance(values, collections.abc.Sequence):
            if not isinstance(variables, collections.abc.Sequence):
                raise ValueError(
                    f"Structure type mismatch at {path if path else '/'}: "
                    f"`values` is a sequence of type {type(values)} but "
                    f"`variables` of type {type(variables)} is not a sequence"
                )
            for index, value in enumerate(values):
                if index < len(variables):
                    updates.extend(
                        self._get_variables_updates(
                            variables[index], value, path + "/" + str(index)
                        )
                    )
                else:
                    variables.append(self._create_variables(value, False))

        return updates

    def _partial_with_positional(self, fn, index, value):
        """Return a new partial with one positional argument set to a value.

        This is needed because `jax2tf` only supports positional arguments and
        `functools.partial` only supports setting positional arguments starting
        from the left. Our use case is the `training` argument which is
        typically the righmost argument.

        Args:
          fn: the function to wrap.
          index: the index of the positional argument to set to `value`.
          value: the value for the positional argument at `index`.
        """

        @functools.wraps(fn)
        def wrapper(*args):
            args = args[0:index] + (value,) + args[index:]
            return fn(*args)

        return wrapper

    def _make_rng(self, training):
        self.rng, new_rng = jax.random.split(self.rng)
        return new_rng

    def _call_init_fn(self, input_shape):
        def create_input(shape):
            shape = [d if d is not None else 1 for d in shape.as_list()]
            return jax.numpy.ones(shape)

        init_inputs = tf.nest.map_structure(create_input, input_shape)
        init_args = []
        for argument_name in self.init_fn_arguments:
            if argument_name == "rng":
                init_args.append(self._make_rng(True))
            elif argument_name == "inputs":
                init_args.append(init_inputs)
            elif argument_name == "training":
                init_args.append(True)

        init_result = self.init_fn(*init_args)
        if self.has_state:
            init_params, init_state = init_result
        else:
            init_params, init_state = init_result, None

        self.params = self._create_variables(init_params, trainable=True)
        self.state = self._create_variables(init_state, trainable=False)

    def build(self, input_shape):
        # Use TensorShapes as tuples won't work with tf.nest.map_structure.
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

        # Initialize `params` and `state` if needed by calling `init_fn`.
        if self.params is None and self.state is None:
            self._call_init_fn(input_shape)

        polymorphic_shapes = []
        for argument in self.apply_fn_arguments:
            if argument == "inputs":
                polymorphic_shapes.append(
                    self._get_jax2tf_input_shape(input_shape)
                )
            elif argument != "training":
                # params, state, rng
                polymorphic_shapes.append("...")

        if "training" in self.apply_fn_arguments:
            training_argument_index = self.apply_fn_arguments.index("training")
            self.jax2tf_training_false_fn = self._jax2tf_convert(
                self._partial_with_positional(
                    self.apply_fn, training_argument_index, False
                ),
                polymorphic_shapes,
            )
            self.jax2tf_training_true_fn = self._jax2tf_convert(
                self._partial_with_positional(
                    self.apply_fn, training_argument_index, True
                ),
                polymorphic_shapes,
            )
        else:
            self.jax2tf_training_false_fn = self._jax2tf_convert(
                self.apply_fn,
                polymorphic_shapes,
            )
            self.jax2tf_training_true_fn = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        if not self.trainable:
            training = False

        # if has_arg[in_index++]: call_args.append(self.params)

        # if self.apply_argument_indexes.params is not None:
        #     call_args[self.apply_argument_indexes.params] = self.params

        call_args = []
        for argument_name in self.apply_fn_arguments:
            if argument_name == "params":
                call_args.append(self.params)
            elif argument_name == "state":
                call_args.append(self.state)
            elif argument_name == "rng":
                call_args.append(self._make_rng(training))
            elif argument_name == "inputs":
                call_args.append(inputs)

        def call_with_jax2tf_fn(fn):
            if self.has_state:
                predictions, new_state = fn(*call_args)
                self.add_update(
                    self._get_variables_updates(self.state, new_state)
                )
                return predictions
            else:
                return fn(*call_args)

        if self.jax2tf_training_true_fn is None:
            return call_with_jax2tf_fn(self.jax2tf_training_false_fn)
        else:
            return backend.in_train_phase(
                lambda: call_with_jax2tf_fn(self.jax2tf_training_true_fn),
                lambda: call_with_jax2tf_fn(self.jax2tf_training_false_fn),
                training=training,
            )

    def get_config(self):
        config = {
            "apply_fn": self.apply_fn,
            "init_fn": self.init_fn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
