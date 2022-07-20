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
"""Tests for JAX Keras layer."""

import os

import flax
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from jax.example_libraries import stax

import keras
from keras.layers.jax.flax_layer import FlaxLayer
from keras.layers.jax.jax_layer import JaxLayer
from keras.saving import object_registration
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import layer_utils

num_classes = 10
input_shape = (28, 28, 1)  # Excluding batch_size


@object_registration.register_keras_serializable()
def jax_stateless_init(rng, inputs):
    layer_sizes = [784, 300, 100, 10]
    params = []
    w_init = jax.nn.initializers.glorot_normal()
    b_init = jax.nn.initializers.normal(0.1)
    for (m, n) in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, w_rng = jax.random.split(rng)
        rng, b_rng = jax.random.split(rng)
        params.append([w_init(w_rng, (m, n)), b_init(b_rng, (n,))])
    return params


@object_registration.register_keras_serializable()
def jax_stateless_apply(params, inputs):
    activations = inputs.reshape((inputs.shape[0], -1))  # flatten
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return jax.nn.softmax(logits, axis=-1)


@object_registration.register_keras_serializable()
def jax_stateful_init(rng, inputs, training):
    params = jax_stateless_init(rng, inputs)
    state = jnp.zeros([], jnp.int32)
    return params, state


@object_registration.register_keras_serializable()
def jax_stateful_apply(params, state, inputs, training):
    outputs = jax_stateless_apply(params, inputs)
    if training:
        state = state + 1
    return outputs, state


def stax_training_independent_model():
    init_fun, apply_fun = stax.serial(
        stax.Flatten,
        stax.Dense(300),
        stax.Relu,
        stax.Dense(100),
        stax.Relu,
        stax.Dense(10),
        stax.Softmax,
    )

    def apply_fn_without_keyword_args(params, inputs):
        return apply_fun(params, inputs)

    return init_fun, apply_fn_without_keyword_args


def StaxDropout(rate):
    """Layer construction function for a dropout layer with given rate.

    Fixes two issues with stax.Dropout
    * Instead of `mode` being a constructor argument, this has a `training`
      argument in `apply_fun`.
    * The `rng` argument is not required when not in training mode.
    """

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        if kwargs.get("training", False):
            rng = kwargs.get("rng", None)
            if rng is None:
                raise ValueError(
                    "StaxDropout apply_fun requires `rng` argument"
                )
            keep = jax.random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs

    return init_fun, apply_fun


def stax_training_dependent_model():
    init_fun, apply_fun = stax.serial(
        stax.Flatten,
        stax.Dense(300),
        stax.Relu,
        stax.Dense(100),
        stax.Relu,
        StaxDropout(0.30),
        stax.Dense(10),
        stax.Softmax,
    )

    def apply_fn_without_keyword_args(params, rng, inputs, training):
        return apply_fun(params, inputs, rng=rng, training=training)

    return init_fun, apply_fn_without_keyword_args


@object_registration.register_keras_serializable()
def haiku_training_independent_module(inputs):
    module = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.softmax,
        ]
    )
    return module(inputs)


class HaikuDropoutModule(hk.Module):
    def __call__(self, x, training):
        x = hk.Conv2D(32, (3, 3))(x)
        x = jax.nn.relu(x)
        x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)
        x = hk.Conv2D(64, (3, 3))(x)
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


@object_registration.register_keras_serializable()
def haiku_dropout_module(inputs, training):
    module = HaikuDropoutModule()
    return module(inputs, training)


class HaikuBatchNormModule(hk.Module):
    def __call__(self, x, training):
        x = hk.Conv2D(12, (3, 3), with_bias=False)(x)
        x = hk.BatchNorm(False, True, 0.9)(x, training)
        x = jax.nn.relu(x)
        x = hk.Conv2D(24, (6, 6), stride=(2, 2))(x)
        x = hk.BatchNorm(False, True, 0.9)(x, training)
        x = jax.nn.relu(x)
        x = hk.Conv2D(32, (6, 6), stride=(2, 2))(x)
        x = hk.BatchNorm(False, True, 0.9)(x, training)
        x = hk.Flatten()(x)
        x = hk.Linear(200)(x)
        x = hk.BatchNorm(False, True, 0.9)(x, training)
        if training:
            x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        x = jax.nn.softmax(x)
        return x


@object_registration.register_keras_serializable()
def haiku_batchnorm_module(inputs, training=False):
    module = HaikuBatchNormModule()
    return module(inputs, training)


@object_registration.register_keras_serializable()
class FlaxTrainingIndependentModel(flax.linen.Module):
    @flax.linen.compact
    def forward(self, inputs):
        x = inputs
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=200)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=10)(x)
        x = flax.linen.softmax(x)
        return x

    def get_config(self):
        return {}


@object_registration.register_keras_serializable()
class FlaxDropoutModel(flax.linen.Module):
    @flax.linen.compact
    def my_apply(self, inputs, training):
        x = inputs
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=200)(x)
        x = flax.linen.Dropout(rate=0.3, deterministic=not training)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=10)(x)
        x = flax.linen.softmax(x)
        return x

    def get_config(self):
        return {}


@object_registration.register_keras_serializable()
def flax_dropout_wrapper(module, x, training):
    return module.my_apply(x, training)


@object_registration.register_keras_serializable()
class FlaxBatchNormModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, inputs, training=False):
        ura = not training
        x = inputs
        x = flax.linen.Conv(features=12, kernel_size=(3, 3), use_bias=False)(x)
        x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Conv(features=24, kernel_size=(6, 6), strides=(2, 2))(x)
        x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Conv(features=32, kernel_size=(6, 6), strides=(2, 2))(x)
        x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=200, use_bias=True)(x)
        x = flax.linen.BatchNorm(use_running_average=ura, use_scale=False)(x)
        x = flax.linen.Dropout(rate=0.3, deterministic=not training)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=10)(x)
        x = flax.linen.softmax(x)
        return x

    def get_config(self):
        return {}


@test_utils.run_v2_only
@test_combinations.run_all_keras_modes()
class TestJaxLayer(test_combinations.TestCase):
    def _test_layer(
        self,
        model_name,
        layer_class,
        layer_init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        # Fake MNIST data
        x_train = tf.random.uniform((320, 28, 28, 1))
        y_train = tf.one_hot(
            tf.random.uniform((320,), maxval=num_classes, dtype=tf.int32),
            num_classes,
        )
        x_test = tf.random.uniform((32, 28, 28, 1))
        y_test = tf.one_hot(
            tf.random.uniform((32,), maxval=num_classes, dtype=tf.int32),
            num_classes,
        )

        def verify_weights_and_params(layer):
            self.assertEqual(trainable_weights, len(layer.trainable_weights))
            self.assertEqual(
                trainable_params,
                layer_utils.count_params(layer.trainable_weights),
            )
            self.assertEqual(
                non_trainable_weights, len(layer.non_trainable_weights)
            )
            self.assertEqual(
                non_trainable_params,
                layer_utils.count_params(layer.non_trainable_weights),
            )

        # functional model
        layer1 = layer_class(**layer_init_kwargs)
        inputs1 = keras.Input(shape=input_shape)
        outputs1 = layer1(inputs1)
        model1 = keras.Model(
            inputs=inputs1, outputs=outputs1, name=model_name + "1"
        )
        model1.summary()

        verify_weights_and_params(layer1)

        model1.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[keras.metrics.CategoricalAccuracy()],
        )

        tw1_before_fit = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.trainable_weights
        )
        ntw1_before_fit = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.non_trainable_weights
        )
        model1.fit(x_train, y_train, epochs=1, steps_per_epoch=10)
        tw1_after_fit = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.trainable_weights
        )
        ntw1_after_fit = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.non_trainable_weights
        )

        # verify both trainable and non-trainable weights did change after fit
        for before, after in zip(tw1_before_fit, tw1_after_fit):
            self.assertNotAllClose(before, after)
        for before, after in zip(ntw1_before_fit, ntw1_after_fit):
            self.assertNotAllClose(before, after)

        eval1 = model1.evaluate(x_test, y_test, steps=1)

        expected_ouput_shape = tf.TensorShape([x_test.shape[0], num_classes])
        output1 = model1(x_test)
        self.assertEqual(output1.shape, expected_ouput_shape)
        predict1 = model1.predict(x_test, steps=1)
        self.assertEqual(predict1.shape, expected_ouput_shape)

        # verify both trainable and non-trainable weights did not change
        tw1_after_call = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.trainable_weights
        )
        ntw1_after_call = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.non_trainable_weights
        )
        self.assertAllClose(tw1_after_fit, tw1_after_call)
        self.assertAllClose(ntw1_after_fit, ntw1_after_call)

        exported_params = tf.nest.map_structure(
            lambda t: t.read_value(), layer1.params
        )
        if layer1.state is not None:
            exported_state = tf.nest.map_structure(
                lambda t: t.read_value(), layer1.state
            )
        else:
            exported_state = None

        def verify_identical_model(model):
            output = model(x_test)
            self.assertAllClose(output1, output)

            predict = model.predict(x_test, steps=1)
            self.assertAllClose(predict1, predict)

            eval = model.evaluate(x_test, y_test, steps=1)
            self.assertAllClose(eval1, eval)

        # sequential model to compare results
        layer2 = layer_class(
            params=exported_params,
            state=exported_state,
            input_shape=input_shape,
            **layer_init_kwargs,
        )
        model2 = keras.Sequential([layer2], name=model_name + "2")
        model2.summary()
        verify_weights_and_params(layer2)
        model2.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
        verify_identical_model(model2)

        # save, load back and compare results
        path = os.path.join(self.get_temp_dir(), "model.tf")
        keras.models.save_model(model2, path, save_format="tf")

        model3 = keras.models.load_model(path)
        layer3 = model3.layers[0]
        model3.summary()
        verify_weights_and_params(layer3)
        verify_identical_model(model3)

    @parameterized.named_parameters(
        {
            "testcase_name": "training_independent",
            "init_kwargs": {
                "apply_fn": jax_stateless_apply,
                "init_fn": jax_stateless_init,
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_kwarg_state",
            "init_kwargs": {
                "apply_fn": jax_stateful_apply,
                "init_fn": jax_stateful_init,
            },
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 1,
            "non_trainable_params": 1,
        },
    )
    def test_jax_layer(
        self,
        init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        self._test_layer(
            init_kwargs["apply_fn"].__name__,
            JaxLayer,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "training_independent",
            "stax_model_fn": stax_training_independent_model,
            "init_kwargs": {},
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_kwarg_rng_kwarg",
            "stax_model_fn": stax_training_dependent_model,
            "init_kwargs": {},
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
    )
    def test_jax_layer_with_stax_model(
        self,
        stax_model_fn,
        init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        stax_init_fn, jax_apply_fn = stax_model_fn()

        def jax_init_fn(rng, inputs):
            # 1. pass a shape as input, not a tensor
            # 2. skip first part of tuple (the output shape)
            return stax_init_fn(rng, jnp.shape(inputs))[1]

        object_registration.register_keras_serializable(
            name=stax_model_fn.__name__ + "_init"
        )(jax_init_fn)
        object_registration.register_keras_serializable(
            name=stax_model_fn.__name__ + "_apply"
        )(jax_apply_fn)

        init_kwargs["apply_fn"] = jax_apply_fn
        init_kwargs["init_fn"] = jax_init_fn

        self._test_layer(
            stax_model_fn.__name__,
            JaxLayer,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "training_independent",
            "module_fn": haiku_training_independent_module,
            "with_state": False,
            "without_rng": True,
            "trainable_weights": 6,
            "trainable_params": 266610,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_positional_rng_positional",
            "module_fn": haiku_dropout_module,
            "with_state": False,
            "without_rng": False,
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_kwarg_rng_positional_state",
            "module_fn": haiku_batchnorm_module,
            "with_state": True,
            "without_rng": False,
            "trainable_weights": 13,
            "trainable_params": 354258,
            "non_trainable_weights": 24,
            "non_trainable_params": 1080,
        },
    )
    def test_jax_layer_with_haiku_model(
        self,
        module_fn,
        with_state,
        without_rng,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        if with_state:
            transformed_module = hk.transform_with_state(module_fn)
        else:
            transformed_module = hk.transform(module_fn)

        if without_rng:
            transformed_module = hk.without_apply_rng(transformed_module)

        object_registration.register_keras_serializable(
            name=module_fn.__name__ + "_init"
        )(transformed_module.init)
        object_registration.register_keras_serializable(
            name=module_fn.__name__ + "_apply"
        )(transformed_module.apply)

        init_kwargs = {
            "apply_fn": transformed_module.apply,
            "init_fn": transformed_module.init,
        }
        self._test_layer(
            module_fn.__name__,
            JaxLayer,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "training_independent_rng_kwarg_none_bound_method",
            "flax_model_class": FlaxTrainingIndependentModel,
            "flax_model_method": "forward",
            "init_kwargs": {},
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_positional_rng_kwarg_unbound_method",
            "flax_model_class": FlaxDropoutModel,
            "flax_model_method": None,
            "init_kwargs": {
                "method": flax_dropout_wrapper,
            },
            "trainable_weights": 8,
            "trainable_params": 648226,
            "non_trainable_weights": 0,
            "non_trainable_params": 0,
        },
        {
            "testcase_name": "training_kwarg_rng_kwarg_state_no_method",
            "flax_model_class": FlaxBatchNormModel,
            "flax_model_method": None,
            "init_kwargs": {},
            "trainable_weights": 13,
            "trainable_params": 354258,
            "non_trainable_weights": 8,
            "non_trainable_params": 536,
        },
    )
    def test_flax_layer(
        self,
        flax_model_class,
        flax_model_method,
        init_kwargs,
        trainable_weights,
        trainable_params,
        non_trainable_weights,
        non_trainable_params,
    ):
        def create_wrapper(**kwargs):
            params = kwargs.pop("params") if "params" in kwargs else None
            state = kwargs.pop("state") if "state" in kwargs else None
            if params and state:
                variables = {**params, **state}
            elif params:
                variables = params
            elif state:
                variables = state
            else:
                variables = None
            kwargs["variables"] = variables
            flax_model = flax_model_class()
            if flax_model_method:
                kwargs["method"] = getattr(flax_model, flax_model_method)
            return FlaxLayer(flax_model_class(), **kwargs)

        self._test_layer(
            flax_model_class.__name__,
            create_wrapper,
            init_kwargs,
            trainable_weights,
            trainable_params,
            non_trainable_weights,
            non_trainable_params,
        )

    def test_with_no_init_fn_and_no_params(self):
        def jax_fn(params, inputs):
            return inputs

        with self.assertRaises(ValueError):
            JaxLayer(jax_fn)

    def test_with_training_in_apply_fn_but_not_init_fn(self):
        def jax_apply_fn(params, state, rng, inputs, training):
            return inputs, {}

        def jax_init_fn(rng, inputs):
            return {}, {}

        layer = JaxLayer(jax_apply_fn, jax_init_fn)
        layer(tf.ones((1,)))

    def test_with_different_argument_order(self):
        def jax_apply_fn(training, inputs, rng, state, params):
            return inputs, {}

        def jax_init_fn(training, inputs, rng):
            return {}, {}

        layer = JaxLayer(jax_apply_fn, jax_init_fn)
        layer(tf.ones((1,)))

    def test_with_minimal_arguments(self):
        def jax_apply_fn(inputs):
            return inputs

        def jax_init_fn(inputs):
            return {}

        layer = JaxLayer(jax_apply_fn, jax_init_fn)
        layer(tf.ones((1,)))

    def test_with_missing_inputs_in_apply_fn(self):
        def jax_apply_fn(params, rng, training):
            return jnp.ones((1,))

        def jax_init_fn(rng, inputs):
            return {}

        with self.assertRaisesRegex(ValueError, "`apply_fn`.*`inputs`"):
            JaxLayer(jax_apply_fn, jax_init_fn)

    def test_with_missing_inputs_in_init_fn(self):
        def jax_apply_fn(params, rng, inputs, training):
            return jnp.ones((1,))

        def jax_init_fn(rng, training):
            return {}

        with self.assertRaisesRegex(ValueError, "`init_fn`.*`inputs`"):
            JaxLayer(jax_apply_fn, jax_init_fn)

    def test_with_unsupported_argument_in_apply_fn(self):
        def jax_apply_fn(params, rng, inputs, mode):
            return jnp.ones((1,))

        def jax_init_fn(rng, inputs):
            return {}

        with self.assertRaisesRegex(ValueError, "`apply_fn`.*`mode`"):
            JaxLayer(jax_apply_fn, jax_init_fn)

    def test_with_unsupported_argument_in_init_fn(self):
        def jax_apply_fn(params, rng, inputs, training):
            return inputs

        def jax_init_fn(rng, inputs, mode):
            return {}

        with self.assertRaisesRegex(ValueError, "`init_fn`.*`mode`"):
            JaxLayer(jax_apply_fn, jax_init_fn)

    def test_with_structures_as_inputs_and_outputs(self):
        def jax_fn(params, inputs):
            a1, a2 = inputs["a"]
            b1, b2 = inputs["b"]
            output1 = jnp.concatenate([a1, b1], axis=1)
            output2 = jnp.concatenate([a2, b2], axis=1)
            return output1, output2

        layer = JaxLayer(jax_fn, params={})
        inputs = {
            "a": (keras.Input((None, 3)), keras.Input((None, 5))),
            "b": [keras.Input((None, 3)), keras.Input((None, 5))],
        }
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        test_inputs = {
            "a": (tf.ones((2, 1, 3)), tf.ones((2, 6, 5))),
            "b": [tf.ones((2, 6, 3)), tf.ones((2, 1, 5))],
        }
        test_outputs = model(test_inputs)
        self.assertAllClose(test_outputs[0], tf.ones((2, 7, 3)))
        self.assertAllClose(test_outputs[1], tf.ones((2, 7, 5)))

    def test_with_polymorphic_shape_more_than_26_dimension_names(self):
        def jax_fn(params, inputs):
            return jnp.concatenate(inputs, axis=1)

        layer = JaxLayer(jax_fn, params=())
        inputs = [keras.Input((None, 3)) for _ in range(60)]
        output = layer(inputs)
        model = keras.Model(inputs, output)

        test_inputs = [tf.ones((2, 1, 3))] * 60
        test_output = model(test_inputs)
        self.assertAllClose(test_output, tf.ones((2, 60, 3)))

    def test_with_flax_state_no_params(self):
        class MyFlaxLayer(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x):
                zeros_init = lambda s: jnp.zeros([], jnp.int32)
                count = self.variable("a", "b", zeros_init, [])
                count.value = count.value + 1
                return x

        layer = FlaxLayer(MyFlaxLayer(), variables={"a": {"b": 0}})
        layer(tf.ones((1,)))
        self.assertIsNone(layer.params)
        self.assertEqual(layer.state["a"]["b"].read_value(), tf.constant(1))

    def test_with_state_none_leaves(self):
        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(jax_fn, state={"foo": None})
        self.assertIsNone(layer.state["foo"])
        layer(tf.ones((1,)))

    def test_with_state_non_tensor_leaves(self):
        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(jax_fn, state={"foo": "bar"})
        self.assertEqual(layer.state["foo"], "bar")
        # layer cannot be invoked as jax2tf will fail on strings

    def test_with_state_jax_registered_node_class(self):
        @jax.tree_util.register_pytree_node_class
        class NamedPoint:
            def __init__(self, x, y, name):
                self.x = x
                self.y = y
                self.name = name

            def tree_flatten(self):
                return ((self.x, self.y), self.name)

            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(*children, aux_data)

        def jax_fn(params, state, inputs):
            return inputs, state

        layer = JaxLayer(jax_fn, state=[NamedPoint(1.0, 2.0, "foo")])
        layer(tf.ones((1,)))

    @parameterized.named_parameters(
        {
            "testcase_name": "sequence_instead_of_mapping",
            "init_state": [0.0],
            "error_regex": " mismatch at /: .* is not a mapping",
        },
        {
            "testcase_name": "mapping_instead_of_sequence",
            "init_state": {"state": {"foo": 0.0}},
            "error_regex": " mismatch at /state: .* is not a sequence",
        },
        {
            "testcase_name": "sequence_instead_of_variable",
            "init_state": {"state": [[0.0]]},
            "error_regex": " mismatch at /state/0: .* is not a variable",
        },
    )
    def test_state_mismatch_during_update(self, init_state, error_regex):
        def jax_fn(params, state, inputs):
            return inputs, {"state": [jnp.ones([])]}

        layer = JaxLayer(jax_fn, state=init_state)
        with self.assertRaisesRegex(ValueError, error_regex):
            layer(tf.ones((1,)))

    @parameterized.named_parameters(
        {
            "testcase_name": "no_initial_state",
            "init_state": None,
        },
        {
            "testcase_name": "missing_key_added_to_mapping",
            "init_state": {"state": {}},
        },
        {
            "testcase_name": "missing_variable_added_to_sequence",
            "init_state": {"state": {"foo": [2.0]}},
        },
    )
    def test_state_variable_creation_during_update(self, init_state):
        def jax_fn(params, state, inputs):
            return inputs, {"state": {"foo": [jnp.zeros([]), jnp.ones([])]}}

        layer = JaxLayer(jax_fn, params={}, state=init_state)
        layer(tf.ones((1,)))
        self.assertEqual(len(layer.state["state"]["foo"]), 2)
        self.assertEqual(layer.state["state"]["foo"][0].read_value(), 0.0)
        self.assertEqual(layer.state["state"]["foo"][1].read_value(), 1.0)

    def test_rng_seeding(self):
        def jax_init(rng, inputs):
            return [jax.nn.initializers.normal(1.0)(rng, inputs.shape)]

        def jax_apply(params, inputs):
            return jnp.dot(inputs, params[0])

        shape = (2, 2)

        keras.utils.set_random_seed(0)
        layer1 = JaxLayer(jax_apply, jax_init)
        layer1.build(shape)
        keras.utils.set_random_seed(0)
        layer2 = JaxLayer(jax_apply, jax_init)
        layer2.build(shape)
        self.assertAllClose(layer1.params[0], layer2.params[0])


if __name__ == "__main__":
    tf.test.main()
