import numpy as np
import keras
# import tensorflow as tf
# import tensorflow.keras.layers as kl
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.activations import relu


def conv_bn_act(ip, hidden_dims, use_bias, act_fn, ksize=3, kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    h = keras.layers.Conv2D(
        hidden_dims, ksize, padding='same', use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(ip)
    h = keras.layers.BatchNormalization()(h)
    if act_fn is not None:
        h = keras.layers.Activation(act_fn)(h)
    return h


def dense_bn_do(ip, hidden_dims, use_bias, act_fn, do_ratio, kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    h = keras.layers.Dense(
        hidden_dims, use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(ip)
    h = keras.layers.BatchNormalization()(h)
    if act_fn is not None:
        h = keras.layers.Activation(act_fn)(h)
    if do_ratio > 0.0:
        h = keras.layers.Dropout(do_ratio)(h)
    return h


def build_simple_cnn(n_rows, n_cols, action_space, hidden_dims=512, use_bias=False, head_dims=1024, do_ratio=0.3, act_fn=keras.activations.relu):
    ip = keras.layers.Input((n_rows, n_cols, 2))
    h = ip

    h = conv_bn_act(h, hidden_dims, use_bias, act_fn)
    h = conv_bn_act(h, hidden_dims, use_bias, act_fn)
    h = conv_bn_act(h, hidden_dims, use_bias, act_fn)
    h = conv_bn_act(h, hidden_dims, use_bias, act_fn)

    h = keras.layers.Flatten()(h)

    h = dense_bn_do(h, head_dims, use_bias, act_fn, do_ratio)
    h = dense_bn_do(h, head_dims, use_bias, act_fn, do_ratio)

    pi = keras.layers.Dense(action_space)(h)
    pi = keras.layers.Softmax()(pi)

    value = keras.layers.Dense(1)(h)
    value = keras.layers.Activation(act_fn)(value)

    model = keras.models.Model(ip, [pi, value])
    return model

#
#
#

def build_resblock(ip, hidden_dims, use_bias, act_fn):
    h = conv_bn_act(ip, hidden_dims, use_bias, act_fn, ksize=3, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)
    h = conv_bn_act(h, hidden_dims, use_bias, None, ksize=3, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)
    h = h + ip
    h = keras.layers.Activation(act_fn)(h)
    return h


def build_resnet(n_rows, n_cols, action_space, num_blocks=3, hidden_dims=256, use_bias=False, act_fn=keras.activations.relu):
    ip = keras.layers.Input((n_rows, n_cols, 2))
    h = ip

    h = conv_bn_act(h, hidden_dims, use_bias, act_fn, ksize=3, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)

    for _ in range(num_blocks):
        h = build_resblock(h, hidden_dims, use_bias, act_fn)

    pi = conv_bn_act(h, 2, use_bias, act_fn, ksize=1, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)
    pi = keras.layers.Flatten()(pi)
    pi = dense_bn_do(pi, action_space, use_bias, keras.activations.softmax, do_ratio=0.0, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)

    value = conv_bn_act(h, 1, use_bias, act_fn, ksize=1, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)
    value = keras.layers.Flatten()(value)
    value = dense_bn_do(value, 1, use_bias, keras.activations.tanh, do_ratio=0.0, kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.he_normal)

    model = keras.models.Model(ip, [pi, value])
    return model




if __name__ == "__main__":
    import reversi

    hoge = reversi.Reversi()
    state = hoge.initialize()
    x = hoge.get_nn_state(state, 1)
    x = x[np.newaxis, ...]
    print(x.shape)

    action_space = hoge.n_action_space

    # model = build_simple_cnn(6, 6, action_space)
    model = build_resnet(6, 6, action_space)

    print(model(x))
