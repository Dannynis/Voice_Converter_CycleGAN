import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow_addons


def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation


class instance_norm_layer(Model):
    def __init__(self,name,epsilon = 1e-06):
        super(instance_norm_layer).__init__()
        self.instance_norm = tensorflow_addons.layers.normalizations.InstanceNormalization(epsilon=epsilon,name=name)

    def call(self,x):
        return self.instance_norm(x)

class conv1d_layer(Model):
    def __init__(self,filters,kernel_size,strides = 1,padding = 'same',activation = None,kernel_initializer = None,name = None):
        super(conv1d_layer).__init__()
        self.conv_1d_layer = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name)

    def call(self,x):
        return self.conv_1d_layer(x)


class conv2d_layer(Model):
    def __init__(self,filters,
    kernel_size,
    strides,
    padding = 'same',
    activation = None,
    kernel_initializer = None,
    name = None):
        super(conv2d_layer).__init__()
        self.conv_2d_layer = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name)

    def call(self,x):
        return self.conv_2d_layer(x)

class residual1d_block(Model):

    def __init__(self,filters = 1024,
    kernel_size = 3,
    strides = 1,name_prefix = 'residule_block_'):
        super(residual1d_block).__init__()
        self.h1 = conv1d_layer( filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
        self.h1_norm = instance_norm_layer( name = name_prefix + 'h1_norm')
        self.h1_gates =  conv1d_layer( filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
        self.h1_norm_gates =  instance_norm_layer( name = name_prefix + 'h1_norm_gates')
        self.h2 =  conv1d_layer(filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
        self.h2_norm = instance_norm_layer(name=name_prefix + 'h2_norm')
        self.name_prefix = name_prefix

    def call(self,x):
        h1 = self.h1(x)
        h1_norm = self.h1_norm(h1)
        h1_gates = self.h1_gates(h1_norm)
        h1_norm_gates = self.h1_norm_gates(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=self.name_prefix + 'h1_glu')
        h2 = self.h2(h1_glu)
        h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=self.name_prefix + 'h2_norm')
        h3 = x + h2_norm
        return h3


class downsample1d_block(Model):
    def __init__(self,filters,
    kernel_size,
    strides,
    name_prefix = 'downsample1d_block_'):
        super(downsample1d_block).__init__()
        self.h1 = conv1d_layer(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                               name=name_prefix + 'h1_conv')
        self.h1_norm = instance_norm_layer(name=name_prefix + 'h1_norm')
        self.h1_gates = conv1d_layer(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                                     name=name_prefix + 'h1_gates')
        self.h1_norm_gates = instance_norm_layer(name=name_prefix + 'h1_norm_gates')

    def call(self,x):
        h1 = self.h1(x)
        h1_norm = self.h1_norm(h1)
        h1_gates = self.h1_gates(h1_norm)
        h1_norm_gates = self.h1_norm_gates(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=self.name_prefix + 'h1_glu')
        return h1_glu

class downsample2d_block(Model):
    def __init__(self, filters,
                 kernel_size,
                 strides,
                 name_prefix='downsample1d_block_'):
        super(downsample2d_block).__init__()
        self.h1 = conv2d_layer(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                               name=name_prefix + 'h1_conv')
        self.h1_norm = instance_norm_layer(name=name_prefix + 'h1_norm')
        self.h1_gates = conv2d_layer(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                                     name=name_prefix + 'h1_gates')
        self.h1_norm_gates = instance_norm_layer(name=name_prefix + 'h1_norm_gates')

    def call(self, x):
        h1 = self.h1(x)
        h1_norm = self.h1_norm(h1)
        h1_gates = self.h1_gates(h1_norm)
        h1_norm_gates = self.h1_norm_gates(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=self.name_prefix + 'h1_glu')
        return h1_glu

class upsample1d_block(Model):
    def __init__(self,inputs,
    filters,
    kernel_size,
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):
        super(upsample1d_block).__init__()
        self.h1 = conv1d_layer(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                          name=name_prefix + 'h1_conv')
        self.h1_norm = instance_norm_layer( name=name_prefix + 'h1_norm')
        self.shuffle_size = shuffle_size
        self.name_prefix = name_prefix

        self.h1_gates = conv1d_layer( filters=filters, kernel_size=kernel_size, strides=strides,
                                activation=None, name=name_prefix + 'h1_gates')
        self.h1_shuffle_gates = pixel_shuffler(shuffle_size=shuffle_size,
                                          name=name_prefix + 'h1_shuffle_gates')
        self.h1_norm_gates = instance_norm_layer( name=name_prefix + 'h1_norm_gates')


    def call(self,x):
        h1 = self.h1(x)
        h1_shuffle = pixel_shuffler(h1,name=self.name_prefix + 'h1_shuffle',shuffle_size=self.shuffle_size)
        h1_norm = self.h1_norm(h1_shuffle)
        h1_gates =self.h1_gates(h1_norm)
        h1_shuffle_gates = pixel_shuffler(h1_gates,name = self.name_prefix + 'h1_shuffle_gates',shuffle_size = self.shuffle_size)
        h1_norm_gates = self.h1_norm_gates(h1_shuffle_gates)
        h1_glu = gated_linear_layer(inputs = h1_norm,gates=h1_norm_gates)
        return h1_glu


def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

class Generator(Model):

    def __init__(self):
        super(Generator).__init__()
        self.h1 = conv1d_layer(filters=128, kernel_size=15, strides=1, activation=None, name='h1_conv')
        self.h1_gates = conv1d_layer(filters=128, kernel_size=15, strides=1, activation=None, name='h1_conv_gates')

        # Downsample
        self.d1 = downsample1d_block( filters=256, kernel_size=5, strides=2,
                                name_prefix='downsample1d_block1_')
        self.d2 = downsample1d_block(filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        self.r1 = residual1d_block( filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block1_')
        self.r2 = residual1d_block( filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block2_')
        self.r3 = residual1d_block( filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block3_')
        self.r4 = residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block4_')
        self.r5 = residual1d_block( filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block5_')
        self.r6 = residual1d_block(filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block6_')

        # Upsample
        self.u1 = upsample1d_block(filters=1024, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample1d_block1_')
        self.u2 = upsample1d_block(filters=512, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample1d_block2_')

        # Output
        self.o1 = conv1d_layer(filters=24, kernel_size=15, strides=1, activation=None, name='o1_conv')

    def call(self,x):
        # inputs has shape [batch_size, num_features, time]
        # we need to convert it to [batch_size, time, num_features] for 1D convolution
        inputs = tf.keras.transpose(x, perm = [0, 2, 1], name = 'input_transpose')

        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(h1)
        h1_glu = gated_linear_layer(inputs=h1, gates = h1_gates, name = 'h1_glu')

        d1= self.d1(h1_glu)
        d2= self.d2(d1)
        r1 = self.r1(d2)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        r4 = self.r4(r3)
        r5 = self.r5(r4)
        r6 = self.r6(r5)

        u1 = self.u1(r6)
        u2 = self.u2(u1)
        o1 = self.o1(u2)
        o2 =  tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')
        return o2

class Discriminator(Model):

    def __int__(self,x):
        super(Discriminator).__init__()
        self.h1 = conv2d_layer( filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        self.h1_gates = conv2d_layer(filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')

        # Downsample
        self.d1 = downsample2d_block(filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        self.d2 = downsample2d_block(filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        self.d3 = downsample2d_block(filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        self.o1 = layers.Dense(units = 1, activation = tf.nn.sigmoid)

    def call(self,x):
        # inputs has shape [batch_size, num_features, time]
        # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
        inputs = tf.expand_dims(x, -1)
        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(h1)
        h1_glu = gated_linear_layer(inputs=h1,gates = h1_gates,name= 'h1_glu')
        d1 = self.d1(h1_glu)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        o1 = self.o1(d3)
        return o1



