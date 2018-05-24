from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import (Conv2D, Conv2DTranspose, Activation, Add,
                          BatchNormalization, Lambda)
from BilinearUpSampling import BilinearUpSampling2D


class Node(Layer):

    """Docstring for Node. """

    def __init__(self,
                 index,
                 cur_channels,
                 start_size,
                 tr_conv_kernel,
                 fabric,
                 **kwargs):
        self.index = index

        self.cur_channels = cur_channels
        self.start_size = start_size
        self.tr_conv_kernel = tr_conv_kernel
        self.fabric_size = fabric.size

        self.conv_param = dict(
            activation=None,
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        self.output_dim = self.start_size / (2**(self.index[1]))

        self.fabric = fabric

        super(Node, self).__init__(**kwargs)

    def __call__(self, x):
        tensors = []
        for in_tensor in x:
            shape = K.int_shape(in_tensor)
            if shape[1] == self.output_dim:
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] > self.output_dim:
                tensor = Conv2D(self.cur_channels, strides=2,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] < self.output_dim:
                tensor = Conv2DTranspose(self.cur_channels,
                                         self.tr_conv_kernel,
                                         strides=2,
                                         padding='same')(in_tensor)

                tensors.append(tensor)

        if len(tensors) > 1:
            tensor = Add()(tensors)
        else:
            tensor = tensors[0]
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        return tensor

    def compute_output_shape(self, input_shape):
        return (self.output_dim, self.output_dim, self.cur_channels)


class NodeResize(Layer):

    """Docstring for Node. """

    def __init__(self,
                 index,
                 cur_channels,
                 start_size,
                 tr_conv_kernel,
                 fabric,
                 **kwargs):
        self.index = index

        self.cur_channels = cur_channels
        self.start_size = start_size
        self.tr_conv_kernel = tr_conv_kernel
        self.fabric_size = fabric.size

        self.conv_param = dict(
            activation=None,
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        self.output_dim = self.start_size / (2**(self.index[1]))

        self.fabric = fabric

        super(NodeResize, self).__init__(**kwargs)

    def __call__(self, x):
        tensors = []
        for in_tensor in x:
            shape = K.int_shape(in_tensor)
            if shape[1] == self.output_dim:
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] > self.output_dim:
                tensor = Conv2D(self.cur_channels, strides=2,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] < self.output_dim:
                tensor = Lambda(
                        lambda x: K.resize_images(x, 2, 2, 'channels_last')
                        )(in_tensor)

                tensors.append(tensor)

        if len(tensors) > 1:
            tensor = Add()(tensors)
        else:
            tensor = tensors[0]
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        return tensor

    def compute_output_shape(self, input_shape):
        return (self.output_dim, self.output_dim, self.cur_channels)

        
class NodeResizeConv(Layer):

    """Docstring for Node. """

    def __init__(self,
                 index,
                 cur_channels,
                 start_size,
                 tr_conv_kernel,
                 fabric,
                 **kwargs):
        self.index = index

        self.cur_channels = cur_channels
        self.start_size = start_size
        self.tr_conv_kernel = tr_conv_kernel
        self.fabric_size = fabric.size

        self.conv_param = dict(
            activation=None,
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        self.output_dim = self.start_size / (2**(self.index[1]))

        self.fabric = fabric

        super(NodeResizeConv, self).__init__(**kwargs)

    def __call__(self, x):
        tensors = []
        for in_tensor in x:
            shape = K.int_shape(in_tensor)
            if shape[1] == self.output_dim:
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] > self.output_dim:
                tensor = Conv2D(self.cur_channels, strides=2,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] < self.output_dim:
                in_tensor = Lambda(
                        lambda x: K.resize_images(x, 2, 2, 'channels_last')
                        )(in_tensor)
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)

                tensors.append(tensor)

        if len(tensors) > 1:
            tensor = Add()(tensors)
        else:
            tensor = tensors[0]
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        return tensor

    def compute_output_shape(self, input_shape):
        return (self.output_dim, self.output_dim, self.cur_channels)

        
class NodeBi(Layer):

    """Docstring for Node. """

    def __init__(self,
                 index,
                 cur_channels,
                 start_size,
                 tr_conv_kernel,
                 fabric,
                 **kwargs):
        self.index = index

        self.cur_channels = cur_channels
        self.start_size = start_size
        self.tr_conv_kernel = tr_conv_kernel
        self.fabric_size = fabric.size

        self.conv_param = dict(
            activation=None,
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        self.output_dim = self.start_size / (2**(self.index[1]))

        self.fabric = fabric

        super(NodeBi, self).__init__(**kwargs)

    def __call__(self, x):
        tensors = []
        for in_tensor in x:
            shape = K.int_shape(in_tensor)
            if shape[1] == self.output_dim:
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] > self.output_dim:
                tensor = Conv2D(self.cur_channels, strides=2,
                                **self.conv_param)(in_tensor)
                tensors.append(tensor)

            if shape[1] < self.output_dim:
                in_tensor = BilinearUpSampling2D(
                        target_size=(self.output_dim, self.output_dim)
                        )(in_tensor)
                tensor = Conv2D(self.cur_channels,
                                strides=1,
                                **self.conv_param)(in_tensor)

                tensors.append(tensor)

        if len(tensors) > 1:
            tensor = Add()(tensors)
        else:
            tensor = tensors[0]
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)

        return tensor

    def compute_output_shape(self, input_shape):
        return (self.output_dim, self.output_dim, self.cur_channels)
