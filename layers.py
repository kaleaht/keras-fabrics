from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.engine.topology import Layer
import tensorflow as tf


class DownSample():

    def __init__(self,
                 filters,
                 start,
                 end,
                 pooling=True,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 pool_size=(2, 2),
                 *kwargs):

        self.pooling = pooling
        self.filters = filters
        self.pool = MaxPooling2D(pool_size=pool_size,
                                 name=start + '-' + end + '_PDown')
        self.conv = Conv2D(self.filters,
                           kernel_size,
                           activation=activation,
                           padding=padding,
                           name=start + '-' + end + '_CDown',
                           use_bias=False)

        super(DownSample, self).__init__(*kwargs)

    def __call__(self, x):
        if self.pooling:
            x = self.pool(x)
        x = self.conv(x)

        return x

    # def compute_output_shape(self, input_shape):
        # shape = list(input_shape)
        # assert len(shape) == 4
        # if self.pooling:
        # shape[1] /= 2
        # shape[2] /= 2
        # shape[3] = self.filters

        # return tuple(shape)


class UpSample():

    def __init__(self,
                 filters,
                 conv_param,
                 kernel_shape,
                 padding='same',
                 strides=(2, 2),
                 *kwargs):
        self.filters = filters
        self.conv_trans = Conv2DTranspose(filters,
                                          kernel_shape,
                                          strides=strides,
                                          padding=padding)
        # self.norm = BatchNormalization()
        self.conv = Conv2D(filters,
                           **conv_param)

        super(UpSample, self).__init__(*kwargs)

    def __call__(self, x):
        x = self.conv_trans(x)
        x = self.conv(x)
        # x = self.norm(x)

        return x

    # def compute_output_shape(self, input_shape):
        # input_shape = input_shape[0]
        # shape = list(input_shape)
        # assert len(shape) == 4
        # shape[1] *= 2
        # shape[2] *= 2
        # shape[3] = self.filters

        # return tuple(shape)


class SameRes():

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 *kwargs):
        self.filters = filters
        self.conv = Conv2D(filters,
                           kernel_size,
                           activation=activation,
                           padding=padding,
                           use_bias=False)
        # self.norm = BatchNormalization()
        super(SameRes, self).__init__(*kwargs)

    def __call__(self, x):
        x = self.conv(x)
        # x = self.norm(x)

        return x

    # def compute_output_shape(self, input_shape):
        # shape = list(input_shape)
        # assert len(shape) == 4
        # shape[3] = self.filters

        # return tuple(shape)

class Bilinear(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Bilinear, self).__init__(**kwargs)

    def call(self, tensor, endpoint=False):
        return tf.image.resize_images(tensor,
                                      self.output_dim,
                                      tf.image.ResizeMethod.BILINEAR)

    def compute_output_shape(self, input_shape):
        return (input_shape[1]*2, input_shape[2]*2, input_shape[3])

class Unpooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(Unpooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize,
            "ignore_border":self.ignore_border}

