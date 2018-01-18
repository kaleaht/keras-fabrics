from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

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
                                 name=start+'-'+end+'_PDown')
        self.conv = Conv2D(self.filters,
                           kernel_size,
                           activation=activation,
                           padding=padding,
                           name=start+'-'+end+'_CDown')

        super(DownSample, self).__init__(*kwargs)

    def __call__(self, x):
        if self.pooling:
            x = self.pool(x)

        return self.conv(x)

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
                 kernel_shape=(4, 4),
                 filter_shape=(3, 3),
                 activation='relu',
                 padding='same',
                 strides=(2, 2),
                 *kwargs):
        self.filters = filters
        self.conv_trans = Conv2DTranspose(filters,
                                          kernel_shape,
                                          strides=strides,
                                          padding=padding)
        self.conv = Conv2D(filters,
                           filter_shape,
                           activation=activation,
                           padding=padding)
        # self.add = Add()

        super(UpSample, self).__init__(*kwargs)

    def __call__(self, x):
        layer = x
        # prev_layer = x[1]

        layer = self.conv_trans(layer)
        # Fabric we sum and not catenate.
        # a = self.add([layer, prev_layer])

        return self.conv(layer)

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
                           padding=padding)

        super(SameRes, self).__init__(*kwargs)

    def __call__(self, x):
        return self.conv(x)

    # def compute_output_shape(self, input_shape):
        # shape = list(input_shape)
        # assert len(shape) == 4
        # shape[3] = self.filters

        # return tuple(shape)

