from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from layers import DownSample, UpSample, SameRes


class Node():

    def __init__(self,
                 name,
                 *kwargs):
        self.incoming_tensors = []
        self.tensor = None
        self.name = name

        super(Node, self).__init__(*kwargs)

    def __call__(self):
        if self.tensor is None:
            if len(self.incoming_tensors) == 1:
                self.tensor = self.incoming_tensors[0]
            else:
                self.tensor = Add(name=self.name)(self.incoming_tensors)

        return self.tensor

    def add(self, t):
        self.incoming_tensors.append(t)


class Fabric():

    def __init__(self,
                 input_shape,
                 size,
                 channels,
                 channels_double=False,
                 kernel_shape=(4, 4),
                 pool_size=(2, 2),
                 filter_shape=(3, 3),
                 activation='relu',
                 padding='same',
                 stride=(2, 2),
                 *kwargs):
        self.kernel_shape = kernel_shape
        self.channels = channels
        self.channels_double = channels_double
        self.size = size
        self.pool_size = pool_size
        self.filter_shape = filter_shape
        self.activation = activation
        self.padding = padding
        self.stride = stride

        self.conv_param = dict(
            activation='relu',
            kernel_size=(3,3),
            padding='same',
            use_bias=False
        )

        self.fabric = self.init_fabric(size)
        inputs = Input(input_shape, name='start')
        self.model = self.populate_fabric(inputs)

        super(Fabric, self).__init__(*kwargs)
        # self.fabric = create_fabric(size)

    def init_fabric(self, size):
        fabric = []
        for layer in range(size[0]):
            fabric.append([])
            for scale in range(size[1]):
                fabric[layer].append(Node(name=str(layer)+str(scale)))

        return fabric

    def populate_fabric(self, inputs):
        # First layer

        layer = 0
        start = 'start'
        tensor = inputs
        cur_channels = self.channels
        for scale in range(self.size[1]):
            end = str(layer)+str(scale)
            tensor = DownSample(cur_channels,
                                start,
                                end,
                                pooling=(scale != 0))(tensor)
            start = end
            node = self.fabric[layer][scale]
            node.add(tensor)
            tensor = node()     # Run add
            if self.channels_double:
                cur_channels = cur_channels * 2

        layer += 1

        # Intermidiate layers
        for layers in range(layer, self.size[0]-1):
            cur_channels = self.channels
            for scale in range(self.size[1]):
                node = self.fabric[layer][scale]
                if (scale < self.size[1]-1):    # After first row
                    incoming_tensor = UpSample(cur_channels, self.conv_param)(self.fabric[layer-1][scale+1]())
                    node.add(incoming_tensor)
                if (scale > 0):    # Before last row
                    previous_node = self.fabric[layer-1][scale-1]

                    incoming_tensor = DownSample(cur_channels, previous_node.name, str(layer)+str(scale))(previous_node())
                    node.add(incoming_tensor)

                incoming_tensor = SameRes(cur_channels)(self.fabric[layer-1][scale]())
                node.add(incoming_tensor)
            if self.channels_double:
                cur_channels = cur_channels * 2

            layer += 1

        # Last layer
        for scale in range(self.size[1]-1, -1, -1):
            cur_channels = int(np.sqrt(cur_channels))
            node = self.fabric[layer][scale]
            if (scale < self.size[1]-1):    # After first row
                incoming_tensor = UpSample(cur_channels, self.conv_param)(self.fabric[layer-1][scale+1]())
                node.add(incoming_tensor)
                incoming_tensor = UpSample(cur_channels, self.conv_param)(self.fabric[layer][scale+1]())
                node.add(incoming_tensor)
            if (scale > 0):    # Before last row
                previous_node = self.fabric[layer-1][scale-1]
                incoming_tensor = DownSample(cur_channels, previous_node.name, str(layer)+str(scale))(previous_node())
                node.add(incoming_tensor)

            incoming_tensor = SameRes(cur_channels)(self.fabric[layer-1][scale]())
            node.add(incoming_tensor)

        conv = Conv2D(3, (1, 1), activation='softmax')(self.fabric[layer][0]())

        model = Model(inputs=[inputs], outputs=[conv])
        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])

        return model
