from keras.layers import Conv2D, Input, Lambda
from keras.models import Model


class Fabric():

    """Docstring for Fabric. """

    def __init__(self,
                 node,
                 input_shape,
                 size,
                 cur_channels,
                 tr_conv_kernel,
                 *kwargs):
        self.cur_channels = cur_channels
        self.tr_conv_kernel = tr_conv_kernel
        self.size = size
        self.input_shape = input_shape
        self.inputs = None

        # Initialising fabric with nodes.
        self.fabric = self.init_network(node, size)

        self.tensors = [[None] * size[1] for _ in range(size[0])]

        # Run tensors
        self.model = self.create_network()

        super(Fabric, self).__init__(*kwargs)

    def init_network(self, node, size):
        fabric = []
        for layer in range(size[0]):
            fabric.append([])
            for scale in range(size[1]):
                fabric[layer].append(node((layer, scale),
                                          self.cur_channels,
                                          self.input_shape[0],
                                          self.tr_conv_kernel,
                                          self))

        return fabric

    def run_node(self, scale, layer):
        num_layers, _ = self.size
        temp_tensors = []

        # Getting previous tensors.
        if layer == 0:
            # First layer only down sample
            temp_tensors = self.first_layer(scale, layer)
        elif layer == num_layers - 1:
            # Last layer only up sample
            temp_tensors = self.last_layer(scale, layer)
        else:
            # Intermediate layers up sample, down sample and same res
            temp_tensors = self.intermidiate_layer(scale, layer)

        # Running tensor
        self.tensors[layer][scale] = self.fabric[layer][scale](temp_tensors)

    def first_layer(self, scale, layer):
        if scale == 0:
            # Get input
            self.inputs = [Input(self.input_shape)]
            return self.inputs

        # Get previous scale in same layer
        return [self.get_tensor(scale-1, layer)]

    def intermidiate_layer(self, scale, layer):
        res = []
        if scale > 0:
            # Get input
            res.append(self.get_tensor(scale-1, layer-1))
        if scale != self.size[1] - 1:
            # Get previous scale in same layer
            res.append(self.get_tensor(scale+1, layer-1))

        # Always same res
        res.append(self.get_tensor(scale, layer-1))
        return res

    def last_layer(self, scale, layer):
        res = []
        if scale > 0:
            # Get input
            res.append(self.get_tensor(scale-1, layer-1))
        if scale != self.size[1] - 1:
            # Get previous scale in previous layer
            res.append(self.get_tensor(scale+1, layer-1))
            # Get previous scale in same layer
            res.append(self.get_tensor(scale+1, layer))

        # Always same res
        res.append(self.get_tensor(scale, layer-1))
        return res

    def get_tensor(self, scale, layer):
        return self.tensors[layer][scale]

    def create_network(self):

        for scale in range(self.size[1]):
            self.run_node(scale, 0)

        layer = 1

        # Intermediate layers
        for _ in range(layer, self.size[0] - 1):
            for scale in range(self.size[1]):
                self.run_node(scale, layer)

            layer += 1

        # Last layer
        for scale in range(self.size[1] - 1, -1, -1):
            self.run_node(scale, layer)

        conv = Conv2D(3, (1, 1), activation='softmax')(self.tensors[layer][0])
        conv = Lambda(lambda x: x[:, 3:-3, 3:-3])(conv)

        model = Model(inputs=self.inputs, outputs=[conv])

        return model
