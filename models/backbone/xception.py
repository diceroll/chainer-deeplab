import chainer
import chainer.functions as F
import chainer.links as L


class SeparableConv2d(chainer.Chain):

    def __init__(self, n_input, n_output, ksize=3, stride=1, dilate=1, nobias=True):
        super(SeparableConv2d, self).__init__()
        pad = ((ksize - 1) * dilate) // 2

        with self.init_scope():
            self.conv1 = L.Convolution2D(n_input, n_input, ksize=ksize, stride=stride, pad=pad,
                                         dilate=dilate, groups=n_input, nobias=nobias)
            self.bn = L.BatchNormalization(n_input)
            self.conv2 = L.Convolution2D(n_input, n_output, ksize=1, nobias=nobias)

    def forward(self, x):
        h = self.bn(self.conv1(x))
        h = self.conv2(h)

        return h


class EntryBlock(chainer.Chain):

    def __init__(self, n_input, n_output, stride=1, dilate=1, nobias=True):
        super(EntryBlock, self).__init__()
        with self.init_scope():
            self.conv1 = SeparableConv2d(n_input, n_output, ksize=3, stride=1, dilate=dilate, nobias=nobias)
            self.bn1 = L.BatchNormalization(n_output)
            self.conv2 = SeparableConv2d(n_output, n_output, ksize=3, stride=1, dilate=dilate, nobias=nobias)
            self.bn2 = L.BatchNormalization(n_output)
            self.conv3 = SeparableConv2d(n_output, n_output, ksize=3, stride=stride, nobias=nobias)
            self.bn3 = L.BatchNormalization(n_output)

            self.conv4 = L.Convolution2D(n_input, n_output, ksize=1, stride=stride, nobias=nobias)
            self.bn4 = L.BatchNormalization(n_output)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))

        h2 = self.bn4(self.conv4(x))

        return h1 + h2


class MiddleBlock(chainer.Chain):

    def __init__(self, n_input, n_output, n_layer=3, dilate=1, nobias=True):
        super(MiddleBlock, self).__init__()
        self._forward = []
        with self.init_scope():
            for i in range(n_layer):
                name_conv = 'conv{}'.format(i + 1)
                conv = SeparableConv2d(n_input, n_output, ksize=3, stride=1, dilate=dilate, nobias=nobias)
                setattr(self, name_conv, conv)
                name_bn = 'bn{}'.format(i + 1)
                bn = L.BatchNormalization(n_output)
                setattr(self, name_bn, bn)
                n_input = n_output
                self._forward.append((name_conv, name_bn))

    def forward(self, x):
        for name_conv, name_bn in self._forward:
            conv = getattr(self, name_conv)
            bn = getattr(self, name_bn)
            h = F.relu(x)
            h = conv(h)
            h = bn(h)

        return h + x


class ExitBlock(chainer.Chain):

    def __init__(self, n_input, n_output, stride=1, dilate=1, nobias=True):
        super(ExitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = SeparableConv2d(n_input, n_input, ksize=3, stride=1, dilate=dilate, nobias=nobias)
            self.bn1 = L.BatchNormalization(n_input)
            self.conv2 = SeparableConv2d(n_input, n_output, ksize=3, stride=1, dilate=dilate, nobias=nobias)
            self.bn2 = L.BatchNormalization(n_output)
            self.conv3 = SeparableConv2d(n_output, n_output, ksize=3, stride=stride, nobias=nobias)
            self.bn3 = L.BatchNormalization(n_output)

            self.conv4 = L.Convolution2D(n_input, n_output, ksize=1, stride=stride, nobias=nobias)
            self.bn4 = L.BatchNormalization(n_output)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))

        h2 = self.bn4(self.conv4(x))

        return h1 + h2


class Xception(chainer.Chain):

    def __init__(self, output_stride=16, n_repeat=16):
        super(Xception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilate = 1
            exit_flow_dilate = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilate = 2
            exit_flow_dilate = (2, 4)
        else:
            raise NotImplementedError

        with self.init_scope():
            # Entry flow
            self.conv1 = L.Convolution2D(3, 32, ksize=3, stride=2, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(64)

            self.entry_block1 = EntryBlock(64, 128, stride=2)
            self.entry_block2 = EntryBlock(128, 256, stride=2)
            self.entry_block3 = EntryBlock(256, 728, stride=entry_block3_stride)

            # Middle flow
            self._forward = []
            for i in range(n_repeat):
                name = 'middle_block{}'.format(i + 1)
                block = MiddleBlock(728, 728, n_layer=3, dilate=middle_block_dilate)
                setattr(self, name, block)
                self._forward.append(name)

            # Exit flow
            self.exit_block = ExitBlock(728, 1024, dilate=exit_flow_dilate[0])
            self.conv3 = SeparableConv2d(1024, 1536, ksize=3, stride=1, dilate=exit_flow_dilate[1])
            self.bn3 = L.BatchNormalization(1536)
            self.conv4 = SeparableConv2d(1536, 1536, ksize=3, stride=1, dilate=exit_flow_dilate[1])
            self.bn4 = L.BatchNormalization(1536)
            self.conv5 = SeparableConv2d(1536, 2048, ksize=3, stride=1, dilate=exit_flow_dilate[1])
            self.bn5 = L.BatchNormalization(2048)

    def forward(self, x):
        # Entry flow
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        low_level_features = F.relu(self.entry_block1(h))
        h = F.relu(self.entry_block2(low_level_features))
        h = self.entry_block3(h)

        # Middle flow
        for name in self._forward:
            block = getattr(self, name)
            h = block(h)

        # Exit flow
        h = F.relu(h)
        h = F.relu(self.exit_block(h))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))

        return h, low_level_features
