import chainer
import chainer.functions as F
import chainer.links as L


class Block(chainer.Chain):

    def __init__(self, n_input, n_output, expantion, ksize=3, stride=1, dilate=1, nobias=True):
        super(Block, self).__init__()
        pad = ((ksize - 1) * dilate) // 2
        n_hidden = n_input * expantion
        self.stride = stride
        self.skip_connection = (n_input == n_output)

        with self.init_scope():
            self.conv1 = L.Convolution2D(n_input, n_hidden, ksize=1, nobias=nobias)
            self.bn1 = L.BatchNormalization(n_hidden)
            self.conv2 = L.Convolution2D(n_hidden, n_hidden, ksize=ksize, stride=stride, pad=pad,
                                         dilate=dilate, groups=n_hidden, nobias=nobias)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.conv3 = L.Convolution2D(n_hidden, n_output, ksize=1, nobias=nobias)
            self.bn3 = L.BatchNormalization(n_output)

    def forward(self, x):
        h = F.clipped_relu(self.bn1(self.conv1(x)), z=6.)
        h = F.clipped_relu(self.bn2(self.conv2(h)), z=6.)
        h = self.bn3(self.conv3(h))
        if self.stride == 1 and self.skip_connection:
            h = h + x

        return h


class MobileNetV2(chainer.Chain):

    def __init__(self):
        super(MobileNetV2, self).__init__()
        params = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        ]

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, ksize=3, stride=2, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32)

            self._forward = []
            n_input = 32
            for i, (t, c, n, s) in enumerate(params):
                for j in range(n):
                    name = 'block{}_{}'.format(i + 1, j + 1)
                    if j == 0:
                        block = Block(n_input, c, t, stride=s)
                        n_input = c
                    else:
                        block = Block(n_input, n_input, t)
                    setattr(self, name, block)
                    self._forward.append(name)

    def forward(self, x):
        h = F.clipped_relu(self.bn1(self.conv1(x)), z=6.)
        for name in self._forward:
            if name == 'block3_1':
                low_level_features = h
            block = getattr(self, name)
            h = block(h)

        return h, low_level_features
