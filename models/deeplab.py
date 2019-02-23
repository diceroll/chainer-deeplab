import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy as np

from models.backbone import xception, mobilenet


def backbone_module(backbone, output_stride):
    if backbone == 'resnet':
        raise NotImplementedError
    elif backbone == 'xception':
        return xception.Xception(output_stride=output_stride)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2()
    else:
        raise ValueError


class ASPPModule(chainer.Chain):

    def __init__(self, n_input, n_output, ksize, pad, dilate):
        super(ASPPModule, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(n_input, n_output, ksize=ksize, pad=pad, dilate=dilate, nobias=True)
            self.bn = L.BatchNormalization(n_output)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))

        return h


class ASPP(chainer.Chain):

    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        if backbone in ['resnet', 'xception']:
            n_input = 2048
        elif backbone == 'mobilenet':
            n_input = 320
        else:
            raise ValueError

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise ValueError

        with self.init_scope():
            self.aspp1 = ASPPModule(n_input, 256, ksize=1, pad=0, dilate=dilations[0])
            self.aspp2 = ASPPModule(n_input, 256, ksize=3, pad=dilations[1], dilate=dilations[1])
            self.aspp3 = ASPPModule(n_input, 256, ksize=3, pad=dilations[2], dilate=dilations[2])
            self.aspp4 = ASPPModule(n_input, 256, ksize=3, pad=dilations[3], dilate=dilations[3])

            self.conv1 = L.Convolution2D(n_input, 256, ksize=1, nobias=True)
            self.bn1 = L.BatchNormalization(256)

            self.conv2 = L.Convolution2D(1280, 256, ksize=1, nobias=True)
            self.bn2 = L.BatchNormalization(256)

    def forward(self, x):
        h1 = self.aspp1(x)
        h2 = self.aspp2(x)
        h3 = self.aspp3(x)
        h4 = self.aspp4(x)
        h5 = F.average(x, axis=(2, 3), keepdims=True)
        h5 = F.relu(self.bn1(self.conv1(h5)))
        h5 = F.resize_images(h5, h4.shape[2:])
        h = F.concat((h1, h2, h3, h4, h5))

        h = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(h)

        return h


class Decoder(chainer.Chain):

    def __init__(self, n_class, backbone):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            # n_low_level_features = 256
            raise NotImplementedError
        elif backbone == 'xception':
            n_low_level_features = 128
        elif backbone == 'mobilenet':
            n_low_level_features = 24
        else:
            raise ValueError

        with self.init_scope():
            self.conv1 = L.Convolution2D(n_low_level_features, 48, ksize=1, nobias=True)
            self.bn1 = L.BatchNormalization(48)
            self.conv2 = L.Convolution2D(304, 256, ksize=3, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(256, 256, ksize=3, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(256, n_class, ksize=1)

    def forward(self, x, low_level_features):
        low_level_features = F.relu(self.bn1(self.conv1(low_level_features)))

        h = F.resize_images(x, low_level_features.shape[2:])
        h = F.concat((h, low_level_features))

        h = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(h, ratio=0.5)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.dropout(h, ratio=0.1)
        h = self.conv4(h)

        return h


class DeepLab(chainer.Chain):

    def __init__(self, n_class, task='semantic', backbone='xception', output_stride=16):
        super(DeepLab, self).__init__()
        self.task = task
        with self.init_scope():
            self.backbone = backbone_module(backbone, output_stride)
            self.aspp = ASPP(backbone, output_stride)
            self.decoder = Decoder(n_class, backbone)

    def forward(self, x):
        h, low_level_features = self.backbone(x)
        h = self.aspp(h)
        h = self.decoder(h, low_level_features)
        if h.shape != x.shape:
            h = F.resize_images(h, x.shape[2:])

        return h


if __name__ == '__main__':
    model = DeepLab(1)
    chainer.cuda.get_device(0).use()
    model.to_gpu()
    x = chainer.Variable(cupy.zeros((2, 3, 513, 513), cupy.float32))
    y = model(x)
    print(x.shape, y.shape)
