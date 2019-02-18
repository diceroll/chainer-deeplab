from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link, links
from chainer import reporter


class ModifiedClassifier(links.Classifier):

    compute_accuracy = True

    def __init__(self, predictor, lossfun=softmax_cross_entropy.softmax_cross_entropy):

        super(ModifiedClassifier, self).__init__(predictor)
        if isinstance(lossfun, dict):
            self.lossfun = lossfun['func']
            del lossfun['func']
            self.lossfun_kwargs = lossfun
        else:
            self.lossfun = lossfun
            self.lossfun_kwargs = None
        self.predictions = None
        self.gt = None

    def forward(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.predictions = None
        self.gt = None

        self.y = self.predictor(*args, **kwargs)
        if self.predictor.task == 'semantic':
            self.predictions = self.y.data.argmax(axis=1)
            self.gt = t

        if self.lossfun_kwargs is not None:
            self.loss = self.lossfun(self.y, t, **self.lossfun_kwargs)
        else:
            self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)

        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'acc': self.accuracy}, self)

        if 'mean_squared_error' in str(self.lossfun):
            self.rmse = self.loss.data ** 0.5
            reporter.report({'rmse': self.rmse}, self)

        return self.loss
