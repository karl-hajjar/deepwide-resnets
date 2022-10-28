from torch import nn
from pytorch_lightning import LightningModule
from numpy import sqrt

from utils.nn import *


class Residual(LightningModule):
    """
    A class defining a residual block consisting of two fully-connected layers followed by a residual connection: that
    is, for an input x, the output of the residual layer is x + alpha * W_2 phi(W_1 x + b_1) + b_2.
    """
    INIT_KEYS = ['kind', 'mode', 'std']
    INIT_KINDS = {'he', 'glorot', 'sphere', 'reproduce', 'gaussian'}  # set

    def __init__(self, d: int, width: int, activation: [str, None] = None, bias=False, alpha=1.0, **kwargs):
        super().__init__()
        act_kwargs = {key: value for key, value in kwargs.items() if key not in self.INIT_KEYS}
        init_kwargs = {key: value for key, value in kwargs.items() if key in self.INIT_KEYS}

        self.d = d  # dimension of the input and output of the residual block
        self.width = width
        if activation is None:
            self.activation_name = DEFAULT_ACTIVATION
        else:
            self.activation_name = activation
        self.activation = ACTIVATION_DICT[self.activation_name](**act_kwargs)
        self.bias = bias

        self.alpha = alpha  # multiplier for the residual connection

        self._build_model()  # define first and second layer of the residual block
        self.initialize_parameters(**init_kwargs)  # initialize with a custom init

    def _build_model(self):
        self.first_layer = nn.Linear(in_features=self.d, out_features=self.width, bias=self.bias)
        self.second_layer = nn.Linear(in_features=self.width, out_features=self.d, bias=self.bias)

    def initialize_parameters(self, kind='gaussian', mode=None, std=None):
        if kind not in self.INIT_KINDS:
            raise ValueError("argument `kind` must be in {} but was {}".format(self.INIT_KINDS, kind))

        else:
            if kind == 'he':
                if mode not in ['fan_in', 'fan_out']:
                    raise ValueError("`mode`argument must one of {{'fan_in', 'fan_out'}}, but was '{}'".format(mode))
                torch.nn.init.kaiming_normal_(self.first_layer.weight, mode=mode, nonlinearity=self.activation_name)
                torch.nn.init.kaiming_normal_(self.second_layer.weight, mode=mode, nonlinearity=self.activation_name)

            elif kind == 'glorot':
                torch.nn.init.xavier_uniform_(self.first_layer.weight)
                torch.nn.init.xavier_uniform_(self.second_layer.weight)

            elif kind == 'sphere':
                with torch.no_grad():
                    weights_1 = generate_uniform_sphere_weights(width=self.width, d=self.d)
                    weights_2 = generate_bernouilli_weights(width=self.width)
                    self.first_layer.weight.data.copy_(weights_1.data)
                    self.second_layer.weight.data.copy_(weights_2.data)

            elif kind == 'reproduce':
                with torch.no_grad():
                    self.first_layer.weight.data.copy_(sqrt(3 / self.d) *
                                                       (2 * torch.rand(size=(self.width, self.d)) - 1))
                    self.second_layer.weight.data.copy_(sqrt(3 / self.width) *
                                                        (2 * torch.rand(size=(self.d, self.width)) - 1))
            elif kind == 'gaussian':
                if std is None:
                    std = self._get_init_std(self.activation_name)
                with torch.no_grad():
                    self.first_layer.weight.data.copy_(torch.randn(size=(self.width, self.d)) / sqrt(self.d))
                    self.second_layer.weight.data.copy_(std * torch.randn(size=(self.d, self.width)) / sqrt(self.width))

        if self.bias:
            with torch.no_grad():
                self.first_layer.bias.data.copy_(torch.randn(size=(self.width, )))
                self.second_layer.bias.data.copy_(torch.randn(size=(self.d, )))

    @staticmethod
    def _get_init_std(activation=None):
        var = 1.0  # default value for the variance
        if activation is not None:
            if activation == 'relu':
                var = 2.0
            elif activation == 'gelu':
                var = 4.0
            elif activation in ['elu', 'tanh']:
                var = 1.0

        return sqrt(var)

    def forward(self, x):
        return x + self.alpha * self.second_layer(self.activation(self.first_layer(x)))
