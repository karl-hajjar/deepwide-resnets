from torch import nn
from pytorch_lightning import LightningModule
from collections import OrderedDict

from layers.residual import Residual
from utils.nn import *


class ResNet(LightningModule):
    """
    A class defining a fully-connected residual network of arbitrary depth, width and inner layers dimension.
    """
    def __init__(self, input_dim: int, depth: int, width: [int, None], d_model: [int, None],
                 activation: [str, None] = None, bias=False, alpha=1.0, scale=1.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self._set_inner_layer_dimensions(width, d_model)
        if activation is None:
            self.activation_name = DEFAULT_ACTIVATION
        else:
            self.activation_name = activation
        self.activation = ACTIVATION_DICT[self.activation_name](**kwargs)
        self.bias = bias
        self.alpha = alpha  # scalar multiplier for the residual connection
        self.scale = scale  # scalar multiplier for the output of the network (to accommodate NTK of MF-regime)

        self._build_model(**kwargs)  # define input and output layer attributes
        self.initialize_parameters(**kwargs)  # initialize with a custom init

    def _set_inner_layer_dimensions(self, width, d_model):
        if width is None:
            if d_model is None:
                raise ValueError("`width` and `d_model` arguments cannot simultaneously be None")
            else:
                self.width = d_model
                self.d_model = d_model

        else:
            if d_model is None:
                self.width = width
                self.d_model = width
            else:
                self.width = width
                self.d_model = d_model

    def _build_model(self, **kwargs):
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=self.d_model, bias=self.bias)
        self.residual_layers = nn.Sequential(OrderedDict([
            ('residual_{}'.format(l), Residual(d=self.d_model, width=self.width, activation=self.activation_name,
                                               bias=self.bias, alpha=self.alpha, **kwargs))
            for l in range(1, self.depth)
        ]))
        self.output_layer = nn.Linear(in_features=self.d_model, out_features=1, bias=self.bias)

    def initialize_parameters(self, kind='he', mode='fan_in'):
        if kind == 'he':
            torch.nn.init.kaiming_normal_(self.input_layer.weight, mode=mode, nonlinearity=self.activation_name)
            torch.nn.init.kaiming_normal_(self.output_layer.weight, mode=mode, nonlinearity=self.activation_name)

        else:
            with torch.no_grad():
                self.input_layer.weight.data.copy_(torch.randn(size=(self.d_model, self.input_dim)))
                self.output_layer.weight.data.copy_(torch.randn(size=(1, self.d_model)))

        if self.bias:
            with torch.no_grad():
                self.input_layer.bias.data.copy_(torch.randn(size=(self.d_model,)))
                self.output_layer.bias.data.copy_(torch.randn(size=(1,)))

    def forward(self, x):
        h = self.input_layer(x)
        h = self.residual_layers(h)
        return self.scale * self.output_layer(h)
