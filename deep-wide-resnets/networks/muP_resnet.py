from torch import nn
from collections import OrderedDict
from numpy import sqrt

from resnet import ResNet
from layers.muP_residual import MuPResidual
from utils.nn import *


class MuPResNet(ResNet):
    """
    A class defining a fully-connected residual network of arbitrary depth, width and inner layers dimension in the
    Maximal Update Parameterization (MuP, allowing maximal feature learning at infinite width).
    """
    def __init__(self, input_dim: int, n_res: int, width: [int, None] = None, d_model: [int, None] = None,
                 activation: [str, None] = None, bias=False, alpha=1.0, **kwargs):
        act_kwargs = {key: value for key, value in kwargs.items() if key not in self.INIT_KEYS}
        super().__init__(input_dim, n_res, width, d_model, activation, bias, alpha, **act_kwargs)

    def _build_model(self, **kwargs):
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=self.d_model, bias=self.bias)
        self.residual_layers = nn.Sequential(OrderedDict([
            ('residual_{}'.format(l), MuPResidual(d=self.d_model, width=self.width, activation=self.activation_name,
                                                  bias=self.bias, alpha=self.alpha, **kwargs))
            for l in range(1, self.n_res+1)
        ]))
        self.output_layer = nn.Linear(in_features=self.d_model, out_features=1, bias=self.bias)

    def initialize_parameters(self):
        with torch.no_grad():
            self.input_layer.weight.data.copy_(torch.randn(size=(self.d_model, self.input_dim)) /
                                               sqrt(self.input_dim * self.d_model))
            self.output_layer.weight.data.copy_(torch.randn(size=(1, self.d_model)) / sqrt(self.d_model))

        if self.bias:
            with torch.no_grad():
                self.input_layer.bias.data.copy_(torch.randn(size=(self.d_model,)))
                self.output_layer.bias.data.copy_(torch.randn(size=(1,)))

    def forward(self, x):
        h = sqrt(self.d_model) * self.input_layer(x)
        h = self.residual_layers(h)
        return self.output_layer(h) / sqrt(self.d_model)
