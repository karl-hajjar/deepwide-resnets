from numpy import sqrt
from torch.nn.init import calculate_gain

import haiku as hk
from utils.jax.nn import *


class Residual(hk.Module):
    """
    A class defining a residual block in JAX consisting of two fully-connected layers followed by a residual connection:
    that is, for an input x, the output of the residual layer is x + alpha * W_2 phi(W_1 x + b_1) + b_2.
    """
    INIT_KEYS = ['kind', 'mode', 'std']
    INIT_KINDS = {'he', 'glorot', 'sphere', 'reproduce', 'gaussian'}  # set

    def __init__(self, d: int, width: int, activation: [str, None] = None, bias=False, alpha=1.0, name='ResidualLayer',
                 **kwargs):
        super().__init__(name=name)
        act_kwargs = {key: value for key, value in kwargs.items() if key not in self.INIT_KEYS}  # not used for now
        init_kwargs = {key: value for key, value in kwargs.items() if key in self.INIT_KEYS}

        self.d = d  # dimension of the input and output of the residual block
        self.width = width
        if activation is None:
            self.activation_name = DEFAULT_ACTIVATION
        else:
            self.activation_name = activation
        self.activation = ACTIVATION_DICT[self.activation_name]
        self.bias = bias

        self.alpha = alpha  # multiplier for the residual connection

        self._build_model()  # define first and second layer of the residual block
        self.initialize_parameters(**init_kwargs)  # initialize with a custom init

    def _build_model(self):
        self.first_layer = hk.Linear(output_size=self.width, with_bias=self.bias, name='first_layer')
        self.second_layer = hk.Linear(output_size=self.d, with_bias=self.bias, name='second_layer')

    def initialize_parameters(self, kind='gaussian', mode='fan_in', std=None):
        if kind not in self.INIT_KINDS:
            raise ValueError("argument `kind` must be in {} but was {}".format(self.INIT_KINDS, kind))

        else:
            if kind == 'he':
                # see https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.initializers.VarianceScaling for the
                # variance scaling init scheme adn the correspondence with classical initialization schemes (He, Glorot)
                # Note that in the link above, there is no precision of the adaptation of the scale according to the
                # activation function / non-linearity.
                self.first_layer.w_init = hk.initializers.VarianceScaling(scale=1.0, mode=mode, distribution='normal')
                gain = calculate_gain(nonlinearity=self.activation_name)  # scale of haiku = gain^2 of pytorch
                self.second_layer.w_init = hk.initializers.VarianceScaling(scale=gain**2, mode=mode,
                                                                           distribution='normal')

            elif kind == 'glorot':
                self.first_layer.w_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                          distribution='uniform')
                self.second_layer.w_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                           distribution='uniform')

            elif kind == 'sphere':
                pass
                # not implemented for now

            elif kind == 'reproduce':
                self.first_layer.w_init = hk.initializers.UniformScaling(scale=1.0)
                self.second_layer.w_init = hk.initializers.UniformScaling(scale=1.0)

            elif kind == 'gaussian':
                if std is None:
                    std = self._get_init_std(self.activation_name)
                    self.first_layer.w_init = hk.initializers.RandomNormal(stddev=std / sqrt(self.d))
                    self.second_layer.w_init = hk.initializers.RandomNormal(stddev=std / sqrt(self.width))

        if self.bias:
            self.first_layer.b_init = hk.initializers.RandomNormal(stddev=1.0)
            self.second_layer.b_init = hk.initializers.RandomNormal(stddev=1.0)

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

    def __call__(self, x):
        return x + self.alpha * self.second_layer(self.activation(self.first_layer(x)))
