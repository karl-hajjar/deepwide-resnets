from layers.jax.residual import Residual


class MuPResidual(Residual):
    """
    A class using Jax and Haiku defining a residual block consisting of two fully-connected layers in the Maximal Update
    (MuP) parameterization followed by a residual connection: that is, for an input x, the output of the residual layer
    is x + alpha * W_2 phi(W_1 x + b_1) + b_2.
    """
    INIT_KEYS = ['kind', 'mode', 'std']
    INIT_KINDS = {'he', 'glorot', 'sphere', 'reproduce', 'gaussian'}  # set

    def __init__(self, d: int, width: int, activation: [str, None] = None, bias=False, alpha=1.0, **kwargs):
        act_kwargs = {key: value for key, value in kwargs.items() if key not in self.INIT_KEYS}
        super().__init__(d, width, activation, bias, alpha, kind='gaussian', **act_kwargs)
