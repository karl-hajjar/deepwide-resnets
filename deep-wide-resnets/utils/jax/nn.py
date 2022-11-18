import jax


def jax_identity(x: jax.numpy.array) -> jax.numpy.array:
    return x


ACTIVATION_DICT = {'relu': jax.nn.relu,
                   'elu': jax.nn.elu,
                   'gelu': jax.nn.gelu,
                   'sigmoid': jax.nn.sigmoid,
                   'tanh': jax.nn.tanh,
                   'leaky_relu': jax.nn.leaky_relu,
                   'identity': jax_identity}
DEFAULT_ACTIVATION = "relu"


def generate_uniform_sphere_weights(width: int, d: int) -> jax.numpy.array:
    """
    Generates `width` iid (input) neurons in R^d uniformly sampled on the unit sphere
    :param width: number of neurons in the layer
    :param d: dimension of the data of the weights
    :return: a tensor with width rows and d columns containing the neurons
    """
    # x = torch.randn(size=(width, d), requires_grad=False)
    # row_norms = torch.linalg.vector_norm(x, dim=1)
    # return x / row_norms.reshape(x.shape[0], 1)


def generate_bernouilli_weights(width: int) -> jax.numpy.array:
    """
    Generates one (output) neuron with `width` coordinates independently sampled from {-1,1}
    :param width:
    :return: a tensor with one row and `width` columns containing the neuron
    """
    # return 2 * torch.randint(low=0, high=2, size=(width,), requires_grad=False) - 1

