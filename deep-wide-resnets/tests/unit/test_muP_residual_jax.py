import unittest
import jax
import haiku as hk

from layers.jax.muP_residual import MuPResidual


INPUT_DIM = 64
WIDTH = 256
N_RES = 500
BIAS = False
ALPHA = 1.0
ACTIVATION = 'relu'

alpha = 1 / N_RES
RNG_KEY = jax.random.PRNGKey(42)


class TestJaxMuPResidual(unittest.TestCase):
    def setUp(self) -> None:
        self.key_0, self.key_1 = jax.random.split(key=RNG_KEY, num=2)
        self.x = jax.random.normal(key=self.key_0, shape=(WIDTH,))
        self.forward = hk.transform(self._forward)

    @staticmethod
    def _forward(x):
        net = MuPResidual(d=WIDTH, width=WIDTH, activation=ACTIVATION, bias=BIAS, alpha=alpha)
        y = net(x)
        return y

    def test_init(self):
        self.params = self.forward.init(rng=self.key_1, x=self.x)
        print(self.params)

    def test_inference_after_init(self):
        self.test_init()
        y = self.forward.apply(self.params, None, self.x)
        print(y.shape)


if __name__ == '__main__':
    unittest.main()
