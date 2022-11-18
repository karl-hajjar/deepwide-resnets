import unittest
import jax
import haiku as hk

from networks.jax.resnet import ResNet


INPUT_DIM = 64
WIDTH = 256
N_RES = 5
BIAS = False
ALPHA = 1.0
ACTIVATION = 'relu'

alpha = 1 / N_RES
RNG_KEY = jax.random.PRNGKey(42)


class TestJaxResNet(unittest.TestCase):
    def setUp(self) -> None:
        self.key_0, self.key_1 = jax.random.split(key=RNG_KEY, num=2)
        self.x = jax.random.normal(key=self.key_0, shape=(WIDTH,))
        self.forward = hk.transform(self._forward)

    @staticmethod
    def _forward(x):
        kind = 'glorot'
        net = ResNet(input_dim=INPUT_DIM, n_res=N_RES, width=WIDTH, activation=ACTIVATION, bias=BIAS,
                     alpha=alpha, kind=kind)
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
