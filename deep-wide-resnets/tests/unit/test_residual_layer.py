from unittest import TestCase
import torch

from layers.residual import Residual


WIDTH = 256
D_MODEL = 128
ALPHA = 1.0
ACTIVATION = 'relu'
N_SAMPLES = 500

R_TOL = 1.0e-5
A_TOL = 1.0e-5


# noinspection PyComparisonWithNone
class TestResidualLayer(TestCase):
    def setUp(self) -> None:
        self.layer_bias = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True, alpha=ALPHA)
        self.layer_no_bias = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False, alpha=ALPHA)

    def test_init_method(self):
        self.assertTrue(self.layer_bias.width == WIDTH)
        self.assertTrue(self.layer_bias.d == D_MODEL)
        self.assertTrue(self.layer_bias.bias)
        self.assertFalse(self.layer_no_bias.bias)
        self.assertTrue(self.layer_bias.activation_name == ACTIVATION)
        self.assertTrue(self.layer_bias.alpha == ALPHA)

        self.test_initialization()

    def test_initialization(self):
        self.assertTrue(self.layer_no_bias.first_layer.bias is None)
        self.assertTrue(self.layer_no_bias.second_layer.bias is None)

        self.assertEqual(self.layer_bias.first_layer.bias.shape, (WIDTH,))
        self.assertEqual(self.layer_bias.second_layer.bias.shape, (D_MODEL,))

        self.assertEqual(self.layer_no_bias.first_layer.weight.shape, (WIDTH, D_MODEL))
        self.assertEqual(self.layer_no_bias.second_layer.weight.shape, (D_MODEL, WIDTH))

        self.test_initialization_kwargs()

    def test_initialization_kwargs(self):
        kind = 'he'
        layer = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True, alpha=ALPHA, kind=kind,
                         mode='fan_in')
        layer = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True, alpha=ALPHA, kind=kind,
                         mode='fan_out')

        kind = 'sphere'
        layer = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True, alpha=ALPHA, kind=kind)
        layer = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False, alpha=ALPHA, kind=kind)

        self.assertTrue(True)

    def test_forward(self):
        x = torch.randn(size=(N_SAMPLES, D_MODEL))
        y_hat_bias = self.layer_bias(x)
        y_hat_no_bias = self.layer_no_bias(x)
        self.assertSequenceEqual(y_hat_bias.shape, (N_SAMPLES, D_MODEL))
        self.assertSequenceEqual(y_hat_no_bias.shape, (N_SAMPLES, D_MODEL))

    def test_alpha(self):
        alphas = [0., 0.5, 10]
        x = torch.randn(size=(N_SAMPLES, D_MODEL))
        y_hat = self.layer_no_bias(x)
        for alpha in alphas:
            layer = Residual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False, alpha=alpha)
            with torch.no_grad():
                layer.first_layer.weight.data.copy_(self.layer_no_bias.first_layer.weight.detach().data)
                layer.second_layer.weight.data.copy_(self.layer_no_bias.second_layer.weight.detach().data)

            y_hat_alpha = layer(x)
            torch.testing.assert_close(alpha*(y_hat - x), y_hat_alpha - x, rtol=R_TOL, atol=A_TOL)
