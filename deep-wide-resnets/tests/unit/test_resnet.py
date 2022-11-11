from unittest import TestCase
import torch

from networks.resnet import ResNet

INPUT_DIM = 20
WIDTH = 256
D_MODEL = 128
N_RES = 5
ALPHA = 1.0
ACTIVATION = 'relu'
N_SAMPLES = 500
SCALE = 1.0

R_TOL = 1.0e-5
A_TOL = 1.0e-5
SEED = 42


class TestResNet(TestCase):
    def setUp(self) -> None:
        self.net_bias = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True,
                               alpha=ALPHA, n_res=N_RES, scale=SCALE)
        self.net_no_bias = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False,
                                  alpha=ALPHA, n_res=N_RES, scale=SCALE)
        
    def test_init_method(self):
        self.assertTrue(self.net_bias.input_dim == INPUT_DIM)
        self.assertTrue(self.net_bias.width == WIDTH)
        self.assertTrue(self.net_bias.n_res == N_RES)
        self.assertTrue(self.net_bias.d_model == D_MODEL)
        self.assertTrue(self.net_bias.bias)
        self.assertFalse(self.net_no_bias.bias)
        self.assertTrue(self.net_bias.activation_name == ACTIVATION)
        self.assertTrue(self.net_bias.alpha == ALPHA)
        self.assertTrue(self.net_bias.scale == SCALE)

        self.test_initialization()

    def test_initialization(self):
        self.assertTrue(self.net_no_bias.input_layer.bias is None)
        self.assertTrue(self.net_no_bias.output_layer.bias is None)

        self.assertEqual(self.net_bias.input_layer.bias.shape, (D_MODEL,))
        self.assertEqual(self.net_bias.output_layer.bias.shape, (1,))

        self.assertEqual(self.net_no_bias.input_layer.weight.shape, (D_MODEL, INPUT_DIM))
        self.assertEqual(self.net_no_bias.output_layer.weight.shape, (1, D_MODEL))

    def test_initialization_kwargs(self):
        kind = 'he'
        net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True,
                     alpha=ALPHA, n_res=N_RES, scale=SCALE, kind=kind, mode='fan_in')
        net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True,
                     alpha=ALPHA, n_res=N_RES, scale=SCALE, kind=kind, mode='fan_out')

        kind = 'sphere'
        net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True,
                     alpha=ALPHA, n_res=N_RES, scale=SCALE, kind=kind)
        net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=True,
                     alpha=ALPHA, n_res=N_RES, scale=SCALE, kind=kind)

        self.assertTrue(True)

    def test_forward(self):
        x = torch.randn(size=(N_SAMPLES, INPUT_DIM))
        y_hat_bias = self.net_bias(x)
        y_hat_no_bias = self.net_no_bias(x)
        self.assertSequenceEqual(y_hat_bias.shape, (N_SAMPLES, 1))
        self.assertSequenceEqual(y_hat_no_bias.shape, (N_SAMPLES, 1))

    def test_scale(self):
        scales = [0., 0.5, 1.0, 2.0, 10.0]
        x = torch.randn(size=(N_SAMPLES, INPUT_DIM))
        y_hat = self.net_no_bias(x)
        for scale in scales:
            net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False,
                         alpha=ALPHA, n_res=N_RES, scale=scale)
            with torch.no_grad():
                net.input_layer.weight.data.copy_(self.net_no_bias.input_layer.weight.detach().data)
                net.output_layer.weight.data.copy_(self.net_no_bias.output_layer.weight.detach().data)

                for l in range(net.n_res - 1):
                    net.residual_layers[l].first_layer.weight.data.copy_(
                        self.net_no_bias.residual_layers[l].first_layer.weight.detach().data)
                    net.residual_layers[l].second_layer.weight.data.copy_(
                        self.net_no_bias.residual_layers[l].second_layer.weight.detach().data)

            y_hat_scale = net(x)
            torch.testing.assert_close(scale * y_hat, y_hat_scale, rtol=R_TOL, atol=A_TOL)

    def test_set_inner_layer_dimension(self):
        net = ResNet(input_dim=INPUT_DIM, d_model=D_MODEL, activation=ACTIVATION, bias=False,
                     alpha=ALPHA, n_res=N_RES)
        self.assertTrue(net.width == D_MODEL)
        self.assertTrue(net.d_model == D_MODEL)

        net = ResNet(input_dim=INPUT_DIM, width=WIDTH, activation=ACTIVATION, bias=False,
                     alpha=ALPHA, n_res=N_RES)
        self.assertTrue(net.d_model == WIDTH)
        self.assertTrue(net.width == WIDTH)

        with self.assertRaises(ValueError):
            ResNet(input_dim=INPUT_DIM, activation=ACTIVATION, bias=False, alpha=ALPHA, n_res=N_RES)
