import unittest
import torch
from numpy import sqrt
from copy import deepcopy

from networks.muP_resnet import MuPResNet
from utils.tools import set_random_seeds

INPUT_DIM = 30
WIDTH = 2500
D_MODEL = 2500
N_RES = 30
ACTIVATION = 'relu'
N_SAMPLES = 1
BASE_LR = 1.0
N_TRIALS = 3

R_TOL = 1.0e-5
A_TOL = 1.0e-5
SEED = 42


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seeds(SEED)
        self.alpha = 1/N_RES
        self.net = MuPResNet(input_dim=INPUT_DIM, d_model=D_MODEL, width=WIDTH, n_res=N_RES, activation=ACTIVATION,
                             bias=False, alpha=self.alpha)

    def test_update_scales(self):
        with torch.no_grad():
            net_0 = deepcopy(self.net)
            torch.testing.assert_close(net_0.input_layer.weight.data, self.net.input_layer.weight.data,
                                       rtol=R_TOL, atol=A_TOL)
            torch.testing.assert_close(net_0.output_layer.weight.data, self.net.output_layer.weight.data,
                                       rtol=R_TOL, atol=A_TOL)
            torch.testing.assert_close(net_0.residual_layers[N_RES//2].first_layer.weight.data,
                                       self.net.residual_layers[N_RES//2].first_layer.weight.data,
                                       rtol=R_TOL, atol=A_TOL)
            torch.testing.assert_close(net_0.residual_layers[N_RES//2].second_layer.weight.data,
                                       self.net.residual_layers[N_RES//2].second_layer.weight.data,
                                       rtol=R_TOL, atol=A_TOL)

        x = torch.randn(size=(N_SAMPLES, INPUT_DIM), requires_grad=False)
        print('First test:')
        with torch.no_grad():
            torch.testing.assert_close(net_0(x), self.net(x), rtol=R_TOL, atol=A_TOL)
        y = torch.rand(size=(N_SAMPLES, 1), requires_grad=False)
        opt = torch.optim.SGD(params=self.net.parameters(), lr=BASE_LR)
        opt.zero_grad()

        hs = dict()
        xs = dict()

        h = sqrt(self.net.d_model) * self.net.input_layer(x)
        hs[0] = deepcopy(h.detach().data)
        for l in range(N_RES):
            layer = self.net.residual_layers[l]
            x_ = layer.activation(layer.first_layer(h))
            xs[l+1] = deepcopy(x_.detach().data)
            h = h + layer.alpha * layer.second_layer(x_)
            hs[l+1] = deepcopy(h.detach().data)

        y_hat = self.net.output_layer(h) / sqrt(self.net.d_model)
        print('Second test:')
        with torch.no_grad():
            torch.testing.assert_close(y_hat, net_0(x), rtol=R_TOL, atol=A_TOL)
        loss = 0.5 * torch.mean((y_hat - y)**2)
        loss.backward()  # gradient computation

        opt.step()  # update params: one step of GD

        with torch.no_grad():
            loss_scale = ((y_hat - y) ** 2).item()
            for _ in range(N_TRIALS):
                x_1 = torch.randn(size=(N_SAMPLES, INPUT_DIM), requires_grad=False)
                inner_prod_scale = (torch.sum(x * x_1) ** 2).item()
                scale = loss_scale * inner_prod_scale

                h0_0 = sqrt(self.net.d_model) * net_0.input_layer(x_1)
                h0_1 = sqrt(self.net.d_model) * self.net.input_layer(x_1)
                delta_h0_norm = torch.sum((h0_1 - h0_0)**2, dim=1) / D_MODEL
                print('delta_h0_norm.item() / scale:', delta_h0_norm.item() / scale)
                self.assertTrue(
                    (delta_h0_norm.item() / scale) > (D_MODEL ** (-1/4))
                )
                print('')

                for l in range(N_RES):
                    print('---- LAYER {:,} -----'.format(l+1))
                    u1_0 = net_0.residual_layers[l].first_layer(h0_1)
                    u1_1 = self.net.residual_layers[l].first_layer(h0_1)

                    delta_u1_norm = torch.sum((u1_1 - u1_0)**2, dim=1) / WIDTH
                    inner_prod_scale = (torch.sum(hs[0] * h0_1 / D_MODEL) ** 2).item()
                    scale = loss_scale * inner_prod_scale * (self.alpha ** 2)
                    print('delta_u1_norm.item() / scale:', delta_u1_norm.item() / scale)
                    self.assertTrue(
                        (delta_u1_norm.item() / scale) > (WIDTH ** (-1/4))
                    )
                    x1_1 = self.net.activation(u1_1)
                    z1_0 = net_0.residual_layers[l].second_layer(x1_1)
                    z1_1 = self.net.residual_layers[l].second_layer(x1_1)

                    delta_z1_norm = torch.sum((z1_1 - z1_0)**2, dim=1) / D_MODEL
                    inner_prod_scale = (torch.sum(xs[l+1] * x1_1 / WIDTH) ** 2).item()
                    scale = loss_scale * inner_prod_scale * (self.alpha ** 2)
                    print('delta_z1_norm.item() / scale:', delta_z1_norm.item() / scale)
                    self.assertTrue(
                        (delta_z1_norm.item() / scale) > (WIDTH ** (-1/4))
                    )

                    h0_1 = h0_1 + self.alpha * z1_1
                print('')


if __name__ == '__main__':
    unittest.main()
