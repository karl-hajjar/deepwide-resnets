import unittest
import torch
from numpy import sqrt

from layers.muP_residual import MuPResidual
from utils.tools import set_random_seeds
WIDTH = 1024
D_MODEL = 1024
ALPHA = 1/100
ACTIVATION = 'relu'
N_SAMPLES = 500

R_TOL = 1.0e-5
A_TOL = 1.0e-5
SEED = 42


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seeds(SEED)
        self.layer = MuPResidual(d=D_MODEL, width=WIDTH, activation=ACTIVATION, bias=False, alpha=ALPHA)

    def test_gradient_norms(self):
        h = torch.randn(size=(N_SAMPLES, D_MODEL), requires_grad=True) / sqrt(D_MODEL)
        h.retain_grad()

        u = self.layer.first_layer(h)
        u.retain_grad()

        x = self.layer.activation(u)
        x.retain_grad()

        z = self.layer.second_layer(x)
        z.retain_grad()

        y_hat = h + ALPHA * z
        y_hat.retain_grad()
        loss = 0.5 * torch.sum(torch.sum(y_hat ** 2, dim=1))
        loss.backward()  # gradient computation

        with torch.no_grad():
            y_norm_squared = torch.sum(y_hat**2, dim=1) / D_MODEL
            torch.testing.assert_close(y_hat, self.layer(h), rtol=R_TOL, atol=A_TOL)
            torch.testing.assert_close(y_hat.grad, y_hat, rtol=R_TOL, atol=A_TOL)

            torch.testing.assert_close(z.grad / ALPHA, y_hat.grad, rtol=R_TOL, atol=A_TOL)
            z_grad_norm = torch.sum((z.grad / ALPHA) ** 2, dim=1) / D_MODEL
            x_grad_norm = torch.sum((x.grad / ALPHA) ** 2, dim=1) / WIDTH
            u_grad_norm = torch.sum((u.grad / ALPHA) ** 2, dim=1) / D_MODEL

            # gradient norms
            self.assertTrue((
                (z_grad_norm / y_norm_squared) > (D_MODEL ** (-1/2))
                             ).all())

            self.assertTrue((
                (x_grad_norm / y_norm_squared) > (D_MODEL ** (-1/2))
                             ).all())
            torch.testing.assert_close(x_grad_norm, 2 * z_grad_norm, rtol=1e-3, atol=1e-3)

            self.assertTrue((
                (u_grad_norm / y_norm_squared) > (D_MODEL ** (-1/2))
                             ).all())
            torch.testing.assert_close(u_grad_norm, 0.5 * x_grad_norm, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(u_grad_norm, z_grad_norm, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
