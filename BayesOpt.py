import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import numpy as np

Z_dist = dist.Normal(0, 1)


class GPBO(object):
    def __init__(
        self,
        X,
        y,
        f,
        bound=(0, 1),
        kernel=gp.kernels.Matern52,
        acquistion_fn="EI",
        num_candidates=5,
        grid_resolution=10,
    ):
        # X = toTorch(X)
        # y = toTorch(y)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)
        self.input_dim = X.shape[1]
        self.next_x_fn = self.next_x
        if acquistion_fn == "EI":
            self.acquistion_fn = self.expected_improvement
        elif acquistion_fn == "LCB":
            self.acquistion_fn = self.lower_confidence_bound
        elif acquistion_fn == "PI":
            self.acquistion_fn = self.probability_improvement
        elif acquistion_fn == "Thompson":
            self.next_x_fn = self.next_x_thompson
        elif acquistion_fn == "random":
            self.next_x_fn = self.next_x_random
        self.f = f
        self.lower_bound, self.upper_bound = bound
        self.num_candidates = num_candidates
        self.gpmodel = gp.models.GPRegression(
            X,
            y,
            kernel(input_dim=self.input_dim),
            noise=torch.tensor(0.1),
            jitter=1.0e-2,
        )
        min_idx = torch.argmin(y.detach())
        self.y_min = y[min_idx].detach().item()
        self.x_min = X[min_idx].detach().numpy()
        self.x_hist = [self.x_min]
        self.y_hist = [self.y_min]

        zz = [
            np.linspace(
                self.lower_bound, self.upper_bound, grid_resolution, endpoint=False
            )
        ] * self.input_dim
        # equivalent to the Cartesian product(*zz)
        self.grid = torch.Tensor(
            np.array(np.meshgrid(*zz)).T.reshape(-1, self.input_dim)
        )
        self.grid_delta = (self.upper_bound - self.lower_bound) / grid_resolution

    def update_posterior(self, x_new, log):
        y_new = self.f(x_new)  # evaluate f at new point.
        # if len(torch.tensor(y_new).shape) == 2:
        y_new = y_new.squeeze(1)
        if log:
            print(f"x={x_new[0].numpy()}, f={y_new.item():.4f}")
        if y_new.item() < self.y_min:
            self.x_min, self.y_min = x_new[0].numpy(), y_new.item()
        self.x_hist.append(self.x_min)
        self.y_hist.append(self.y_min)
        X = torch.cat([self.gpmodel.X, x_new])  # incorporate new evaluation
        y = torch.cat([self.gpmodel.y, y_new])
        self.gpmodel.set_data(X, y)
        # optimize the GP hyperparameters using Adam with lr=0.001
        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
        gp.util.train(self.gpmodel, optimizer)

    def lower_confidence_bound(self, x, kappa=2):
        mu, variance = self.gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def probability_improvement(self, x, xi=0.01):
        mu, variance = self.gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        Z = (self.y_min - xi - mu) / sigma
        return -Z_dist.cdf(Z)

    def expected_improvement(self, x, xi=0.01):
        mu, variance = self.gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        Z = (self.y_min - xi - mu) / sigma
        return -(
            (self.y_min - xi - mu) * Z_dist.cdf(Z) + sigma * Z_dist.log_prob(Z).exp()
        )

    def find_a_candidate(self, x_init):
        # transform x to an unconstrained domain
        constraint = constraints.interval(self.lower_bound, self.upper_bound)
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn="strong_wolfe")

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = self.acquistion_fn(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)

        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(constraint)(unconstrained_x)
        return x.detach()

    def next_x(self):
        candidates = []
        values = []

        x_init = self.gpmodel.X[-1:]
        for i in range(self.num_candidates):
            x = self.find_a_candidate(x_init)
            y = self.acquistion_fn(x)
            candidates.append(x)
            values.append(y)
            x_init = x.new_empty((1, self.input_dim)).uniform_(
                self.lower_bound, self.upper_bound
            )
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]

    def next_x_thompson(self):
        # print(len(self.grid))
        pts = self.grid + self.grid_delta * torch.rand_like(self.grid)
        mean, cov = self.gpmodel(pts, full_cov=True, noiseless=False)
        cov += (cov.diag().max() * self.gpmodel.noise) * torch.eye(len(self.grid))
        sample = dist.MultivariateNormal(mean, cov).sample()
        min_idx = torch.argmin(sample)
        return pts[[min_idx.item()]]

    def next_x_random(self):
        return dist.Uniform(self.lower_bound, self.upper_bound).sample(
            (1, self.input_dim)
        )

    def plot(self, gs, xmin, xlabel=None, with_title=True):
        xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
        Xnew = torch.linspace(self.lower_bound, self.upper_bound, steps=1000)
        Xnew = torch.unsqueeze(Xnew, 1)
        ax1 = plt.subplot(gs[0])
        ax1.plot(
            self.gpmodel.X.numpy(), self.gpmodel.y.numpy(), "kx"
        )  # plot all observed data
        with torch.no_grad():
            loc, var = self.gpmodel(Xnew, full_cov=False, noiseless=False)
            sd = var.sqrt()
            ax1.plot(
                Xnew.numpy().squeeze(), loc.numpy(), "r", lw=2
            )  # plot predictive mean
            ax1.fill_between(
                Xnew.numpy().squeeze(),
                loc.numpy() - 2 * sd.numpy(),
                loc.numpy() + 2 * sd.numpy(),
                color="C0",
                alpha=0.3,
            )  # plot uncertainty intervals
        ax1.set_xlim(self.lower_bound, self.upper_bound)
        ax1.set_title("Find {}".format(xlabel))
        if with_title:
            ax1.set_ylabel("Gaussian Process Regression")

        ax2 = plt.subplot(gs[1])
        with torch.no_grad():
            # plot the acquisition function
            ax2.plot(Xnew.numpy().squeeze(), self.acquistion_fn(Xnew).numpy())
            # plot the new candidate point
            ax2.plot(
                xmin.numpy(),
                self.acquistion_fn(xmin).numpy(),
                "^",
                markersize=10,
                label="{} = {:.5f}".format(xlabel, xmin.item()),
            )
        ax2.set_xlim(self.lower_bound, self.upper_bound)
        if with_title:
            ax2.set_ylabel("Acquisition Function")
        ax2.legend(loc=1)

    def train(self, iter=8, log=False):
        plt.figure(figsize=(12, 30))
        if self.input_dim == 1:
            outer_gs = gridspec.GridSpec(5, 2)
        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
        gp.util.train(self.gpmodel, optimizer)
        for i in range(iter):
            xmin = self.next_x_fn()
            if self.input_dim == 1 and i < 10:
                gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i])
                self.plot(gs, xmin, xlabel=i + 1, with_title=(i % 2 == 0))
            if log and i % 10 == 0:
                print(f"iter {i}: ", end="")
                self.update_posterior(xmin, log)
        print(f"Final result: {self.x_min}, {self.y_min:.4f}")
        plt.figure()
        plt.plot(self.y_hist)
        plt.show()
