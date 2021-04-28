"""
GPBO demo using Pyro, reference: http://pyro.ai/examples/bo.html
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp

pyro.set_rng_seed(2021)
print(pyro.__version__)

from GPBO import GPBO


def f(x):
    return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)


x = torch.linspace(0, 1, steps=100)
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), f(x).numpy())
plt.show()

# initialize the model with four input points: 0.0, 0.33, 0.66, 1.0
X = torch.tensor([0.0, 0.33, 0.66, 1.0])
y = f(X)
gpmodel = gp.models.GPRegression(
    X, y, gp.kernels.Matern52(input_dim=1), noise=torch.tensor(0.1), jitter=1.0e-4
)

Z_dist = pyro.distributions.Normal(0, 1)


gpbo = GPBO(X, y, f)
gpbo.train(iter=20)
