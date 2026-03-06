# Classical GAN Baseline
# Same depth as quantum model for fair comparison

import torch
import torch.nn as nn


class ClassicalGenerator(nn.Module):
    """Classical generator with same capacity as quantum generator."""
    def __init__(self, n_features=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_features),
            nn.Tanh()   # output in [-1, 1] like quantum circuit
        )

    def forward(self, x):
        return self.net(x)


class ClassicalDiscriminator(nn.Module):
    """Classical discriminator with same capacity as quantum discriminator."""
    def __init__(self, n_features=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_features)
        )

    def forward(self, x):
        return self.net(x)