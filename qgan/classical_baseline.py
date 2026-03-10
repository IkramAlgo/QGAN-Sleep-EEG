# qgan/classical_baseline.py
# Classical GAN for fair comparison with QGAN
# Same depth and capacity as the quantum model

import torch.nn as nn
from qgan.config import N_LAYERS

HIDDEN = 4 * N_LAYERS * 2   # 16 units, same capacity as quantum model


class ClassicalGenerator(nn.Module):
    # takes random noise, outputs fake sleep data in range [-1, 1]

    def __init__(self, n_features=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),     nn.Tanh(),
            nn.Linear(HIDDEN, n_features), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ClassicalDiscriminator(nn.Module):
    # takes real or fake data, outputs judgment scores

    def __init__(self, n_features=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, HIDDEN),     nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, n_features)
        )

    def forward(self, x):
        return self.net(x)