import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
np.random.seed(1)


class Policy(nn.Module):

    def __init__(self, n_in, n_mid, n_out, lr=0.001):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_out)
        self.lr = lr
        self.eps = 1e-5
        self.optimizer = optim.SGD(
            self.parameters(), lr=self.lr)
        self.loss_fn = lambda action, q: \
            - torch.log(action + self.eps) * q

    def forward(self, x):
        x = torch.relu(self.l1(x))
        action = torch.softmax(self.l2(x), dim=1)
        return action
    
    def init(self):
        nn.init.normal_(self.l1.weight, mean=0, std=0.1)
        nn.init.zeros_(self.l1.bias)
        nn.init.normal_(self.l2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.l2.bias)
    
    def fit(self, state, q):
        state = torch.tensor(state, dtype=torch.float32)
        q = torch.tensor(q, dtype=torch.float32)
        action = self(state)
        loss = self.loss_fn(action, q)
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
        return loss.data


class Reasoner(nn.Module):

    def __init__(self, n_in, n_mid, n_out, lr = 0.001):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l1_2 = nn.Linear(n_mid, n_mid)
        self.l2 = nn.Linear(n_mid, n_out)
        self.lr = lr
        self.optimizer = optim.SGD(
            self.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l1_2(x))
        x = torch.sigmoid(self.l2(x))
        return x
    
    def init(self):
        nn.init.normal_(self.l1.weight, mean=0, std=0.1)
        nn.init.zeros_(self.l1.bias)
        nn.init.normal_(self.l1_2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.l1_2.bias)
        nn.init.normal_(self.l2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.l2.bias)


class Detector:

    def __init__(self):
        self.eps = 1e-8
        self.loss_fn = nn.MSELoss(reduction='mean')

    def __call__(self, p, q):
        log_prob_ratio = torch.log(
            (p + self.eps) / (q + self.eps))
        return log_prob_ratio

