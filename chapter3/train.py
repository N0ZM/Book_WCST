import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys
par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(par_dir)

import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from models import Policy, Reasoner, Detector
from utils import get_dummy_variables, get_q_by_t


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


image_dir = os.path.join(par_dir, 'imgs/')
image_list = os.listdir(image_dir)
samples = get_dummy_variables(image_list)

# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2, lr=0.01)
reasoner = Reasoner(2, 256, 15, lr=0.001)
detector = Detector()

policy.to(device)
reasoner.to(device)

n_epoch = 500
t_idx = 4
loss_list = []

# Train reasoner by detector
for epoch in range(n_epoch):
    state = torch.tensor(
        samples, dtype=torch.float32).to(device)
    q = get_q_by_t(state, t_idx)
    q = torch.tensor(q, dtype=torch.float32).to(device)
    pred_state = reasoner(q)
    diff = detector(pred_state, state)
    t = torch.zeros_like(diff)
    loss = detector.loss_fn(diff, t)
    reasoner.optimizer.zero_grad()
    loss.backward()
    reasoner.optimizer.step()
    loss_list.append(loss.data)

    print(f'epoch: {epoch}, total loss: {loss.data}')

plt.plot(loss_list)
plt.xlabel('N Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(
    os.path.dirname(__file__), 'result.png'), dpi=300)
plt.show()

