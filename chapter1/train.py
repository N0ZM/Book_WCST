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
import torch.optim as optim

from models import Policy
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

policy = Policy(15, 128, 2)
policy.to(device)

lr_policy = 0.001
optimizer_policy = optim.SGD(
    policy.parameters(), lr=lr_policy)

eps = 1e-8
loss_fn_policy = lambda action, q: \
    - torch.log(action + eps) * q

n_epoch = 1000
t_idx = 4
reward_list = []

for epoch in range(n_epoch):
    state = torch.tensor(
        samples, dtype=torch.float32).to(device)

    action = policy(state)
    q = get_q_by_t(state, t_idx)
    q = torch.tensor(q, dtype=torch.float32).to(device)
    loss_policy = loss_fn_policy(action, q)
    optimizer_policy.zero_grad()
    loss_policy.sum().backward()
    optimizer_policy.step()

    action_idx = torch.argmax(action, dim=1)
    total_q = 0.
    for i, v in enumerate(q):
        total_q += float(v[action_idx[i]].data)
    reward_list.append(total_q)

    print(f'epoch: {epoch}, total reward: {total_q}')

plt.plot(reward_list)
plt.xlabel('N Epoch')
plt.ylabel('Reward')
plt.savefig(os.path.join(
    os.path.dirname(__file__), 'result.png'), dpi=300)
plt.show()
