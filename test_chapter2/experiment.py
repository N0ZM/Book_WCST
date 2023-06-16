import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from copy import deepcopy

import numpy as np
import torch
import pandas as pd
from scipy import stats as st
from matplotlib import pyplot as plt

import sys
par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(par_dir)

from models import Policy, Detector
from utils import get_dummy_variable


# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2)
detector = Detector()

policy.to(device)

THRESHOULD_NORMAL = 0.2
THRESHOULD_ANOMALY = 8.0

def exec_one_trial(policy, correct_attr, on_detector):
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [
        *deepcopy(shapes), *deepcopy(numbers), 
        *deepcopy(colors)]

    dummy_list = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        dummy_list.append([get_dummy_variable(sample)])

    target_idx = all_attr.index(correct_attr)
    
    res_list = []
    for i, d in enumerate(dummy_list):
        d = torch.tensor(d, dtype=torch.float32).to(device)
        res = policy(d)
        res_list.append(res[0][0].detach().numpy())
        if d[0][target_idx] == 1:
            target_card_idx = i

    res_idx = np.argmax(res_list)
    result = 'Correct' if res_idx == target_card_idx \
        else 'Incorrect'
    
    state = torch.tensor(
        dummy_list[res_idx], dtype=torch.float32).to(device)
    res_pred = policy(state)

    if result == 'Correct':
        res_real = torch.tensor(
            [[1., 0.]], dtype=torch.float32).to(device)
        diff = detector(res_pred, res_real)
        diff = diff[0][0]
    else:
        res_real = torch.tensor(
            [[0., 1.]], dtype=torch.float32).to(device)
        diff = detector(res_pred, res_real)
        diff = diff[0][1]

    is_normal = True if torch.abs(diff) < THRESHOULD_NORMAL \
        else False
    is_anomaly = True if torch.abs(diff) > THRESHOULD_ANOMALY \
        else False

    n_epoch = 500
    loss_list_policy = []
    if on_detector:
        # Init policy if anomaly
        if result == 'Incorrect' and is_anomaly:
            policy.init()
            print('reset')
    # Train policy
    if (not on_detector) or \
        not (result == 'Correct' and is_normal):
        for _ in range(n_epoch):
            for i, d in enumerate(dummy_list):
                if i == res_idx:
                    if result == 'Correct':
                        for _ in range(len(dummy_list)):
                            q = torch.tensor(
                                [[1, -1]], 
                                dtype=torch.float32)
                            q = q.to(device)
                            state = torch.tensor(
                                d, dtype=torch.float32)
                            state = state.to(device)
                            pred_action = policy(state)
                            loss = policy.loss_fn(
                                pred_action, q)
                            policy.optimizer.zero_grad()
                            loss.sum().backward()
                            policy.optimizer.step()
                            loss_list_policy.append(
                        loss.sum().detach().numpy())
                    else:
                        q = torch.tensor(
                            [[-1, 1]], 
                            dtype=torch.float32)
                        q = q.to(device)
                        state = torch.tensor(
                            d, dtype=torch.float32)
                        state = state.to(device)
                        pred_action = policy(state)
                        loss = policy.loss_fn(
                            pred_action, q)
                        policy.optimizer.zero_grad()
                        loss.sum().backward()
                        policy.optimizer.step()
                        loss_list_policy.append(
                            loss.sum().detach().numpy())
    return result

def exec_WCST_policy(n_test, n_trial, file_name, 
                     on_detector=False):
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [*deepcopy(shapes), *deepcopy(numbers), 
                *deepcopy(colors)]
    results = []
    for i in range(n_test):
        policy = Policy(15, 128, 2, lr=0.001)
        res_test = 0
        print(f'---- i: {i} ----')
        for j in range(n_trial):
            if j % 20 == 0:
                correct_attr = all_attr[
                    np.random.choice(len(all_attr))]
            res = exec_one_trial(
                policy, correct_attr, on_detector)
            if res == 'Correct':
                res_test += 1
            print(f'j: {j}, {res}')
        results.append(res_test)
    df = pd.DataFrame(results)
    df.to_csv(f'{file_name}.csv', index=False, header=None)

def exec_WCST_by_random_agent(n_test, n_trial, file_name):
    results = []
    for i in range(n_test):
        res_test = 0
        print(f'---- i: {i} ----')
        for j in range(n_trial):
            if j % 20 == 0:
                correct_card_idx = np.random.choice(range(5))
            rdm = np.random.choice(range(1, 5))
            res = 'Correct' if rdm == correct_card_idx \
                else 'Incorrect'
            if res == 'Correct':
                res_test += 1
            print(f'j: {j}, {res}')
        results.append(res_test)
    df = pd.DataFrame(results)
    df.to_csv(f'{file_name}.csv', index=False, header=None)

result_path = os.path.join(os.path.dirname(__file__),'result')
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)

exec_WCST_policy(50, 100, os.path.join(
    result_path,'policy_on_detector'), on_detector=True)
exec_WCST_policy(50, 100, os.path.join(
    result_path,'policy_no_detector'), on_detector=False)
exec_WCST_by_random_agent(50, 100, os.path.join(
    result_path,'random'))

def plot_dist(*result_files):
    df = pd.DataFrame()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.linspace(0, 100, 1001)
    result_dir = os.path.join(
            os.path.dirname(__file__),'result')
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    for i, result_file in enumerate(result_files):
        result_file_path = os.path.join(
            result_dir, result_file)
        tmp_df = pd.read_csv(result_file_path, header=None)
        data = np.array(tmp_df[0])
        mu = data.mean()
        sigma = data.std()
        norm = st.norm(mu, sigma)
        ls = line_styles[i % len(line_styles)]
        ax.plot(xs, norm.pdf(xs), label=result_file,
                linestyle=ls)
        df = pd.concat([df, tmp_df], axis=1)
    ax.legend()
    result_dist_path = os.path.join(
        os.path.dirname(__file__), 'result_dist.png')
    plt.savefig(result_dist_path, dpi=300)
    plt.show()

result_files = os.listdir(result_path)
plot_dist(*result_files)
