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

from models import Policy, Reasoner, Detector
from utils import get_dummy_variable


# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2)
reasoner = Reasoner(2, 256, 15, lr=0.001)
detector = Detector()

policy.to(device)
reasoner.to(device)

THRESHOULD_NORMAL = 0.2
THRESHOULD_ANOMALY = 16.0

def exec_one_trial(reasoner, correct_attr, prev_result):
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

    q = torch.tensor([[1, -1]], dtype=torch.float32)
    q = q.to(device)
    result = reasoner(q)
    result_idx = torch.argmax(result)
    unselected_cards = []
    for i, d in enumerate(dummy_list):
        if d[0][result_idx] == 1:
            selected_card_idx = i
            selected_card = d[0]
            target_card_idx = i
        else:
            unselected_cards.append(d)

    target_idx = all_attr.index(correct_attr)
    result = 'Correct' if selected_card[target_idx] == 1 \
        else 'Incorrect'

    if result == 'Correct':
        q_real = [[1, -1]]
    else:
        q_real = [[-1, 1]]
    q_real = torch.tensor(
        q_real, dtype=torch.float32).to(device)

    state = torch.tensor(dummy_list[target_card_idx], 
                         dtype=torch.float32).to(device)
    q = torch.tensor([[1, -1]], dtype=torch.float32)
    q = q.to(device)
    pred_state = reasoner(q_real)
    diff = detector(pred_state, state)
    target_diff = diff[0][result_idx]

    is_normal = \
        True if torch.abs(target_diff) < THRESHOULD_NORMAL \
            else False
    is_anomaly = \
        True if torch.abs(target_diff) > THRESHOULD_ANOMALY \
            else False

    n_epoch = 1000
    loss_list_reasoner = []
    if result == 'Incorrect' and prev_result == 'Incorrect' \
        and is_anomaly:
        reasoner.init()
    if not (result == 'Correct' and is_normal):
        for _ in range(n_epoch):
            # Train reasoner by detector
            for _ in range(len(unselected_cards)):
                if result == 'Correct':
                    q = torch.tensor(
                        [[1, -1]], dtype=torch.float32)
                    q = q.to(device)
                else:
                    q = torch.tensor(
                        [[-1, 1]], dtype=torch.float32)
                    q = q.to(device)
                state = torch.tensor(
                    [selected_card], dtype=torch.float32)
                state = state.to(device)
                pred_state = reasoner(q)
                diff = detector(pred_state, state)
                t = torch.zeros_like(diff)
                loss = detector.loss_fn(diff, t)

                reasoner.optimizer.zero_grad()
                loss.backward()
                reasoner.optimizer.step()
                loss_list_reasoner.append(loss.data)

            for card in unselected_cards:
                if result == 'Correct':
                    q = torch.tensor(
                        [[-1, 1]], dtype=torch.float32)
                    q = q.to(device)
                else:
                    q = torch.tensor(
                        [[1, -1]], dtype=torch.float32)
                    q = q.to(device)
                state = torch.tensor(
                    card, dtype=torch.float32)
                state = state.to(device)
                pred_state = reasoner(q)
                diff = detector(pred_state, state)
                t = torch.zeros_like(diff)
                loss = detector.loss_fn(diff, t)

                reasoner.optimizer.zero_grad()
                loss.backward()
                reasoner.optimizer.step()

    q = torch.tensor([[1, -1]], dtype=torch.float32)
    q = q.to(device)
    pred_correct_feature = reasoner(q)
    pred_correct_feature = pred_correct_feature.detach().numpy()
    dict_pred_correct_feature = {}
    for i, attr in enumerate(all_attr):
        dict_pred_correct_feature[attr] = \
            pred_correct_feature[0][i]

    return result

def exec_WCST(n_test, n_trial, file_name):
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [*deepcopy(shapes), *deepcopy(numbers), 
                *deepcopy(colors)]
    results = []
    prev_result = 'Incorrect'
    for i in range(n_test):
        reasoner = Reasoner(2, 256, 15, lr=0.001)
        correct_attr = all_attr[np.random.choice(len(all_attr))]
        res_test = 0
        print(f'---- i: {i} ----')
        for j in range(n_trial):
            if j % 20 == 0:
                correct_attr = all_attr[
                    np.random.choice(len(all_attr))]
            res = exec_one_trial(
                reasoner, correct_attr, prev_result)
            if res == 'Correct':
                res_test += 1
                prev_result = 'Correct'
            else:
                prev_result = 'Incorrect'
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
            rdm = np.random.choice(range(1, 5))
            res = 'Correct' if rdm == 1 else 'Incorrect'
            if res == 'Correct':
                res_test += 1
            print(f'j: {j}, {res}')
        results.append(res_test)
    df = pd.DataFrame(results)
    df.to_csv(f'{file_name}.csv', index=False, header=None)

result_path = os.path.join(os.path.dirname(__file__),'result')
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)

exec_WCST(50, 100, os.path.join(result_path, 'reasoner'))
exec_WCST_by_random_agent(
    50, 100, os.path.join(result_path, 'random'))

def plot_dist(*result_files):
    base_dir = os.path.dirname(__file__)
    df = pd.DataFrame()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.linspace(0, 100, 1001)
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    for i, result in enumerate(result_files):
        file_path = os.path.join(base_dir, result)
        tmp_df = pd.read_csv(file_path, header=None)
        data = np.array(tmp_df[0])
        mu = data.mean()
        sigma = data.std()
        norm = st.norm(mu, sigma)
        ls = line_styles[i % len(line_styles)]
        ax.plot(xs, norm.pdf(xs), label=os.path.basename(result),
                linestyle=ls)
        df = pd.concat([df, tmp_df], axis=1)
    ax.legend()
    result_dist_path = os.path.join(
        os.path.dirname(__file__), 'result_dist.png')
    plt.savefig(result_dist_path, dpi=300)
    plt.show()

par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
result_files = [
    os.path.join(par_dir,
                'test_chapter3/result/random.csv'),
    os.path.join(par_dir,
                 'test_chapter3/result/reasoner.csv'), 
    os.path.join(par_dir, 
                 'test_chapter2/result/policy_no_detector.csv'), 
    os.path.join(par_dir, 
                 'test_chapter2/result/policy_on_detector.csv')] 

plot_dist(*result_files)
