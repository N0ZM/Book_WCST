import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from copy import deepcopy

from flask import Flask, render_template, session
import numpy as np
import torch

import sys
par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(par_dir)

from models import Policy, Reasoner, Detector
from utils import get_dummy_variable


app = Flask(__name__)
app.secret_key = 'top_secret'
static_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'static'))

# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2)
reasoner = Reasoner(2, 256, 15, lr=0.001)
detector = Detector()

policy.to(device)
reasoner.to(device)

THRESHOULD_NORMAL_POLICY = 0.2
THRESHOULD_ANOMALY_POLICY = 2.0
THRESHOULD_NORMAL_REASONER = 0.2
THRESHOULD_ANOMALY_REASONER = 16.0

@app.route('/', methods=['GET', 'POST'])
def index():
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [
        *deepcopy(shapes), *deepcopy(numbers), *deepcopy(colors)]

    if not 'n' in session:
        session['n'] = 1
    else:
        session['n'] += 1
    if not 'score' in session:
        session['score'] = 0
    if not 'correct' in session:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]
    if session['n'] == 101:
        session['n'] = 1
        session['score'] = 0
    if session['n'] % 20 == 0:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]
    if not 'prev_result_reasoner' in session:
        session['prev_result_reasoner'] = 'Incorrect'
    if not 'prev_result_policy' in session:
        session['prev_result_policy'] = 'Incorrect'

    img_list = []
    dummy_list = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        img_list.append(sample)
        dummy_list.append([get_dummy_variable(sample)])

    q = torch.tensor([[1, -1]], dtype=torch.float32).to(device)
    state = reasoner(q)
    result_idx = torch.argmax(state)
    pred_correct_attr = all_attr[result_idx]

    target = session['correct']
    target_idx = all_attr.index(session['correct'])

    for i, d in enumerate(dummy_list):
        if d[0][result_idx] == 1:
            selected_card_reasoner = d[0]
            result_card_idx = i
        if d[0][target_idx] == 1:
            correct_card_idx = i

    result_reasoner = \
        'Correct' if selected_card_reasoner[target_idx] == 1 \
            else 'Incorrect'
    
    if result_reasoner == 'Correct':
        q_real = [[1, -1]]
    else:
        q_real = [[-1, 1]]
    q_real = torch.tensor(q_real, dtype=torch.float32).to(device)
    
    state = torch.tensor(
        dummy_list[result_card_idx], dtype=torch.float32)
    state = state.to(device)
    pred_state = reasoner(q)
    diff = detector(pred_state, state)
    target_diff = diff[0][result_idx]

    anomaly_value = torch.abs(target_diff).detach().numpy()
    is_normal_reasoner = \
        True if torch.abs(target_diff) < THRESHOULD_NORMAL_REASONER \
            else False
    is_anomaly_reasoner = \
        True if torch.abs(target_diff) > THRESHOULD_ANOMALY_REASONER \
            else False

    ######## calc result of policy ######## 
    res_list = []
    for i, d in enumerate(dummy_list):
        d = torch.tensor(d, dtype=torch.float32).to(device)
        res = policy(d)
        res_list.append(res[0][0].detach().numpy())
    res_idx = np.argmax(res_list)

    unselected_cards = []
    for i, d in enumerate(dummy_list):
        if i == res_idx:
            selected_card = d[0]
            selected_card_img = img_list[i]
        else:
            unselected_cards.append(d)

    result_policy = \
        'Correct' if res_idx == correct_card_idx \
            else 'Incorrect'

    state = torch.tensor(
        dummy_list[res_idx], dtype=torch.float32).to(device)
    res_pred = policy(state)
    if result_policy == 'Correct':
        res_real = torch.tensor(
            [[1., 0.]], dtype=torch.float32).to(device)
    else:
        res_real = torch.tensor(
            [[0., 1.]], dtype=torch.float32).to(device)
    diff = detector(res_pred, res_real)
    diff = diff[0][0]

    anomaly_value = torch.abs(diff).detach().numpy()
    is_normal_policy = \
        True if torch.abs(diff) < THRESHOULD_NORMAL_POLICY \
            else False
    is_anomaly_policy = \
        True if torch.abs(diff) > THRESHOULD_ANOMALY_POLICY \
            else False
    ######## end calc result of policy ######## 
    
    if result_policy == 'Correct':
        session['score'] += 1

    n_epoch = 100

    if result_reasoner == 'Incorrect' and \
        session['prev_result_reasoner'] == 'Incorrect' \
            and is_anomaly_reasoner:
        reasoner.init()
    if result_policy == 'Incorrect' and \
        session['prev_result_policy'] == 'Incorrect' \
            and is_anomaly_policy:
        policy.init()
    if not (result_reasoner == 'Correct' and is_normal_reasoner):
        for _ in range(n_epoch):
            # Train reasoner by detector
            for _ in range(len(unselected_cards)):
                if result_policy == 'Correct':
                    q = torch.tensor(
                        [[1, -1]], dtype=torch.float32).to(device)
                else:
                    q = torch.tensor(
                        [[-1, 1]], dtype=torch.float32).to(device)
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
            for card in unselected_cards:
                if result_policy == 'Correct':
                    q = torch.tensor(
                        [[-1, 1]], dtype=torch.float32).to(device)
                else:
                    q = torch.tensor(
                        [[1, -1]], dtype=torch.float32).to(device)
                state = torch.tensor(
                    card, dtype=torch.float32).to(device)
                pred_state = reasoner(q)
                diff = detector(pred_state, state)
                t = torch.zeros_like(diff)
                loss = detector.loss_fn(diff, t)
                reasoner.optimizer.zero_grad()
                loss.backward()
                reasoner.optimizer.step()
    
        if not (result_policy == 'Correct' and is_normal_policy):
            for i, d in enumerate(dummy_list):
                if i == res_idx:
                    for _ in range(len(dummy_list) - 1):
                        if result_policy == 'Correct':
                            q = torch.tensor(
                                [[1, -1]], dtype=torch.float32)
                            q = q.to(device)
                        else:
                            q = torch.tensor(
                                [[-1, 1]], dtype=torch.float32)
                            q = q.to(device)
                        state = torch.tensor(
                            d, dtype=torch.float32).to(device)
                        pred_action = policy(state)
                        loss = policy.loss_fn(pred_action, q)
                        policy.optimizer.zero_grad()
                        loss.sum().backward()
                        policy.optimizer.step()
                else:
                    for card in unselected_cards:
                        if result_policy == 'Correct':
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
                        pred_action = policy(state)
                        loss = policy.loss_fn(pred_action, q)
                        policy.optimizer.zero_grad()
                        loss.sum().backward()
                        policy.optimizer.step()
    else:
        if not (result_policy == 'Correct' and is_normal_policy):
            q = torch.tensor(
                [[1, -1]], dtype=torch.float32).to(device)
            state = reasoner(q)
            result_idx = torch.argmax(state)
            tmp_unselected_cards = []
            for i, d in enumerate(dummy_list):
                if d[0][result_idx] == 1:
                    tmp_selected_card = d
                else:
                    tmp_unselected_cards.append(d)
            for _ in range(n_epoch):
                # Train policy by reasoner
                for i in range(len(tmp_unselected_cards)):
                    q = torch.tensor(
                        [[1, -1]], dtype=torch.float32).to(device)
                    state = torch.tensor(
                        tmp_selected_card, dtype=torch.float32)
                    state = state.to(device)
                    pred_action = policy(state)
                    loss = policy.loss_fn(pred_action, q)
                    policy.optimizer.zero_grad()
                    loss.sum().backward()
                    policy.optimizer.step()
                for card in tmp_unselected_cards:
                    q = torch.tensor(
                        [[-1, 1]], dtype=torch.float32).to(device)
                    state = torch.tensor(
                        card, dtype=torch.float32).to(device)
                    pred_action = policy(state)
                    loss = policy.loss_fn(pred_action, q)
                    policy.optimizer.zero_grad()
                    loss.sum().backward()
                    policy.optimizer.step()

    q = torch.tensor([[1, -1]], dtype=torch.float32).to(device)
    pred_correct_feature = reasoner(q)
    pred_correct_feature = pred_correct_feature.detach().numpy()
    dict_pred_correct_feature = {}
    for i, attr in enumerate(all_attr):
        dict_pred_correct_feature[attr] = pred_correct_feature[0][i]

    session['prev_result_reasoner'] = result_reasoner
    session['prev_result_policy'] = result_policy

    return render_template('index.html',
                           img_list=img_list,
                           correct_card_attribute=target, 
                           result=result_policy,
                           n=session['n'],
                           score=session['score'],
                           correct=session['correct'],
                           selected_card_img=selected_card_img,
                           anomaly_value=anomaly_value,
                           is_anomaly=is_anomaly_policy,
                           dict_pred_correct_feature=\
                           dict_pred_correct_feature,
                           pred_correct_attr=pred_correct_attr)

if __name__ == '__main__':
    app.run()
