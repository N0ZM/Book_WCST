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

THRESHOULD_NORMAL = 0.2
THRESHOULD_ANOMALY = 16.0

@app.route('/', methods=['GET', 'POST'])
def index():
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [
        *deepcopy(shapes), *deepcopy(numbers), 
        *deepcopy(colors)]

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
    if session['n'] % 10 == 0:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]
    if 'prev_result' not in session:
        session['prev_result'] = 'Incorrect'

    img_list = []
    dummy_list = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        img_list.append(sample)
        dummy_list.append([get_dummy_variable(sample)])

    q = torch.tensor([[1, -1]], dtype=torch.float32)
    q = q.to(device)
    result = reasoner(q)
    result_idx = torch.argmax(result)
    pred_correct_attr = all_attr[result_idx]
    unselected_cards = []
    for i, d in enumerate(dummy_list):
        if d[0][result_idx] == 1:
            selected_card_idx = i
            selected_card = d[0]
            selected_card_img = img_list[selected_card_idx]
            target_card_idx = i
        else:
            unselected_cards.append(d)

    target = session['correct']
    target_idx = all_attr.index(session['correct'])

    result = 'Correct' if selected_card[target_idx] == 1 \
        else 'Incorrect'
    
    if result == 'Correct':
        session['score'] += 1
        q_real = [[1, -1]]
    else:
        q_real = [[-1, 1]]
    q_real = torch.tensor(q_real, dtype=torch.float32)
    q_real = q_real.to(device)

    state = torch.tensor(
        dummy_list[target_card_idx], dtype=torch.float32)
    state = state.to(device)
    pred_state = reasoner(q_real)
    diff = detector(pred_state, state)
    target_diff = diff[0][result_idx]

    anomaly_value = torch.abs(target_diff).detach().numpy()
    is_normal = \
        True if torch.abs(target_diff) < THRESHOULD_NORMAL \
        else False
    is_anomaly = \
        True if torch.abs(target_diff) > THRESHOULD_ANOMALY \
        else False

    n_epoch = 1000

    if result == 'Incorrect' and \
        session['prev_result'] == 'Incorrect' and is_anomaly:
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
                    card, dtype=torch.float32).to(device)
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

    session['prev_result'] = result

    return render_template('index.html',
                           img_list=img_list,
                           correct_card_attribute=target, 
                           result=result,
                           n=session['n'],
                           score=session['score'],
                           correct=session['correct'],
                           selected_card_img=selected_card_img,
                           anomaly_value=anomaly_value,
                           is_anomaly=is_anomaly,
                           dict_pred_correct_feature=\
                            dict_pred_correct_feature,
                           pred_correct_attr=pred_correct_attr)

if __name__ == '__main__':
    app.run()
