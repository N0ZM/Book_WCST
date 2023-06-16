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

from models import Policy, Detector
from utils import get_dummy_variable


app = Flask(__name__)
app.secret_key = 'top_secret'
static_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'static'))

# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2)
detector = Detector()

policy.to(device)

THRESHOULD_NORMAL = 0.2
THRESHOULD_ANOMALY = 8.0

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
    if session['n'] % 20 == 0:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]

    img_list = []
    dummy_list = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        img_list.append(sample)
        dummy_list.append([get_dummy_variable(sample)])

    target = session['correct']
    target_idx = all_attr.index(session['correct'])
    
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
    selected_card_img = img_list[res_idx]
    
    if result == 'Correct':
        session['score'] += 1

    n_epoch = 500
    
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

    anomaly_value = torch.abs(diff).detach().numpy()
    is_normal = True if torch.abs(diff) < THRESHOULD_NORMAL \
                else False
    is_anomaly = True if torch.abs(diff) > THRESHOULD_ANOMALY \
                else False

    # Init policy if anomaly
    if result == 'Incorrect' and is_anomaly:
        policy.init()
    # Train policy if not normal
    if not (result == 'Correct' and is_normal):
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
                           )

if __name__ == '__main__':
    app.run()
