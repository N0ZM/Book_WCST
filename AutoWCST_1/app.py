import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from copy import deepcopy

from flask import Flask, render_template, session
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

import sys
par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(par_dir)

from models import Policy
from utils import get_dummy_variable


app = Flask(__name__)
app.secret_key = 'top_secret'
static_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'static'))

# device = torch.device(
#     'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

policy = Policy(15, 128, 2)
policy.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [
        *deepcopy(shapes), *deepcopy(numbers), *deepcopy(colors)]

    session['correct'] = '4'
    if not 'n' in session:
        session['n'] = 1
    else:
        session['n'] += 1
    if not 'score' in session:
        session['score'] = 0
    if session['n'] == 101:
        session['n'] = 1
        session['score'] = 0
    # if session['n'] % 20 == 0:
    #     session['correct'] = all_attr[
    #         np.random.choice(len(all_attr))]

    img_list = []
    dummy_list = []
    action_probs = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        img_list.append(sample)
        dv = [get_dummy_variable(sample)]
        dummy_list.append([get_dummy_variable(sample)])
        dv = torch.tensor(dv, dtype=torch.float32).to(device)
        ap = policy(dv)
        action_probs.append(float(ap[0][0]))

    target = session['correct']
    target_idx = all_attr.index(target)
    selected_card_idx = np.argmax(action_probs)
    selected_card_img = img_list[selected_card_idx]
    selected_card = dummy_list[selected_card_idx][0]

    result = 'Correct' if selected_card[target_idx] == 1 \
        else 'Incorrect'

    n_epoch = 100
    if result == 'Correct':
        session['score'] += 1
        for _ in range(n_epoch):
            policy.fit([selected_card], q=[[1, -1]])
    else:
        for _ in range(n_epoch):
            policy.fit([selected_card], q=[[-1, 1]])

    return render_template('index.html',
                           img_list=img_list,
                           correct_card_attribute=target, 
                           result=result,
                           n=session['n'],
                           score=session['score'],
                           correct=session['correct'],
                           selected_card_img=selected_card_img)

if __name__ == '__main__':
    app.run()
