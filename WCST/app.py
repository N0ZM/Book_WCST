from flask import Flask, render_template, request, session
import numpy as np
from copy import deepcopy

app = Flask(__name__)
app.secret_key = 'top_secret'

@app.route('/', methods=['GET', 'POST'])
def index():
    shapes = ['Star', 'Square', 'Circle', 'Cross', 'Triangle']
    colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
    numbers = ['1', '2', '3', '4', '5']
    all_attr = [
        *deepcopy(shapes), *deepcopy(numbers), 
        *deepcopy(colors)]

    if not 'n' in session:
        session['n'] = 0
    else:
        session['n'] += 1
    if not 'score' in session:
        session['score'] = 0
    if not 'correct' in session:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]
    if session['n'] == 50:
        session['n'] = 0
        session['score'] = 0
    if session['n'] % 10 == 0:
        session['correct'] = all_attr[
            np.random.choice(len(all_attr))]

    img_list = []
    for _ in range(len(shapes)):
        s = shapes.pop(np.random.choice(len(shapes)))
        c = colors.pop(np.random.choice(len(colors)))
        n = numbers.pop(0)
        sample = f'{s}{n}{c}.png'
        img_list.append(sample)
    result = None
    if request.method == 'POST':
        ans = request.form['selected']
        if session['correct'] in ans:
            result = 'Correct'
            session['score'] += 1
        else:
            result = 'Incorrect'
    return render_template('index.html',
                           img_list=img_list,
                           result=result,
                           n=session['n'],
                           score=session['score'],
                           correct=session['correct'])

if __name__ == '__main__':
    app.run()
