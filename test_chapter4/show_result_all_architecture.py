import os
import numpy as np
import pandas as pd
from scipy import stats as st
from matplotlib import pyplot as plt


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
    plt.show()

par_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
result_files = [
    os.path.join(par_dir,
                'test_chapter4/result/random.csv'),
    os.path.join(par_dir, 
                 'test_chapter2/result/policy_no_detector.csv'), 
    os.path.join(par_dir, 
                 'test_chapter2/result/policy_on_detector.csv'), 
    os.path.join(par_dir,
                 'test_chapter3/result/reasoner.csv'), 
    os.path.join(par_dir, 
                 'test_chapter4/result/policy_with_reasoner.csv')] 

plot_dist(*result_files)
