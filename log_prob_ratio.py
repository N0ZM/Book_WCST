import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d


def log_prob_ratio(p, q):
    p += 1e-8
    q += 1e-8
    return np.log(p / q)

n_scale = 100
xs = np.linspace(0.1, 0.9, n_scale)
ys = np.linspace(0.1, 0.9, n_scale)
lpr = np.zeros([n_scale, n_scale])

for i in range(len(xs)):
    for j in range(len(ys)):
        lpr[i, j] = log_prob_ratio(xs[i], ys[j])

xs, ys = np.meshgrid(xs, ys)

plt.figure(figsize=[5, 5])
ax = plt.subplot(111, projection='3d')
ax.plot_surface(xs, ys, lpr)
ax.set_xlabel('p(x)')
ax.set_ylabel('p(y)')
ax.set_zlabel('LPR(p(x),q(x))')

plt.savefig("log_prob_ratio.png", format="png", dpi=300)
plt.show()
