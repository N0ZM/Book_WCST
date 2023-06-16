from matplotlib import pyplot as plt
from matplotlib import patches
import os


p_star = [[0.15, 0.], [0.5, 1.], 
          [0.85, 0.], [0., 0.66], [1., 0.66]]
r_star = 0.5

n_sample = 5
n_row = 1
n_col = 5

colors = ['Blue', 'Green', 'Red', 'Yellow', 'Black']
shape = 'Star'

# Card size: 2x3
p_samples = [
    [[1., 1.5]], 
    [[1., 2.], [1., 1.]],
    [[1., 2], [0.5, 1.], [1.5, 1.]],
    [[0.5, 2.], [1.5, 2.], [0.5, 1.], [1.5, 1.]],
    [[0.5, 2.], [1.5, 2.], [1., 1.5], [0.5, 1.], [1.5, 1.]]
]

# Card size: 3x4
p_samples_3x4 = [
    [[1.5, 2.]], 
    [[1.5, 3.], [1.5, 1.]],
    [[1.5, 3], [1., 1.], [2., 1.]],
    [[1., 3.], [2., 3.], [1., 1.], [2, 1.]],
    [[1., 3.], [2., 3.], [1.5, 2.], [1., 1.], [2, 1.]]
]

for c in colors:
    for i in range(n_sample):
        fig = plt.figure(figsize=[0.64, 0.64])
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_xlim(0, 2)
        # ax.set_ylim(0, 3)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.tick_params(
            labelbottom=False, labelleft=False, 
            labelright=False, labeltop=False,
            bottom=False, left=False, right=False, top=False
        )

        points = p_samples_3x4[i]
        for point in points:
            xy = []
            for p in p_star:
                tmp = [p[0] + point[0] - r_star, 
                       p[1] + point[1] - r_star]
                xy.append(tmp)
            patch = patches.Polygon(xy=xy, fc=c, closed=True)
            ax.add_patch(patch)

        dir_name = os.path.join(
            os.path.dirname(__file__), 'imgs')
        os.makedirs(dir_name, exist_ok=True)
        f_name = os.path.join(dir_name, f'{shape}{i+1}{c}.png')
        fig.savefig(f_name)

plt.show()
