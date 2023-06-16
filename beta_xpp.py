from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt


xs = np.linspace(0, 1, 1001)

pds = st.beta.pdf(xs, 1, 9)
pp = st.beta.ppf(0.9, 1, 9)
print(pp)
interval_x = xs[xs > pp]
interval_pd = pds[xs > pp]

print(np.log(pp / 0.01))

pds2 = st.beta.pdf(xs, 9, 1)
pp2 = st.beta.ppf(0.1, 9, 1)
print(pp2)
interval_x2 = xs[xs < pp2]
interval_pd2 = pds2[xs < pp2]

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(xs, pds, label='Beta')
ax1.fill_between(interval_x, interval_pd, alpha=0.3, 
                 label='10% Rejection area')
ax1.axvline(pp, linewidth=0.5, color='black', 
            label=f'10% point: {round(pp, 4)}')
ax1.legend()

ax2.plot(xs, pds2, label='Beta')
ax2.fill_between(interval_x2, interval_pd2, alpha=0.3, 
                 label='10% Rejection area')
ax2.axvline(pp2, linewidth=0.5, color='black', 
            label=f'10% point: {round(pp2, 4)}')
ax2.legend()

plt.savefig('beta_xpp.png', dpi=300)
plt.show()
