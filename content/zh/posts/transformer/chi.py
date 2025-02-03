# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats

x = np.linspace(0, 30, 200)
y_1 = 2 * scipy.stats.chi2.pdf(x**2, 1) * x
y_2 = 2 * scipy.stats.chi2.pdf(x**2, 2) * x
y_100 = 2 * scipy.stats.chi2.pdf(x**2, 100) * x
y_200 = 2 * scipy.stats.chi2.pdf(x**2, 200) * x
y_400 = 2 * scipy.stats.chi2.pdf(x**2, 400) * x

z_400 = scipy.stats.norm.pdf(x, 15, 1/np.sqrt(2))

plt.figure(figsize=(8, 5))
plt.xlim(0, 30)
plt.ylim(0, 0.8)
plt.plot(x, y_1, label='d=1')
plt.plot(x, y_2, label='d=2')
plt.plot(x, y_100, label='d=100')
plt.plot(x, y_400, label='d=400')
plt.plot(x, z_400, label='N(15, 1/2)', linestyle='--', color='gray')
plt.legend()

plt.vlines(10, 0, 0.8, linestyles='--', colors='tab:green', alpha=0.5)
plt.vlines(20, 0, 0.8, linestyles='--', colors='tab:red', alpha=0.5)
plt.savefig('chi.jpg')
# %%
