
import numpy as np
import matplotlib.pyplot as plt
from gmm_em import gaussian
#1:
n1 = 1000
mu1 = [0]
sigma2_1 = [[0.3]]

sample1 = np.random.multivariate_normal(mean=mu1, cov=sigma2_1, size=n1)
Y = sample1

Y_plot = np.linspace(-3, 3, 1000)[:, np.newaxis]
true_dens = (gaussian(Y_plot[:, 0], mu1, sigma2_1))

fig, ax = plt.subplots()
ax.fill(Y_plot[:, 0], true_dens, fc='blue', alpha=0.2, label='true distribution')
ax.plot(sample1[:, 0], -0.005 - 0.01 * np.random.random(sample1.shape[0]), 'b+', label="input")
ax.set_xlim(-3, 3)
plt.show()

mu = [np.mean(Y)]
sigma2 = [[np.mean((Y-mu)**2)]]
est_dens = (gaussian(Y_plot[:, 0], mu, sigma2))

ax.plot(Y_plot[:, 0], est_dens, 'black',label='estimated distribution')
plt.legend(loc="best")
plt.title("Gaussian Distribution")
plt.show()
