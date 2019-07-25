
import numpy as np
import matplotlib.pyplot as plt
from gmm_em import gmm_em
from gmm_em import gaussian
from sklearn.neighbors import KernelDensity

D = 1 # 2d data
K = 3 # 3 mixtures

#1:
n1 = 70
mu1 = [0]
sigma2_1 = [[0.3]]
#2:
n2 = 150
mu2 = [2]
sigma2_2  = [[0.2]]
#3:
n3 = 100
mu3 = [4]
sigma2_3  = [[0.3]]

N = n1 + n2 + n3
alpha1 = n1/N
alpha2 = n2/N
alpha3 = n3/N


sample1 = np.random.multivariate_normal(mean=mu1, cov=sigma2_1, size=n1)
sample2 = np.random.multivariate_normal(mean=mu2, cov=sigma2_2, size=n2)
sample3 = np.random.multivariate_normal(mean=mu3, cov=sigma2_3, size=n3)
Y = np.concatenate((sample1, sample2, sample3),axis=0)


Y_plot = np.linspace(-2, 6, 1000)[:, np.newaxis]
true_dens = (alpha1 * gaussian(Y_plot[:, 0], mu1, sigma2_1)
             + alpha2 * gaussian(Y_plot[:, 0], mu2, sigma2_2)
             + alpha2 * gaussian(Y_plot[:, 0], mu3, sigma2_3))

fig, ax = plt.subplots()
ax.fill(Y_plot[:, 0], true_dens, fc='black', alpha=0.2, label='true distribution')
ax.plot(sample1[:, 0], -0.005 - 0.01 * np.random.random(sample1.shape[0]), 'b+', label="input class1")
ax.plot(sample2[:, 0], -0.005 - 0.01 * np.random.random(sample2.shape[0]), 'r+', label="input class2")
ax.plot(sample3[:, 0], -0.005 - 0.01 * np.random.random(sample3.shape[0]), 'g+', label="input class3")

kernel = 'gaussian'
kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(Y)
log_dens = kde.score_samples(Y_plot)
ax.plot(Y_plot[:, 0], np.exp(log_dens), '-', label="input distribution".format(kernel))

ax.set_xlim(-2, 6)

plt.show()

omega, alpha, mu, cov = gmm_em(Y, K, 100)

category = omega.argmax(axis=1).flatten().tolist()
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
class3 = np.array([Y[i] for i in range(N) if category[i] == 2])

est_dens = (alpha[0] * gaussian(Y_plot[:, 0], mu[0], cov[0])
             + alpha[1] * gaussian(Y_plot[:, 0], mu[1], cov[1])
             + alpha[2] * gaussian(Y_plot[:, 0], mu[2], cov[2]))

ax.fill(Y_plot[:, 0], est_dens, fc='blue', alpha=0.2, label='estimated distribution')
ax.plot(class1[:, 0], -0.03 - 0.01 * np.random.random(class1.shape[0]), 'bo', label="estimated class1")
ax.plot(class2[:, 0], -0.03 - 0.01 * np.random.random(class2.shape[0]), 'ro', label="estimated class2")
ax.plot(class3[:, 0], -0.03 - 0.01 * np.random.random(class3.shape[0]), 'go', label="estimated class3")
plt.legend(loc="best")
plt.title("GMM Data")
plt.show()

#Errors:
print("Original alpha: ", alpha1, alpha2, alpha3)
print("Estimated alpha: ", alpha[0], alpha[1], alpha[2])

print("Original mu: ", mu1, mu2, mu3)
print("Estimated mu: ", mu[0], mu[1], mu[2])

print("Original cov: ", sigma2_1, sigma2_2, sigma2_3)
print("Estimated cov: ", cov[0], cov[1], cov[2])