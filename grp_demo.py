import torch

from GPR import GPR
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

gpr = GPR()
gpr.max_step = 100000

n = 10 # n >=2
m = 1000

dt = 3/(n-1)

def foo(x):
    if x<1:
        return [x,0]
    elif x<2:
        return [1,x-1]
    else:
        return [3-x, 1]
    return [x, x*x]


times = [i*dt for i in range(n)]
anchors = [foo(i*dt) for i in range(n)]
# times = [0,1]
# anchors = [[0,0],[1,0]]
a = [0,0]
da = [1,0]
eps = 0.05

times.append(-eps)
anchors.append(foo(-eps))

time_array = [i*3/m for i in range(m)]

times = torch.FloatTensor(times).unsqueeze(1)
anchors = torch.FloatTensor(anchors)
time_array = torch.FloatTensor(time_array).unsqueeze(1)
print(times.shape)
print(anchors.shape)
print(time_array.shape)

# for x-axis
gpr.set_hyperparameter(0.1,0.1,0.1)
x_mean = anchors[:,0].mean()
print(x_mean)
gpr.load_data(times, anchors[:,0])
gpr.optimize()
mu1, cov1 = gpr.predict_posterior(time_array)
print(cov1)
x = np.random.multivariate_normal(mu1.detach().numpy(), cov1.detach().numpy(), 10)
print(x)

# for y-axis
gpr.set_hyperparameter(0.1,0.1,0.1)
y_mean = anchors[:,0].mean()
gpr.load_data(times, anchors[:,1])
gpr.optimize()
mu2, cov2 = gpr.predict_posterior(time_array)
y = np.random.multivariate_normal(mu2.detach().numpy(), cov2.detach().numpy(), 10)


plt.scatter(x, y, alpha=0.5)
plt.scatter(anchors.detach().numpy()[:,0], anchors.detach().numpy()[:,1], s=5, color='red')
plt.show()
