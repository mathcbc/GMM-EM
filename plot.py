# author: Bing-Cheng Chen

import numpy as np
from scipy.stats import multivariate_normal
from numpy import genfromtxt
import matplotlib.pyplot as pyplot

import pickle

with open('results.pkl', 'rb') as f:
    [prior, mu, sigma, ll_evol] = pickle.load(f)

# Show how the log-likelihood evolves as the training proceeds
pyplot.plot(ll_evol, 'o')
pyplot.show()

# The learned mathematical expression for the GMM model after training on the given dataset
print('prior:',prior)
print('mu:', mu)
print('sigma:', sigma)

# Randomly select 500 data points from the given dataset and plot them on a 2-D coordinate system.
# Mark the data points coming from the same cluster with the same color.

# Reading the data file
X = genfromtxt('TrainingData_GMM.csv', delimiter=',')
print('data shape:', X.shape)

sel_num = 500
X_sel = []
sel_idxs = []
while len(sel_idxs) < sel_num:
    idx = np.random.randint(0, 5000, 1)
    while idx in sel_idxs:
        idx = np.random.randint(0, 5000, 1)
    sel_idxs.append(idx[0])
X_sel = X[sel_idxs]


# get the labels of the points
def get_label(x, prior, mu, sigma):
    K = len(prior)
    p = np.zeros(K)
    for k in range(0,K):
        p[k] = prior[k] * multivariate_normal.pdf(x, mu[k,:], sigma[k,:,:])
    label = np.argmax(p)
    return label

lbs = []
for i in range(0, sel_num):
    lb = get_label(X_sel[i], prior, mu, sigma)
    lbs.append(lb)

# plot
pyplot.scatter(X_sel[:,0], X_sel[:,1], marker='o', c=lbs)
pyplot.show()
