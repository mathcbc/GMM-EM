# author: Bing-Cheng Chen

import numpy as np
from scipy.stats import multivariate_normal
from numpy import genfromtxt


def Estep(X, prior, mu, sigma):
    N, D = X.shape
    K = len(prior)
    gama_mat = np.zeros((N,K))
    for i in range(0,N):
        xi = X[i, :]
        sum = 0
        for k in range(0, K):
            p = multivariate_normal.pdf(xi, mu[k,:], sigma[k,:,:])
            sum += prior[k] * p
        for k in range(0,K):
            gama_mat[i, k] = prior[k] * multivariate_normal.pdf(xi, mu[k,:], sigma[k,:,:]) / sum
    return gama_mat


def Mstep(X, gama_mat):
    (N, D) = X.shape
    K = np.size(gama_mat, 1)
    mu = np.zeros((K, D))
    sigma = np.zeros((K, D, D))
    prior = np.zeros(K)
    for k in range(0, K):
        N_k = np.sum(gama_mat, 0)[k]
        for i in range(0,N):
            mu[k] += gama_mat[i, k] * X[i]
        mu[k] /= N_k
        for i in range(0, N):
            left = np.reshape((X[i] - mu[k]), (2,1))
            right = np.reshape((X[i] - mu[k]), (1,2))
            sigma[k] += gama_mat[i,k] * left * right
        sigma[k] /= N_k
        prior[k] = N_k/N
    return mu, sigma, prior


def logLike(X, prior, Mu, Sigma):
    K = len(prior)
    N, M = np.shape(X)
    # P is an NxK matrix where (i,j)th element represents the likelihood of
    # the ith datapoint to be in jth Cluster
    P = np.zeros([N, K])
    for k in range(K):
        for i in range(N):
            P[i, k] = multivariate_normal.pdf(X[i], Mu[k, :], Sigma[k, :, :])
    return np.sum(np.log(P.dot(prior)))


def main():
    # Reading the data file
    X = genfromtxt('TrainingData_GMM.csv', delimiter=',')
    print('training data shape:', X.shape)

    N, D = X.shape
    K = 4

    # initialization
    mu = np.zeros((K, D))
    sigma = np.zeros((K, D, D))
    # mu[0] = [-0.5, -0.5]
    # mu[1] = [0.3, 0.3]
    # mu[2] = [-0.3, 0.3]
    # mu[3] = [1.3, -1.3]

    mu[0] = [1, 0]
    mu[1] = [0, 1]
    mu[2] = [0, 0]
    mu[3] = [1, 1]

    for k in range(0, K):
        sigma[k] = [[1, 0], [0, 1]]

    prior = np.ones(K) / K
    iter = 0
    prevll = -999999
    ll_evol = []

    print('initialization of the params:')
    print('mu:\n', mu)
    print('sigma:\n', sigma)
    print('prior:', prior)

    # Iterate with E-Step and M-step
    while (True):
        W = Estep(X, prior, mu, sigma)
        mu, sigma, prior = Mstep(X, W)
        ll_train = logLike(X, prior, mu, sigma)
        print('iter:',iter, 'log likelihood:',ll_train)
        ll_evol.append(ll_train)

        iter = iter + 1
        if (iter > 150 or abs(ll_train - prevll) < 0.01):
            break

        abs(ll_train - prevll)
        prevll = ll_train

    import pickle
    with open('results.pkl', 'wb') as f:
        pickle.dump([prior, mu, sigma, ll_evol], f)



if __name__ == '__main__':
    main()