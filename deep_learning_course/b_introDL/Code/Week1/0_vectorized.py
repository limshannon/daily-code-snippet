# Source: www.machinelearningmastery.com

import numpy as np

def unvectorized(n, theta, x):
    prediction = 0.0
    for j in range(n):
        prediction = prediction + theta[j] * x[j]
    return prediction

def vectorized(theta, x):
    prediction = 0.0
    prediction = np.dot(theta.transpose(), x)
    return prediction

n = 10000
theta = np.random.rand(n)
x = np.random.randint(0, 100, n)

print(unvectorized(n, theta, x))
print(vectorized(theta, x))
