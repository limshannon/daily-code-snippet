# Source: www.machinelearningmastery.com

from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot

def objective(x):
	return x**2.0

def derivative(x):
	return x * 2.0

def gradient_descent(objective, derivative, bounds, n_iter, learningRate):
    solutions, scores = list(), list()
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    print(solution)
    for i in range(n_iter):
        gradient = derivative(solution)
        solution = solution - learningRate * gradient
        solution_eval = objective(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solutions, scores]

bounds = asarray([[-1.0, 1.0]])
n_iter = 30
learningRate = 0.1
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, learningRate)
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
results = objective(inputs)
pyplot.plot(inputs, results)
pyplot.plot(solutions, scores, '.-', color='red')
pyplot.show()