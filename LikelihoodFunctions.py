from math import e, pi, sqrt, factorial, exp
from numpy import linspace
import numpy as np
from BasicBayes.Exceptions import *
import math
import BasicBayes.Utils as utils

def choose(x, n):
    """Performs "choose" operation.
    
    Arguments:
        x {int} -- Number of successes
        n {int} -- Number of trials
    """
    return factorial(n)/( factorial(x) * factorial(n-x) )

def poisson(x, l):
    x = int(x)
    l = int(l)
    ls = list( [l for _ in range(x)] )
    es = list( [math.e for _ in range(l)] )
    xfactorials = list( [i+1 for i in range(x)] )
    length = max(len(ls), len(es), len(xfactorials))
    while len(ls) < length:
        ls.append(1)
    while len(es) < length:
        es.append(1)
    while len(xfactorials) < length:
        xfactorials.append(1)
    p = 1
    for i in range(length):
        if ls[i]*es[i]*xfactorials[i] < 0: assert False
        p *= ls[i]
        p /= es[i]
        p /= xfactorials[i]
    if p > 1 or p < 0: assert False
    return p

def binomial(x, n, p):
    """Calculated binomial distribution
    
    Arguments:
        x {int} -- Number of successes
        p {float} -- Probability of success
    
    Returns:
        float -- Probability of x successes
    """
    assert n >= x and x >= 0
    return choose(x, n)*p**x*(1-p)**(n-x)

def gaussian(x, mu, sig):
    """Calculates gaussian distribution
    
    Arguments:
        x {float} -- Data value
        mu {float} -- Average
        sig {float} -- Standard Deviation
    
    Returns:
        float -- Probability of data value
    """
    return 1/(sig*sqrt(2*pi)) * exp(-0.5*( (x-mu)/sig )**2)

def multinomial(hist, ps):
    curr_xs = list( [x for x in hist.values()] )
    ret = 1
    for idx in range(sum(hist.values())):
        ret *= idx + 1
        for x_idx in range(len(curr_xs)):
            ret /= curr_xs[x_idx]
            if curr_xs[x_idx] > 1:
                curr_xs[x_idx] -= 1
    for idx in range(len(hist.keys())):
        ret *= ps[idx]**hist[idx]
    return ret

def gaussian_linear_regression(data_point, sigma, slope, y_intercept):
    mu = data_point[0] * slope + y_intercept
    return gaussian(data_point[1], mu, sigma)

def poisson_linear_regression(data_point, slope, y_intercept):
    return poisson(data_point[1], data_point[0] * slope + y_intercept)

class Regression():
    def __init__(self, regression_function, parameters, likelihood_function):
        self.regression_function = regression_function
        self.parameters = parameters
        self.likelihood_function = likelihood_function
    def __call__(self, x, *args):
        likelihood_parameters = self.regression_function(*x[0:-1], *args)
        if utils.isiterable(likelihood_parameters):
            return self.likelihood_function(x[1], *likelihood_parameters)
        else:
            return self.likelihood_function(x[1], likelihood_parameters)

def linear_regression(x, slope, y_intercept):
    return x * slope + y_intercept

def quadratic_regression(x, a, b, c):
    return a*x**2 + b*x**2 + c

def exponential_regression(x, a, b, c):
    return a*math.e**(b*x) + c

def counts(x, list):
    return sum([x == e for e in list])

if __name__ == "__main__":
    print(zeros(5, 5))
    print(array_sum(zeros(3, 3, 3)))
    print(marginalize(zeros(5, 4, 5), 1))

    zero = zeros(10, 10)
    zero[5][5] = 1
    print(array_max(zero))

    print(gaussian(10, 0, 1))