#! /usr/bin/env python

import numpy as np
import gc
import matplotlib.pyplot as plt

from random import seed, sample, randint
from ransac import LineModel, ransac
from time import time

random_seed     = 0

num_iterations  = 100
num_samples     = 1000
noise_ratio     = 0.8
num_noise       = int(noise_ratio * num_samples)

def setup():
    global data, model
    seed(random_seed)
    X = np.asarray(range(num_samples))
    Y = 2 * X
    noise = [randint(0, 2 * (num_samples - 1)) for i in xrange(num_noise)]
    Y[sample(xrange(len(Y)), num_noise)] = noise
    data = np.asarray([X, Y]).T
    model = LineModel()

    plt.plot(X, Y, 'bx')

def run():
    global params, residual, mean_time
    gc.disable()
    start_time = time()
    for i in xrange(num_iterations):
        try:
            (params, inliers, residual) = ransac(data, model, 2, (1 - noise_ratio) * num_samples)
        except ValueError:
            pass
    end_time = time()
    mean_time = (end_time - start_time) / num_iterations
    gc.enable()

def summary():
    if params:
        print ' Parameters '.center(40, '=')
        print params
        print ' Residual '.center(40, '=')
        print residual
        print ' Time '.center(40, '=')
        print '%.1f msecs mean time spent per call' % (1000 * mean_time)
        X = np.asarray([0, num_samples - 1])
        Y = params[0] * X + params[1]
        plt.plot(X, Y, 'k-')
    else:
        print 'RANSAC failed to find a sufficiently good fit for the data.'

    plt.show()

if __name__ == '__main__':
    setup()
    run()
    summary()

