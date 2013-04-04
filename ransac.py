from __future__ import division

import numpy as np

from random import sample

class Model(object):
    def fit(self, data):
        raise NotImplementedError

    def distance(self, point):
        raise NotImplementedError

class LineModel(Model):
    """
    A 2D line model.
    """

    def fit(self, data):
        """
        Fits the model to the data, minimizing the sum of absolute errors.
        """
        data = np.asarray(data)
        X = data[:,0]
        Y = data[:,1]
        #ret = np.polyfit(X, Y, 1, full=True)
        #self.params     = ret[0]
        #self.residual   = ret[1]
        k = (Y[-1] - Y[0]) / (X[-1] - X[0])
        m = Y[0] - k * X[0]
        self.params = [k, m]
        self.residual = sum(self.distance(p) for p in data)

    def distance(self, point):
        """
        Calculates the shortest vertical distance from a data point to the model.
        """
        x = point[0]
        y = point[1]
        k = self.params[0]
        m = self.params[1]
        dist = abs(k * x + m - y)

        return dist

def ransac(data, model, min_samples, min_inliers, iterations=100, eps=1e-10):
    """
    Fits a model to observed data.

    Uses the RANSAC iterative method of fitting a model to observed
    data. The method is robust as it ignores outlier data.

    Parameters
    ----------
    data : numpy.ndarray
        The data that the model should be fitted to.

    model : Model
        The model that is used to describe the data.

    min_samples : int
        The minimum number of samples needed to fit the model.

    min_inliers : int
        The number of inliers required to assert that the model
        is a good fit.

    iterations : int
        The number of iterations that the algorithm should run.

    eps : float
        The maximum allowed distance from the model that a sample is
        allowed to be to be counted as an inlier.

    Returns
    -------
    best_params : list
        The parameters of the model with the best fit.

    best_inliers : list
        A list of the inliers belonging to the model with the best fit.

    best_residual : float
        The residual of the inliers.

    """
    best_params     = None
    best_inliers    = None
    best_residual   = float('inf')

    for i in xrange(iterations):
        potential_inliers = sample(xrange(len(data)), min_samples)
        inliers = [data[i] for i in potential_inliers]
        model.fit(inliers)

        for (i, point) in enumerate(data):
            if i not in potential_inliers and model.distance(point) <= eps:
                inliers.append(point)

        if len(inliers) >= min_inliers:
            model.fit(inliers)
            if model.residual < best_residual:
                best_params     = model.params
                best_inliers    = inliers
                best_residual   = model.residual

    return (best_params, best_inliers, best_residual)

