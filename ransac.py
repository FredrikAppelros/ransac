from __future__ import division

import numpy as np

from random import shuffle

class Model(object):
    def fit(self, data):
        raise NotImplementedError

    def distance(self, samples):
        raise NotImplementedError

class LineModel(Model):
    """
    A 2D line model.
    """

    def fit(self, data):
        """
        Fits the model to the data, minimizing the sum of absolute errors.
        """
        X = data[:,0]
        Y = data[:,1]
        #ret = np.polyfit(X, Y, 1, full=True)
        #self.params     = ret[0]
        #self.residual   = ret[1]
        k = (Y[-1] - Y[0]) / (X[-1] - X[0])
        m = Y[0] - k * X[0]
        self.params = [k, m]
        self.residual = sum(abs(k * X + m - Y))

    def distance(self, samples):
        """
        Calculates the vertical distances from the samples to the model.
        """
        X = samples[:,0]
        Y = samples[:,1]
        k = self.params[0]
        m = self.params[1]
        dists = abs(k * X + m - Y)

        return dists

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

    best_inliers : numpy.ndarray
        A list of the inliers belonging to the model with the best fit.

    best_residual : float
        The residual of the inliers.

    """
    best_params     = None
    best_inliers    = None
    best_residual   = np.inf

    for i in xrange(iterations):
        indices = range(len(data))
        shuffle(indices)
        inliers             = np.asarray([data[i] for i in indices[:min_samples]])
        shuffled_data       = np.asarray([data[i] for i in indices[min_samples:]])

        model.fit(inliers)
        dists = model.distance(shuffled_data)
        more_inliers = shuffled_data[np.where(dists <= eps)[0]]
        inliers = np.concatenate((inliers, more_inliers))

        if len(inliers) >= min_inliers:
            model.fit(inliers)
            if model.residual < best_residual:
                best_params     = model.params
                best_inliers    = inliers
                best_residual   = model.residual

    return (best_params, best_inliers, best_residual)

