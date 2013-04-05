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
        Fits the model to the data, minimizing the sum of absolute
        errors.

        The fitting is done in the simplest manner possible; drawing a
        line through two of the samples instead of the more common
        least squares method.

        """
        X = data[:,0]
        Y = data[:,1]
        denom = (X[-1] - X[0])
        if denom == 0:
            raise ZeroDivisionError
        k = (Y[-1] - Y[0]) / denom
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

    min_inliers : number
        The number of inliers required to assert that the model
        is a good fit. If 0 < min_inliers < 1 then min_inliers
        is considered a percentage of the number of samples in
        the data.

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

    Raises
    ------
    ValueError
        If the algorithm could not find a good fit for the data.

    """
    if len(data) <= min_samples:
        raise ValueError("Not enough input data to fit the model.")

    if 0 < min_inliers < 1:
        min_inliers = int(min_inliers * len(data))

    best_params     = None
    best_inliers    = None
    best_residual   = np.inf

    for i in xrange(iterations):
        indices = range(len(data))
        shuffle(indices)
        inliers             = np.asarray([data[i] for i in indices[:min_samples]])
        shuffled_data       = np.asarray([data[i] for i in indices[min_samples:]])

        try:
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
        except ZeroDivisionError:
            pass

    if not best_params:
        raise ValueError("RANSAC failed to find a sufficiently good fit for "
                "the data. Check that input data has sufficient rank.")

    return (best_params, best_inliers, best_residual)

