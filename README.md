ransac
======

Fits a model to observed data via the RANSAC-method.

Introduction
------------

ransac is a Python module that provides the RANSAC-method for fitting a model to
observed data. The method is robust as it ignores outlier data. An *iterative*,
*non-deterministic* algorithm is used which means that you might need to tune it
in order for it to work for a specific problem.

You can read more about RANSAC [here](http://en.wikipedia.org/wiki/RANSAC).

Installation
------------

Install ransac with ```python setup.py install```

Usage
-----
```python
>>> import numpy as np
>>> X = np.asarray(range(10))
>>> Y = 2 * X
>>> data = np.asarray([X, Y]).T
>>> import ransac
>>> model = ransac.LineModel()
>>> (params, inliers, residual) = ransac.ransac(data, model, 2, 8)
>>> params
[2.0, 0.0]
>>> residual
0.0
```

License
-------

Distributed under the MIT license. See the ```LICENSE``` file.

