# Copyright 2015 Matthew J. Aburn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Simulation of standard multiple stochastic integrals, both Ito and Stratonovich
I_{ij}(t) = \int_{0}^{t}\int_{0}^{s} dW_i(u) dW_j(s)  (Ito)
J_{ij}(t) = \int_{0}^{t}\int_{0}^{s} \circ dW_i(u) \circ dW_j(s)  (Stratonovich)

These multiple integrals I and J are important building blocks that will be
used by most of the higher-order algorithms that integrate multi-dimensional
SODEs.

We first implement the method of Kloeden, Platen and Wright (1992) to
approximate the integrals by the first n terms from the series expansion of a
Brownian bridge process. By default using n=5.

Finally we implement the method of Wiktorsson (2001) which improves on the
previous method by also approximating the tail-sum distribution by a
multivariate normal distribution.
"""

import numpy as np


def deltaW(n, m, delta_t):
    """pre-generate sequence of Wiener increments for each of m independent
    Wiener processes W_j(t) j=1..m over n time steps with constant
    time step size delta_t.
    
    Returns:
      array of shape (n, m), where the [k, j] element has the value
        W_j((k+1)delta_t) - W_j(k*delta_t)
    """
    sqrth = np.sqrt(h)
    return np.random.normal(0.0, sqrth, (n, m))


def _kprod(A, B):
    """Kroenecker tensor product of two matrices.
    If A is m x n matrix and B is p x q then _kprod(A,B) is a mp x nq matrix.
    """
    pass


def _vec(A):
    """Stack columns of matrix A on top of each other to give a long vector.
    If A is m x n matrix then _vec(A) will be mn x 1
    """


def Ikp(h, m, n=5):
    pass


def Jkp(h, m, n=5):
    pass


def Iwiktorsson(h, m, n=5):
    pass


def Jwiktorsson(h, m, n=5):
    pass
