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


def deltaW(N, m, delta_t):
    """Generate sequence of Wiener increments for each of m independent
    Wiener processes W_j(t) j=1..m over N time steps with constant
    time step size delta_t.
    
    Returns:
      array of shape (N, m), where the [k, j] element has the value
        W_j((k+1)delta_t) - W_j(k*delta_t)
    """
    return np.random.normal(0.0, np.sqrt(delta_t), (N, m))


def _vec(A):
    """Stack columns of matrix A on top of each other to give a long vector.
    If A is m x n matrix then _vec(A) will be mn x 1
    """
    return A.T.reshape((A.size, 1))


def _t(a):
    """transpose the last two axes of a three axis array"""
    return a.transpose((0, 2, 1))


def _dot(a, b):
    """ for rank 3 arrays a and b, return \sum_k a_ij^k . b_ik^l (no sum on i)
    i.e. This is just normal matrix multiplication at each point on first axis
    """
    return np.einsum('ijk,ikl->ijl', a, b)


def _Aterm(N, h, m, k, dW):
    sqrt2h = np.sqrt(2.0/h)
    Xk = np.random.normal(0.0, 1.0, (N, m, 1))
    Yk = np.random.normal(0.0, 1.0, (N, m, 1))
    term1 = _dot(Xk, _t(Yk + sqrt2h*dW))
    term2 = _dot(Yk + sqrt2h*dW, _t(Xk))
    return (term1 - term2)/k


def Ikp(N, h, m, n=5):
    dW = deltaW(N, m, h)
    dW = np.expand_dims(dW, -1) # array of shape N x m x 1
    A = _Aterm(N, h, m, 1, dW)
    for k in range(2, n+1):
        A += _Aterm(N, h, m, k, dW)
    A = (h/(2.0*np.pi))*A
    I = 0.5*(_dot(dW, _t(dW)) - np.diag(h*np.ones(m))) + A
    return (dW, A, I)


def Jkp(h, m, n=5):
    pass


def Iwiktorsson(h, m, n=5):
    pass


def Jwiktorsson(h, m, n=5):
    pass
