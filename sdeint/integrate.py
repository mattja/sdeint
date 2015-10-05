# Copyright 2015 Matthew J. Aburn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Numerical integration algorithms for Ito and Stratonovich stochastic ordinary
differential equations.

Usage:
    itoint(f, G, y0, tspan)  for Ito equation dy = f dt + G dW
    stratint(f, G, y0, tspan)  for Stratonovich equation dy = f dt + G \circ dW

    y0 is the initial value
    tspan is an array of time values (currently these must be equally spaced)
    f is the deterministic part of the system (can be a scalar or  dx1  vector)
    G is the stochastic part of the system (can be a scalar or  d x m matrix)

sdeint will choose an algorithm for you. Or you can choose one explicitly:

Algorithms implemented so far:
    itoEuler: the Euler Maruyama algorithm for Ito equations.
    stratHeun: the Stratonovich Heun algorithm for Stratonovich equations.
"""

import numpy as np
from .wiener import deltaW


class Error(Exception):
    pass


class SDEValueError(Error):
    """Thrown if integration arguments fail some basic sanity checks"""
    pass


def _check_args(f, G, y0, tspan):
    """Do some common validations. Find dimension d, number of Wiener noises m.
    """
    if not np.isclose(min(np.diff(tspan)), max(np.diff(tspan))):
        raise SDEValueError('Currently time steps must be equally spaced.')
    # Be flexible to allow scalar equations. convert them to a 1D vector system
    if isinstance(y0, numbers.Number):
        if isinstance(y0, numbers.Integral):
            numtype = np.float64
        else:
            numtype = type(y0)
        y0_orig = y0
        y0 = np.array([y0], dtype=numtype)
        def make_vector_fn(fn):
            def newfn(y, t):
                return np.array([fn(y[0], t)], dtype=numtype)
            newfn.__name__ = fn.__name__
            return newfn
        def make_matrix_fn(fn):
            def newfn(y, t):
                return np.array([[fn(y[0], t)]], dtype=numtype)
            newfn.__name__ = fn.__name__
            return newfn
        if isinstance(f(y0_orig, 0.0), numbers.Number):
            f = make_vector_fn(f)
        if isinstance(G(y0_orig, 0.0), numbers.Number):
            G = make_matrix_fn(G)
    # determine dimension d of the system
    d = len(y0)
    if len(f(y0, tspan[0])) != d or len(G(y0, tspan[0])) != d:
        raise SDEValueError('y0, f and G have incompatible shapes.')
    # determine number of independent Wiener processes m
    m = G(y0, tspan[0]).shape[1]
    return (d, m, f, G, y0, tspan)


def itoint(f, G, y0, tspan):
    """ Numerically integrate Ito equation  dy = f dt + G dW
    """
    # In future versions we can automatically choose here the most suitable
    # Ito algorithm based on properties of the system and noise.
    (d, m, f, G, y0, tspan) = _check_args(f, G, y0, tspan)
    chosenAlgorithm = itoEuler
    return chosenAlgorithm(f, G, y0, tspan)


def stratint(f, G, y0, tspan):
    """ Numerically integrate Stratonovich equation  dy = f dt + G \circ dW
    """
    # In future versions we can automatically choose here the most suitable
    # Stratonovich algorithm based on properties of the system and noise.
    (d, m, f, G, y0, tspan) = _check_args(f, G, y0, tspan)
    chosenAlgorithm = stratHeun
    return chosenAlgorithm(f, G, y0, tspan)


def itoEuler(f, G, y0, tspan):
    """Use the Euler-Maruyama algorithm to integrate the Ito equation
    dy = f(y,t)dt + G(y,t) dW(t)

    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    Args:
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      G. Maruyama (1955) Continuous Markov processes and stochastic equations
      Kloeden and Platen (1999) Numerical Solution of Differential Equations
    """
    (d, m, f, G, y0, tspan) = _check_args(f, G, y0, tspan)
    n = len(tspan)
    dt = (tspan[n-1] - tspan[0])/(n - 1)
    # allocate space for result
    y = np.zeros((n, d), dtype=type(y0[0]))
    # pre-generate all Wiener increments (for m independent Wiener processes):
    dWs = deltaW(n - 1, m, dt)
    y[0] = y0;
    for i in range(1, n):
        t1 = tspan[i - 1]
        t2 = tspan[i]
        y1 = y[i - 1]
        dW = dWs[i - 1]
        y[i] = y1 + f(y1, t1)*dt + G(y1, t1).dot(dW)
    return y


def stratHeun(f, G, y0, tspan):
    """Use the Stratonovich Heun algorithm to integrate Stratonovich equation
    dy = f(y,t)dt + G(y,t) \circ dW(t)

    where y is the d-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    Args:
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      G: callable(y, t) returning (d,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      W. Rumelin (1982) Numerical Treatment of Stochastic Differential
         Equations
      R. Mannella (2002) Integration of Stochastic Differential Equations
         on a Computer
      K. Burrage, P. M. Burrage and T. Tian (2004) Numerical methods for strong
         solutions of stochastic differential equations: an overview
    """
    (d, m, f, G, y0, tspan) = _check_args(f, G, y0, tspan)
    n = len(tspan)
    dt = (tspan[n-1] - tspan[0])/(n - 1)
    # allocate space for result
    y = np.zeros((n, d), dtype=type(y0[0]))
    # pre-generate all Wiener increments (for m independent Wiener processes):
    dWs = deltaW(n - 1, m, dt)
    y[0] = y0;
    for i in range(1, n):
        t1 = tspan[i - 1]
        t2 = tspan[i]
        y1 = y[i - 1]
        dW = dWs[i - 1]
        ybar = y1 + f(y1, t1)*dt + G(y1, t1).dot(dW)
        y[i] = (y1 + 0.5*(f(y1, t1) + f(ybar, t2))*dt +
                0.5*(G(y1, t1) + G(ybar, t2)).dot(dW))
    return y
