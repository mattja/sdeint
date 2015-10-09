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
    function f is the deterministic part of the system (scalar or  dx1  vector)
    function G is the stochastic part of the system (scalar or  d x m matrix)

sdeint will choose an algorithm for you. Or you can choose one explicitly:

    itoEuler: the Euler Maruyama algorithm for Ito equations.
    stratHeun: the Stratonovich Heun algorithm for Stratonovich equations.
    itoSRI2: the Roessler 2010 order 1.0 strong Stochastic Runge-Kutta
      algorithm SRI2 for Ito equations
"""

from __future__ import absolute_import
from .wiener import deltaW, Ikpw, Iwik, Jkpw, Jwik
import numpy as np
import numbers


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
        if isinstance(f(y0_orig, tspan[0]), numbers.Number):
            f = make_vector_fn(f)
        if isinstance(G(y0_orig, tspan[0]), numbers.Number):
            G = make_matrix_fn(G)
    # determine dimension d of the system
    d = len(y0)
    if len(f(y0, tspan[0])) != d:
        raise SDEValueError('y0 and f have incompatible shapes.')
    message = """y0 has length %d. So G must either be a single function
              returning a matrix of shape (%d, m), or else a list of m separate
              functions each returning a column of G, with shape (%d,)""" % (
                  d, d, d)
    if callable(G):
        # then G must be a function returning a d x m matrix
        Gtest = G(y0, tspan[0])
        if Gtest.ndim != 2 or Gtest.shape[0] != d:
            raise SDEValueError(message)
        # determine number of independent Wiener processes m
        m = Gtest.shape[1]
    else:
        # G should be a list of m functions g_i giving columns of G
        G = tuple(G)
        m = len(G)
        Gtest = np.zeros((d, m))
        for k in range(0, m-1):
            if not callable(G[k]):
                raise SDEValueError(message)
            Gtestk = G[k](y0, tspan[0])
            if shape(Gtestk) != (d,):
                raise SDEValueError(message)
            Gtest[:,k] = Gtestk
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


def itoSRI2(f, G, y0, tspan, Imethod=Iwik):
    """Use the Roessler2010 order 1.0 strong Stochastic Runge-Kutta algorithm
    SRI2 to integrate an Ito equation dy = f(y,t)dt + G(y,t)dW(t)

    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.

    This algorithm is suitable for Ito systems with an arbitrary noise
    coefficient matrix G (i.e. the noise does not need to be scalar, diagonal,
    or commutative). The algorithm has order 2.0 convergence for the
    deterministic part alone and order 1.0 strong convergence for the complete
    stochastic system.

    Args:
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system

      G: The d x m coefficient function G can be given in two different ways:

         You can provide a single function G(y, t) that returns an array of
         shape (d, m). In this case the entire matrix G() will be evaluated
         2m+1 times at each time step so complexity grows quadratically with m.

         Alternatively you can provide a list of m functions g(y, t) each
         defining one column of G (each returning an array of shape (d,).
         In this case each g will be evaluated 3 times at each time step so
         complexity grows linearly with m. If your system has large m and
         G involves complicated functions, consider using this way.

      y0: array of shape (d,) giving the initial state vector y(t==0)

      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

      Imethod (callable, optional): which function to use to simulate repeated
        Ito integrals. Here you can choose either sdeint.Iwik (the default) or
        sdeint.Ikpw (which uses less memory in the current implementation).

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return _Roessler2010_SRK2(f, G, y0, tspan, Imethod)


def stratSRS2(f, G, y0, tspan, Jmethod=Jwik):
    """Use the Roessler2010 order 1.0 strong Stochastic Runge-Kutta algorithm
    SRS2 to integrate a Stratonovich equation dy = f(y,t)dt + G(y,t)\circ dW(t)

    where y is d-dimensional vector variable, f is a vector-valued function,
    G is a d x m matrix-valued function giving the noise coefficients and
    dW(t) is a vector of m independent Wiener increments.

    This algorithm is suitable for Stratonovich systems with an arbitrary noise
    coefficient matrix G (i.e. the noise does not need to be scalar, diagonal,
    or commutative). The algorithm has order 2.0 convergence for the
    deterministic part alone and order 1.0 strong convergence for the complete
    stochastic system.

    Args:
      f: A function f(y, t) returning an array of shape (d,)
         Vector-valued function to define the deterministic part of the system

      G: The d x m coefficient function G can be given in two different ways:

         You can provide a single function G(y, t) that returns an array of
         shape (d, m). In this case the entire matrix G() will be evaluated
         2m+1 times at each time step so complexity grows quadratically with m.

         Alternatively you can provide a list of m functions g(y, t) each
         defining one column of G (each returning an array of shape (d,).
         In this case each g will be evaluated 3 times at each time step so
         complexity grows linearly with m. If your system has large m and
         G involves complicated functions, consider using this way.

      y0: array of shape (d,) giving the initial state vector y(t==0)

      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

      Imethod (callable, optional): which function to use to simulate repeated
        Stratonovich integrals. Here you can choose either sdeint.Jwik (the
        default) or sdeint.Jkpw (which uses less memory in current version).

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    return _Roessler2010_SRK2(f, G, y0, tspan, Jmethod)


def _Roessler2010_SRK2(f, G, y0, tspan, IJmethod):
    """Implements the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
    algorithms SRI2 (for Ito equations) and SRS2 (for Stratonovich equations). 

    Algorithms SRI2 and SRS2 are almost identical and have the same extended
    Butcher tableaus. The difference is that Ito repeateded integrals I_ij are
    replaced by Stratonovich repeated integrals J_ij when integrating a
    Stratonovich equation (Theorem 6.2 in Roessler2010).

    Args:
      f: A function f(y, t) returning an array of shape (d,)
      G: Either a function G(y, t) that returns an array of shape (d, m), 
         or a list of m functions g(y, t) each returning an array shape (d,).
      y0: array of shape (d,) giving the initial state
      tspan (array): Sequence of equally spaced time points
      IJmethod (callable): which function to use to generate repeated
        integrals. N.B. for an Ito equation, must use an Ito version here
        (either Ikpw or Iwik). For a Stratonovich equation, must use a
        Stratonovich version here (Jkpw or Jwik).

    Returns:
      y: array, with shape (len(tspan), len(y0))

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    (d, m, f, G, y0, tspan) = _check_args(f, G, y0, tspan)
    have_separate_g = (not callable(G)) # if G is given as m separate functions
    N = len(tspan)
    # pre-generate all Wiener increments (for m independent Wiener processes):
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    dW = deltaW(N - 1, m, h) # shape (N, m)
    # pre-generate repeated stochastic integrals for each time step.
    # This must be I_ij for the Ito case or J_ij for the Stratonovich case:
    __, I = IJmethod(dW, h) # shape (N, m, m)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    Gn = np.zeros((d, m), dtype=y.dtype)
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        sqrth = np.sqrt(h)
        Yn = y[n] # shape (d,)
        Ik = dW[n,:] # shape (m,)
        Iij = I[n,:,:] # shape (m, m)
        fnh = f(Yn, tn)*h # shape (d,)
        if have_separate_g:
            for k in range(0, m-1):
                Gn[:,k] = G[k](Yn, tn)
        else:
            Gn = G(Yn, tn)
        sum1 = np.dot(Gn, Iij)/sqrth # shape (d, m)
        H20 = Yn + fnh # shape (d,)
        H20b = np.reshape(H20, (d, 1))
        H2 = H20b + sum1 # shape (d, m)
        H30 = Yn
        H3 = H20b - sum1
        fn1h = f(H20, tn1)*h
        Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(Gn, Ik)
        if have_separate_g:
            for k in range(0, m-1):
                Yn1 += 0.5*sqrth*(G[k](H2[:,k], tn1) - G[k](H3[:,k], tn1))
        else:
            for k in range(0, m-1):
                Yn1 += 0.5*sqrth*(G(H2[:,k], tn1)[:,k] - G(H3[:,k], tn1)[:,k])
        y[n+1] = Yn1
    return y
