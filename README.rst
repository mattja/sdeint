sdeint
======
| Numerical integration of Ito or Stratonovich SDEs.

Overview
--------
sdeint is a collection of numerical algorithms for integrating Ito and Stratonovich stochastic ordinary differential equations (SODEs). It has simple functions that can be used in a similar way to ``scipy.integrate.odeint()`` or MATLAB's ``ode45``.

There already exist some python and MATLAB packages providing Euler-Maruyama and Milstein algorithms, and a couple of others. So why am I bothering to make another package?  

It is because there has been 25 years of further research with better methods but for some reason I can't find any open source reference implementations. Not even for those methods published by Kloeden and Platen way back in 1992. So I will aim to gradually add some improved methods here.

This is prototype code in python, so not aiming for speed. Later can always rewrite these with loops in C when speed is needed.

Warning: this is an early pre-release. Wait for version 1.0. Bug reports are very welcome!

functions
---------
| ``itoint(f, G, y0, tspan)`` for Ito equation dy = f(y,t)dt + G(y,t)dW
| ``stratint(f, G, y0, tspan)`` for Stratonovich equation dy = f(y,t)dt + G(y,t)∘dW

These work with scalar or vector equations. They will choose an algorithm for you. Or you can use a specific algorithm directly:

specific algorithms:
--------------------
| ``itoEuler(f, G, y0, tspan)``: the Euler-Maruyama algorithm for Ito equations.
| ``stratHeun(f, G, y0, tspan)``: the Stratonovich Heun algorithm for Stratonovich equations.
| ``itoSRI2(f, G, y0, tspan)``: the Rößler2010 order 1.0 strong Stochastic Runge-Kutta algorithm SRI2 for Ito equations.
| ``itoSRI2(f, [g1,...,gm], y0, tspan)``: as above, with G matrix given as a separate function for each column (gives speedup for large m or complicated G).
| ``stratSRS2(f, G, y0, tspan)``: the Rößler2010 order 1.0 strong Stochastic Runge-Kutta algorithm SRS2 for Stratonovich equations.
| ``stratSRS2(f, [g1,...,gm], y0, tspan)``: as above, with G matrix given as a separate function for each column (gives speedup for large m or complicated G).
| ``stratKP2iS(f, G, y0, tspan)``: the Kloeden and Platen two-step implicit order 1.0 strong algorithm for Stratonovich equations.

For more information and advanced options for controlling random value generation and repeated integrals see the documentation for each function.

utility functions:
~~~~~~~~~~~~~~~~~~
| ``deltaW(N, m, h, generator=None)``: Generate increments of m independent Wiener processes for each of N time intervals of length h. Optionally provide a `numpy.random.Generator` instance to use.

| Repeated integrals by the method of Kloeden, Platen and Wright (1992):
| ``Ikpw(dW, h, n=5, generator=None)``: Approximate repeated Ito integrals.
| ``Jkpw(dW, h, n=5, generator=None)``: Approximate repeated Stratonovich integrals.

| Repeated integrals by the method of Wiktorsson (2001):
| ``Iwik(dW, h, n=5, generator=None)``: Approximate repeated Ito integrals.
| ``Jwik(dW, h, n=5, generator=None)``: Approximate repeated Stratonovich integrals.

Examples:
---------
| Integrate the one-dimensional Ito equation |_| |eqn1|
| with initial condition ``x0 = 0.1``

.. |eqn1| image:: https://cloud.githubusercontent.com/assets/7663625/12638687/f984ae7c-c5ea-11e5-9b99-ac173d7dfe4c.png
   :alt: dx = -(a + x*b**2)*(1 - x**2)dt + b*(1 - x**2)dW
.. code-block::

    import numpy as np
    import sdeint

    a = 1.0
    b = 0.8
    tspan = np.linspace(0.0, 5.0, 5001)
    x0 = 0.1

    def f(x, t):
        return -(a + x*b**2)*(1 - x**2)

    def g(x, t):
        return b*(1 - x**2)

    result = sdeint.itoint(f, g, x0, tspan)

| Integrate the two-dimensional vector Ito equation |_| |eqn2|
| where ``x = (x1, x2)``, |_| ``dW = (dW1, dW2)`` and with initial condition ``x0 = (3.0, 3.0)``

.. |eqn2| image:: https://cloud.githubusercontent.com/assets/7663625/12638691/012a861a-c5eb-11e5-805d-d704eaff00dd.png
   :alt: dx = A.x dt + B.dW
.. code-block::

    import numpy as np
    import sdeint

    A = np.array([[-0.5, -2.0],
                  [ 2.0, -1.0]])

    B = np.diag([0.5, 0.5]) # diagonal, so independent driving Wiener processes

    tspan = np.linspace(0.0, 10.0, 10001)
    x0 = np.array([3.0, 3.0])

    def f(x, t):
        return A.dot(x)

    def G(x, t):
        return B

    result = sdeint.itoint(f, G, x0, tspan)

References for these algorithms:
--------------------------------

| ``itoEuler``: 
| G. Maruyama (1955) Continuous Markov processes and stochastic equations
| ``stratHeun``: 
| W. Rumelin (1982) Numerical Treatment of Stochastic Differential Equations
| R. Mannella (2002) Integration of Stochastic Differential Equations on a Computer
| K. Burrage, P. M. Burrage and T. Tian (2004) Numerical methods for strong solutions of stochastic differential equations: an overview
| ``itoSRI2, stratSRS2``: 
| A. Rößler (2010) Runge-Kutta Methods for the Strong Approximation of Solutions of Stochastic Differential Equations
| ``stratKP2iS``:
| P. Kloeden and E. Platen (1999) Numerical Solution of Stochastic Differential Equations, revised and updated 3rd printing
| ``Ikpw, Jkpw``:
| P. Kloeden, E. Platen and I. Wright (1992) The approximation of multiple stochastic integrals
| ``Iwik, Jwik``:
| M. Wiktorsson (2001) Joint Characteristic Function and Simultaneous Simulation of Iterated Ito Integrals for Multiple Independent Brownian Motions

TODO
----
- Fast, parallel GPU implementation in C++, wrapped with this python interface.

- Rewrite ``Iwik()`` and ``Jwik()`` so they don't waste so much memory.

- Fix ``stratKP2iS()``. In the unit tests it is currently less accurate than ``itoEuler()`` and this is likely due to a bug.

- Implement the Ito version of the Kloeden and Platen two-step implicit alogrithm.

- Add more strong stochastic Runge-Kutta algorithms. Perhaps starting with
  Burrage and Burrage (1996)

- Currently prioritizing those algorithms that work for very general d-dimensional systems with arbitrary noise coefficient matrix, and which are derivative free. Eventually will add special case algorithms that give a speed increase for systems with certain symmetries. That is, 1-dimensional systems, systems with scalar noise, diagonal noise or commutative noise, etc. The idea is that ``itoint()`` and ``stratint()`` will detect these situations and dispatch to the most suitable algorithm.

- Some time in the dim future, implement support for stochastic delay differential equations (SDDEs).

See also:
---------

``nsim``: Framework that uses this ``sdeint`` library to enable massive parallel simulations of SDE systems (using multiple CPUs or a cluster) and provides some tools to analyze the resulting timeseries. https://github.com/mattja/nsim For parallel simulation this will be obsoleted by the GPU implementation in development.

.. |_| unicode:: 0xa0
