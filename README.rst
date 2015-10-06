sdeint
======

| Numerical integration of Ito or Stratonovich SDEs.

Overview
--------
sdeint is a collection of numerical algorithms for integrating Ito and Stratonovich stochastic ordinary differential equations (SODEs). It has simple functions that can be used in a similar way to ``scipy.integrate.odeint()`` or MATLAB's ``ode45``.

There already exist some python and MATLAB packages providing Euler-Maruyama and Milstein algorithms, and a couple of others. So why am I bothering to make another package?  

It is because there has been 25 years of further research with better methods but for some reason I can't find any open source reference implementations. Not even for those methods published by Kloeden and Platen way back in 1992. So I will aim to put some improved methods here.

This is prototype code in python, so not aiming for speed. Later can always rewrite these with loops in C when speed is needed.

Bug reports are very welcome!

functions
---------

| ``itoint(f, G, y0, tspan)`` for Ito equation dy = f dt + G dW
| ``stratint(f, G, y0, tspan)`` for Stratonovich equation dy = f dt + G∘dW

These will choose an algorithm for you. Or you can use a specific algorithm directly:

specific algorithms:
--------------------
So far have these algorithms as a starting point.

| ``itoEuler(f, G, y0, tspan)``: the Euler Maruyama algorithm for Ito equations
| ``stratHeun(f, G, y0, tspan)``: the Stratonovich Heun algorithm for Stratonovich equations

Repeated integrals by the method of Kloeden, Platen and Wright (1992):

| ``Ikpw(N, h, m, n=5)`` Approximate repeated Ito integrals
| ``Jkpw(N, h, m, n=5)`` Approximate repeated Stratonovich integrals


TODO
----
- Write tests (using systems that can be solved exactly)

- Add more recent strong stochastic Runge-Kutta algorithms.
  Perhaps starting with Rößler (2010) or Burrage and Burrage (1996)

- Add an implicit strong algorithm useful for stiff equations, perhaps the one
  from Kloeden and Platen (1999) section 12.4.

- Currently prioritizing those algorithms that work for very general d-dimensional systems with arbitrary noise coefficient matrix, and which are derivative free. Eventually will add special case algorithms that give a speed increase for systems with certain symmetries. That is, 1-dimensional systems, systems with scalar noise, diagonal noise or commutative noise, etc.

- Eventually implement the main loops in C for speed.

- Some time in the dim future, implement support for stochastic delay differential equations (SDDEs).

See also:
---------

``nsim``: Framework that uses this ``sdeint`` library to do massive parallel simulations of SDE systems (using multiple CPUs or a cluster) and provides some tools to analyze the resulting timeseries. https://github.com/mattja/nsim
