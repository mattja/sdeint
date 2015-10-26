"""Still looking for a multivariate, multi-noise test case against an exactly
solvable system.

Refs:
    P. Kloeden, E. Platen (1999) Numerical Solution of Stochastic Differential
      Equations

    P. Kloeden, E. Platen and H. Schurz (2003) Numerical Solution of SDE
      Through Computer Experiments
"""

import pytest
import numpy as np
import sdeint
from scipy import stats
from matplotlib import pyplot as plt

s = np.random.randint(2**32)
print('Testing using random seed %d' % s)
np.random.seed(s)

tspan = np.arange(0.0, 2000.0, 0.002)

def test_mismatched_f():
    y0 = np.zeros(3)
    f = lambda y, t: np.array([1.0, 2.0, 3.0, 4.0])
    G = lambda y, t: np.ones((3, 3))
    with pytest.raises(sdeint.SDEValueError):
        y = sdeint.itoint(f, G, y0, tspan)


def test_ito_1D_additive():
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.itoint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


def test_strat_1D_additive():
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.stratint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


# now for some proper tests
def test_exact_KP4446(h=0.002):
    """exactly solvable test system from Kloeden & Platen ch 4.4 eqn (4.46)"""
    tspan = np.arange(0.0, 20.0, h)
    N = len(tspan)
    y0 = 0.1
    a = 1.0
    b = 0.80
    def f(y, t):
        return -(a + y*b**2)*(1 - y**2)
    def G(y, t):
        return b*(1 - y**2)
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return -a*(1 - y**2)
    dW = sdeint.deltaW(N-1, 1, h)
    # compute exact solution y for that sample path:
    y = np.zeros(N)
    y[0] = y0
    Wn = 0.0
    for n in range(1, N):
        tn = tspan[n]
        Wn += dW[n-1]
        y[n] = (((1 + y0)*np.exp(-2.0*a*tn + 2.0*b*Wn) + y0 - 1.0)/
                ((1 + y0)*np.exp(-2.0*a*tn + 2.0*b*Wn) + 1.0 - y0))
    # now compute approximate solution with our Ito algorithms:
    # Wiener increments and repeated Ito integrals for that same sample path:
    __, I = sdeint.Iwik(dW, h)
    ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
    yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
    # also test our Stratonovich algorithms
    J = I + 0.5*h  # since m==1 for this test example
    ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
    # plot the exact and approximated paths:
    fig0 = plt.figure()
    plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
             tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
    plt.title('sample paths, with delta_t = %g s' % h)
    plt.xlabel('time (s)')
    plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                'stratKP2iS'])
    fig0.show()
    errEuler = np.abs(yEuler - y)
    errSRI2 = np.abs(ySRI2 - y)
    errHeun = np.abs(yHeun - y)
    errSRS2 = np.abs(ySRS2 - y)
    errKP2iS = np.abs(yKP2iS - y)


def test_exact_KP4459(h=0.002):
    """exactly solvable test system from Kloeden & Platen ch 4.4 eqn (4.59)"""
    tspan = np.arange(0.0, 5.0, h)
    N = len(tspan)
    y0 = np.array([1.0])
    a = -0.02
    b1 = 1.0
    b2 = 2.0
    def f(y, t):
        return np.array([a*y[0]])
    def G(y, t):
        return np.array([[b1*y[0], b2*y[0]]])
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return np.array([(a - (b1**2 + b2**2)/2.0)*y[0]])
    dW = sdeint.deltaW(N-1, 2, h)
    # compute exact solution y for that sample path:
    y = np.zeros(N)
    y[0] = y0
    Wn = np.array([0.0, 0.0])
    for n in range(1, N):
        tn = tspan[n]
        Wn += dW[n-1,:]
        y[n] = y0*np.exp((a - (b1**2 + b2**2)/2.0)*tn + b1*Wn[0] + b2*Wn[1])
    # now compute approximate solution with our Ito algorithms:
    # Wiener increments and repeated Ito integrals for that same sample path:
    __, I = sdeint.Iwik(dW, h)
    ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
    yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
    # also test our Stratonovich algorithms
    J = I + 0.5*h  # since m==1 for this test example
    ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
    # plot the exact and approximated paths:
    fig0 = plt.figure()
    plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
             tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
    plt.title('sample paths, with delta_t = %g s' % h)
    plt.xlabel('time (s)')
    plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                'stratKP2iS'])
    fig0.show()
    errEuler = np.abs(yEuler - y)
    errSRI2 = np.abs(ySRI2 - y)
    errHeun = np.abs(yHeun - y)
    errSRS2 = np.abs(ySRS2 - y)
    errKP2iS = np.abs(yKP2iS - y)


def test_exact_KPS445(h=0.002):
    """exactly solvable test system Kloeden,Platen&Schurz ch 4.4 eqn (4.5)"""
    tspan = np.arange(0.0, 30.0, h)
    N = len(tspan)
    y0 = 1.0
    sqrt2 = np.sqrt(2.0)
    def f(y, t):
        return -np.sin(2*y) - 0.25*np.sin(4*y) # Ito version
    def G(y, t):
        return sqrt2*np.cos(y)**2
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return -np.sin(2*y) - 0.25*np.sin(4*y) + 2.0*np.sin(y)*np.cos(y)**3
    eq2 = (np.exp(2.0*h) - 1)/2.0 - 2.0*h*np.exp(h) + h + h**2 + h**3/3.0
    ewq = np.exp(h) - (1 + h + h**2/2.0)
    ezq = np.exp(h) - (1 + h + h**2/2.0 + h**3/6.0)
    covar = np.array([[h,        h**2/2.0, ewq],
                      [h**2/2.0, h**3/3.0, ezq],
                      [ewq     , ezq     , eq2]])
    distribution = stats.multivariate_normal(mean=np.zeros(3), cov=covar,
                                             allow_singular=True)
    WZQ = distribution.rvs(size=N-1) # shape (N-1, 3)
    # compute exact solution y for that sample path:
    y = np.zeros(N)
    y[0] = y0
    etvn = np.tan(1.0) # \exp{T} V_T
    for n in range(1, N):
        tn = tspan[n]
        deltaW = WZQ[n-1,0]
        deltaZ = WZQ[n-1,1]
        deltaQ = WZQ[n-1,2]
        etvn += sqrt2*np.exp((n-1)*h)*(deltaW*(1+h) + deltaZ + deltaQ)
        y[n] = np.arctan(np.exp(-n*h)*etvn)
    # now compute approximate solution with our Ito algorithms:
    # Wiener increments and repeated Ito integrals for that same sample path:
    dW = WZQ[:,0:1]
    __, I = sdeint.Iwik(dW, h)
    ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
    yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
    # also test our Stratonovich algorithms
    J = I + 0.5*h  # since m==1 for this test example
    ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
    yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
    # plot the exact and approximated paths:
    fig0 = plt.figure()
    plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
             tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
    plt.title('sample paths, with delta_t = %g s' % h)
    plt.xlabel('time (s)')
    plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                'stratKP2iS'])
    fig0.show()
    errEuler = np.abs(yEuler - y)
    errSRI2 = np.abs(ySRI2 - y)
    errHeun = np.abs(yHeun - y)
    errSRS2 = np.abs(ySRS2 - y)
    errKP2iS = np.abs(yKP2iS - y)


def test_strat_ND_additive():
    y0 = np.zeros(3)
    def f(y, t):
        return np.array([ -1.0*y[0],
                          y[2],
                          -1.0*y[1] - 0.4*y[2] ])
    def G(y, t):
        return np.diag([0.2, 0.0, 0.5])
    y = sdeint.stratint(f, G, y0, tspan)
    w = np.fft.rfft(y[:, 2])
    # TODO assert spectral peak is around 1.0 radians/s


def test_itoEuler_1D_additive():
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.itoEuler(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


def test_stratHeun_ND_additive():
    y0 = np.zeros(3)
    def f(y, t):
        return np.array([ -1.0*y[0],
                          y[2],
                          -1.0*y[1] - 0.4*y[2] ])
    def G(y, t):
        return np.diag([0.2, 0.0, 0.5])
    y = sdeint.stratHeun(f, G, y0, tspan)
    w = np.fft.rfft(y[:, 2])
    # TODO assert spectral peak is around 1.0 radians/s


def test_stratKP2iS_1D_additive():
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.stratKP2iS(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))
