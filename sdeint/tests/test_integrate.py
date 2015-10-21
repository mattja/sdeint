"""Still need a multivariate, multi-noise test against an exactly solvable
system. (I suppose it will have to be linear)
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
def test_exact_KP44E(h=0.002):
    """exactly solvable test system given in Kloeden&Platen ch 4.4 eqn (4.5)"""
    tspan = np.arange(0.0, 30.0, h)
    N = len(tspan)
    y0 = 1.0
    sqrt2 = np.sqrt(2.0)
    def f(y, t):
        return -np.sin(2*y) - 0.25*np.sin(4*y) # Ito version
    def G(y, t):
        return sqrt2*np.cos(y)**2
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
    y[0] = 1.0
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
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return -np.sin(2*y) - 0.25*np.sin(4*y) + 2.0*np.sin(y)*np.cos(y)**3
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
    # plot the errors:
    errEuler = np.abs(yEuler - y)
    errSRI2 = np.abs(ySRI2 - y)
    errHeun = np.abs(yHeun - y)
    errSRS2 = np.abs(ySRS2 - y)
    errKP2iS = np.abs(yKP2iS - y)
    fig1 = plt.figure()
    plt.plot(tspan, errEuler, 'b--', tspan, errHeun, 'g--', tspan,errSRI2,'r:',
             tspan, errSRS2, 'm:', tspan, errKP2iS, 'c--')
    plt.title('absolute error, with delta_t=%g s' % h)
    plt.xlabel('time (s)')
    plt.legend(['itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2', 'stratKP2iS'])
    fig1.show()
    return


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
