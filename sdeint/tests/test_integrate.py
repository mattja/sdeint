"""Tests for the numerical integration algorithms.

Refs:
    P. Kloeden, E. Platen (1999) Numerical Solution of Stochastic Differential
      Equations

    P. Kloeden, E. Platen and H. Schurz (2003) Numerical Solution of SDE
      Through Computer Experiments

    A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
      Solutions of Stochastic Differential Equations
"""

import pytest
import numpy as np
import sdeint
from scipy import stats, linalg

s = np.random.randint(2**32)
np.random.seed(s)

def pytest_report_header(config):
    return 'Testing using random seed %d' % s


def test_mismatched_f():
    tspan = np.arange(0.0, 2000.0, 0.002)
    y0 = np.zeros(3)
    f = lambda y, t: np.array([1.0, 2.0, 3.0, 4.0])
    G = lambda y, t: np.ones((3, 3))
    with pytest.raises(sdeint.SDEValueError):
        y = sdeint.itoint(f, G, y0, tspan)


def test_ito_1D_additive():
    tspan = np.arange(0.0, 2000.0, 0.002)
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.itoint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


def test_strat_1D_additive():
    tspan = np.arange(0.0, 2000.0, 0.002)
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.stratint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


# Now for some proper tests: comparing against exactly solvable systems

def _assert_close(approx_sol, exact_sol, relTol=1e-2, absTol=1e-2):
    """Pass the test if at ALL simulated time points the approximated solution
    from the integration algorithm matches the known exact solution to within
    either the relative tolerance or absolute tolerance."""
    if exact_sol.ndim > 1:
        absError = np.linalg.norm(approx_sol - exact_sol, axis=1)
        relError = absError/np.linalg.norm(exact_sol, axis=1)
    else:
        absError = np.abs(approx_sol - exact_sol)
        relError = absError/np.abs(exact_sol)
    ok = (absError <= absTol) | (relError <= relTol)
    if np.alltrue(ok):
        return
    else:
        ind = np.nonzero(~ok)[0][0]
        msg = """Sample path approximation not close enough at time step n==%d:
              exact_sol==%s, approx_sol==%s, relError==%g, absError==%g""" % (
                      ind, exact_sol[ind], approx_sol[ind], relError[ind],
                      absError[ind])
        raise AssertionError(msg)


@pytest.fixture(scope='module')
def exact_solution_KP4446():
    """Exactly solvable test system from Kloeden & Platen ch 4.4 eqn (4.46)
    By making this a fixture, the exact solution can be re-used in several
    tests without re-calculating it.
    """
    h = 0.0004
    tspan = np.arange(0.0, 10.0, h)
    N = len(tspan)
    y0 = 0.1
    a = 1.0
    b = 0.80
    # Ito version:
    def f(y, t):
        return -(a + y*b**2)*(1 - y**2)
    def G(y, t):
        return b*(1 - y**2)
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return -a*(1 - y**2)
    # Generate Wiener increments and repeated integrals for one sample path:
    dW = sdeint.deltaW(N-1, 1, h)
    __, I = sdeint.Iwik(dW, h) # Ito repeated integrals
    J = I + 0.5*h  # Stratonovich repeated integrals (m==1)
    # Compute exact solution y for that sample path:
    y = np.zeros(N)
    y[0] = y0
    Wn = 0.0
    for n in range(1, N):
        tn = tspan[n]
        Wn += dW[n-1]
        y[n] = (((1 + y0)*np.exp(-2.0*a*tn + 2.0*b*Wn) + y0 - 1.0)/
                ((1 + y0)*np.exp(-2.0*a*tn + 2.0*b*Wn) + 1.0 - y0))
    return (dW, I, J, f, f_strat, G, y0, tspan, y)


class Test_KP4446:
    def test_itoEuler_KP4446(self, exact_solution_KP4446):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4446
        yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yEuler, y, 1e-1, 1e-1)
        return yEuler

    def test_itoSRI2_KP4446(self, exact_solution_KP4446):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4446
        ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
        _assert_close(ySRI2, y, 1e-2, 1e-2)
        return ySRI2

    def test_stratHeun_KP4446(self, exact_solution_KP4446):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4446
        yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yHeun, y, 1e-2, 1e-2)
        return yHeun

    def test_stratSRS2_KP4446(self, exact_solution_KP4446):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4446
        ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(ySRS2, y, 1e-2, 1e-2)
        return ySRS2

    def test_stratKP2iS_KP4446(self, exact_solution_KP4446):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4446
        yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(yKP2iS, y, 1e-1, 1e-1)
        return yKP2iS

    def plot(self):
        from matplotlib import pyplot as plt
        es = exact_solution_KP4446()
        (dW, I, J, f, f_strat, G, y0, tspan, y) = es
        yEuler = self.test_itoEuler_KP4446(es)
        ySRI2 = self.test_itoSRI2_KP4446(es)
        yHeun = self.test_stratHeun_KP4446(es)
        ySRS2 = self.test_stratSRS2_KP4446(es)
        yKP2iS = self.test_stratKP2iS_KP4446(es)
        # plot the exact and approximated paths:
        fig0 = plt.figure()
        h = (tspan[len(tspan)-1] - tspan[0])/(len(tspan) - 1)
        plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
                 tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
        plt.title('sample paths for test KP4446, delta_t = %g s' % h)
        plt.xlabel('time (s)')
        plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                    'stratKP2iS'])
        fig0.show()


@pytest.fixture(scope='module')
def exact_solution_KP4459():
    """Exactly solvable test system from Kloeden & Platen ch 4.4 eqn (4.59)"""
    h = 0.0004
    tspan = np.arange(0.0, 5.0, h)
    N = len(tspan)
    y0 = np.array([1.0])
    a = -0.02
    b1 = 1.0
    b2 = 2.0
    # Ito version:
    def f(y, t):
        return np.array([a*y[0]])
    def G(y, t):
        return np.array([[b1*y[0], b2*y[0]]])
    # Stratonovich version of the same equation:
    def f_strat(y, t):
        return np.array([(a - (b1**2 + b2**2)/2.0)*y[0]])
    # Generate Wiener increments and repeated integrals for one sample path:
    dW = sdeint.deltaW(N-1, 2, h)
    __, I = sdeint.Iwik(dW, h)
    J = I + 0.5*h*np.eye(2).reshape((1, 2, 2)) # since m==2
    # compute exact solution y for that sample path:
    y = np.zeros(N)
    y[0] = y0
    Wn = np.array([0.0, 0.0])
    for n in range(1, N):
        tn = tspan[n]
        Wn += dW[n-1,:]
        y[n] = y0*np.exp((a - (b1**2 + b2**2)/2.0)*tn + b1*Wn[0] + b2*Wn[1])
    return (dW, I, J, f, f_strat, G, y0, tspan, y)


class Test_KP4459:
    def test_itoEuler_KP4459(self, exact_solution_KP4459):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4459
        yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yEuler, y, 1e-1, 1e-1)
        return yEuler

    def test_itoSRI2_KP4459(self, exact_solution_KP4459):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4459
        ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
        _assert_close(ySRI2, y, 1e-2, 1e-2)
        return ySRI2

    def test_stratHeun_KP4459(self, exact_solution_KP4459):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4459
        yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yHeun, y, 1e-2, 1e-2)
        return yHeun

    def test_stratSRS2_KP4459(self, exact_solution_KP4459):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4459
        ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(ySRS2, y, 1e-2, 1e-2)
        return ySRS2

    def test_stratKP2iS_KP4459(self, exact_solution_KP4459):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KP4459
        yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(yKP2iS, y, 1e-1, 1e-1)
        return yKP2iS

    def plot(self):
        from matplotlib import pyplot as plt
        es = exact_solution_KP4459()
        (dW, I, J, f, f_strat, G, y0, tspan, y) = es
        yEuler = self.test_itoEuler_KP4459(es)
        ySRI2 = self.test_itoSRI2_KP4459(es)
        yHeun = self.test_stratHeun_KP4459(es)
        ySRS2 = self.test_stratSRS2_KP4459(es)
        yKP2iS = self.test_stratKP2iS_KP4459(es)
        # plot the exact and approximated paths:
        fig0 = plt.figure()
        h = (tspan[len(tspan)-1] - tspan[0])/(len(tspan) - 1)
        plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
                 tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
        plt.title('sample paths for test KP4459, delta_t = %g s' % h)
        plt.xlabel('time (s)')
        plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                    'stratKP2iS'])
        fig0.show()


@pytest.fixture(scope='module')
def exact_solution_KPS445():
    """exactly solvable test system Kloeden,Platen&Schurz ch 4.4 eqn (4.5)"""
    h = 0.0004
    tspan = np.arange(0.0, 30.0, h)
    N = len(tspan)
    y0 = 1.0
    sqrt2 = np.sqrt(2.0)
    # Ito version:
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
    # Wiener increments and repeated integrals for that sample path:
    dW = WZQ[:,0:1]
    __, I = sdeint.Iwik(dW, h)
    J = I + 0.5*h  # since m==1
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
    return (dW, I, J, f, f_strat, G, y0, tspan, y)


class Test_KPS445:
    def test_itoEuler_KPS445(self, exact_solution_KPS445):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KPS445
        yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yEuler, y, 1e-1, 1e-1)
        return yEuler

    def test_itoSRI2_KPS445(self, exact_solution_KPS445):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KPS445
        ySRI2 = sdeint.itoSRI2(f, G, y0, tspan, dW=dW, I=I)[:,0]
        _assert_close(ySRI2, y, 1e-2, 1e-2)
        return ySRI2

    def test_stratHeun_KPS445(self, exact_solution_KPS445):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KPS445
        yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)[:,0]
        _assert_close(yHeun, y, 1e-2, 1e-2)
        return yHeun

    def test_stratSRS2_KPS445(self, exact_solution_KPS445):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KPS445
        ySRS2 = sdeint.stratSRS2(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(ySRS2, y, 1e-2, 1e-2)
        return ySRS2

    def test_stratKP2iS_KPS445(self, exact_solution_KPS445):
        (dW, I, J, f, f_strat, G, y0, tspan, y) = exact_solution_KPS445
        yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)[:,0]
        _assert_close(yKP2iS, y, 1e-1, 1e-1)
        return yKP2iS

    def plot(self):
        from matplotlib import pyplot as plt
        es = exact_solution_KPS445()
        (dW, I, J, f, f_strat, G, y0, tspan, y) = es
        yEuler = self.test_itoEuler_KPS445(es)
        ySRI2 = self.test_itoSRI2_KPS445(es)
        yHeun = self.test_stratHeun_KPS445(es)
        ySRS2 = self.test_stratSRS2_KPS445(es)
        yKP2iS = self.test_stratKP2iS_KPS445(es)
        # plot the exact and approximated paths:
        fig0 = plt.figure()
        h = (tspan[len(tspan)-1] - tspan[0])/(len(tspan) - 1)
        plt.plot(tspan, y, 'k-', tspan, yEuler, 'b--', tspan, yHeun, 'g--',
                 tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
        plt.title('sample paths for test KPS445, delta_t = %g s' % h)
        plt.xlabel('time (s)')
        plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                    'stratKP2iS'])
        fig0.show()


@pytest.fixture(scope='module')
def exact_solution_R74():
    """exactly solvable test system from Roessler2010 eqn (7.4)"""
    d = 5
    m = 4
    h = 0.0004
    tspan = np.arange(0.0, 10.0, h)
    N = len(tspan)
    A = np.ones((d, d))*0.05
    B0 = np.ones((d, d))*0.01
    for i in range(0, d):
        A[i, i] = -1.5
        B0[i, i] = 0.2
    B = np.vstack([B0[np.newaxis,...]]*m)
    y0 = np.ones(d)
    # Ito version:
    def f(y, t):
        return np.dot(A, y)
    def G(y, t):
        return np.dot(B, y).T
    G_separate = [lambda y, t: np.dot(B0, y)] * m
    # Stratonovich version of the same equation:
    S = np.ones((d, d))*-0.0086
    for i in range(0, d):
        S[i, i] = -0.0808
    def f_strat(y, t):
        return np.dot(A + S, y)
    # Generate Wiener increments and repeated integrals for one sample path:
    dW = sdeint.deltaW(N-1, m, h) # shape (N-1, m)
    __, I = sdeint.Iwik(dW, h)
    J = I + 0.5*h*np.eye(m).reshape((1, m, m))
    # compute exact solution y for that sample path:
    y = np.zeros((N, d))
    y[0] = y0
    Wn = np.zeros(m)
    term1 = A - 0.5*np.einsum('kij,kjl', B, B)
    for n in range(1, N):
        tn = tspan[n]
        Wn += dW[n-1]
        term2 = np.einsum('kij,k', B, Wn)
        y[n] = linalg.expm(term1*tn + term2).dot(y0)
    return (dW, I, J, f, f_strat, G, G_separate, y0, tspan, y)


class Test_R74:
    def test_itoEuler_R74(self, exact_solution_R74):
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan,y) = exact_solution_R74
        yEuler = sdeint.itoEuler(f, G, y0, tspan, dW=dW)
        _assert_close(yEuler, y, 1e-1, 1e-1)
        return yEuler

    def test_itoSRI2_R74(self, exact_solution_R74):
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan,y) = exact_solution_R74
        ySRI2 = sdeint.itoSRI2(f, G_separate, y0, tspan, dW=dW, I=I)
        _assert_close(ySRI2, y, 1e-2, 1e-2)
        return ySRI2

    def test_stratHeun_R74(self, exact_solution_R74):
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan,y) = exact_solution_R74
        yHeun = sdeint.stratHeun(f_strat, G, y0, tspan, dW=dW)
        _assert_close(yHeun, y, 1e-2, 1e-2)
        return yHeun

    def test_stratSRS2_R74(self, exact_solution_R74):
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan,y) = exact_solution_R74
        ySRS2 = sdeint.stratSRS2(f_strat, G_separate, y0, tspan, dW=dW, J=J)
        _assert_close(ySRS2, y, 1e-2, 1e-2)
        return ySRS2

    def test_stratKP2iS_R74(self, exact_solution_R74):
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan,y) = exact_solution_R74
        yKP2iS = sdeint.stratKP2iS(f_strat, G, y0, tspan, dW=dW, J=J)
        _assert_close(yKP2iS, y, 1e-1, 1e-1)
        return yKP2iS

    def plot(self):
        from matplotlib import pyplot as plt
        es = exact_solution_R74()
        (dW, I, J, f, f_strat, G, G_separate, y0, tspan, y) = es
        yEuler = self.test_itoEuler_R74(es)[:,0]
        ySRI2 = self.test_itoSRI2_R74(es)[:,0]
        yHeun = self.test_stratHeun_R74(es)[:,0]
        ySRS2 = self.test_stratSRS2_R74(es)[:,0]
        yKP2iS = self.test_stratKP2iS_R74(es)[:,0]
        # plot (the first component of) the exact and approximated paths:
        fig0 = plt.figure()
        h = (tspan[len(tspan)-1] - tspan[0])/(len(tspan) - 1)
        plt.plot(tspan, y[:,0], 'k-', tspan, yEuler, 'b--', tspan, yHeun,'g--',
                 tspan, ySRI2,'r:', tspan, ySRS2, 'm:', tspan, yKP2iS, 'c--')
        plt.title('sample paths component 0 for test R74, delta_t = %g s' % h)
        plt.xlabel('time (s)')
        plt.legend(['exact', 'itoEuler', 'stratHeun', 'itoSRI2', 'stratSRS2',
                    'stratKP2iS'])
        fig0.show()


def test_strat_ND_additive():
    tspan = np.arange(0.0, 2000.0, 0.002)
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
    tspan = np.arange(0.0, 2000.0, 0.002)
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.itoEuler(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))


def test_stratHeun_ND_additive():
    tspan = np.arange(0.0, 2000.0, 0.002)
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
    tspan = np.arange(0.0, 2000.0, 0.002)
    y0 = 0.0
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.stratKP2iS(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))
