"""Still need to write adequate tests.
"""

import pytest
import numpy as np
import sdeint

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
    y0 = 0.0;
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.itoint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))

def test_strat_1D_additive():
    y0 = 0.0;
    f = lambda y, t: -1.0 * y
    G = lambda y, t: 0.2
    y = sdeint.stratint(f, G, y0, tspan)
    assert(np.isclose(np.mean(y), 0.0, rtol=0, atol=1e-02))
    assert(np.isclose(np.var(y), 0.2*0.2/2, rtol=1e-01, atol=0))

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
    y0 = 0.0;
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
