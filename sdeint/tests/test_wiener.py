"""Still need to write adequate tests.
"""

import pytest
import numpy as np
from sdeint.wiener import _t, _dot, Ikpw, _vec, _unvec, _kp, _kp2, _P, _K, _a

s = np.random.randint(2**32)
print('Testing using random seed %d' % s)
np.random.seed(s)

N = 10000
h = 0.002
m = 8

X = np.random.normal(0.0, 1.0, (N, m, 1))
Y = np.random.normal(0.0, 1.0, (N, m, 1))


def test_Ikpw_identities():
    """Test the relations given in Wiktorsson2001 equation (2.1)"""
    dW, A, I = Ikpw(N, h, m)
    NIm = np.broadcast_to(np.eye(m), (N, m, m))
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*NIm))
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2*(I - A), _dot(dW, _t(dW)) - h*NIm))


def test_vec_unvec():
    A = np.arange(10*3*3).reshape((10, 3, 3))
    vecA = _vec(A)
    assert(vecA.shape == (10, 9, 1))
    assert(np.allclose(_unvec(vecA) - A, np.zeros((10, 3, 3))))


def test_kp():
    """Test our special case Kronecker tensor product function _kp() by
    comparing it against the built-in numpy function np.kron()
    """
    XkY = _kp(X, Y)
    for n in range(0, N):
        assert(np.allclose(np.kron(X[n,:,0], Y[n,:,0]), XkY[n,:,0]))


def test_kp2():
    """Test special case Kronecker tensor product function _kp2() by
    comparing it against the built-in numpy function np.kron()
    """
    for i in range(1, 10, 2):
        for j in range(1, 20, 3):
            for k in range(1, 50, 10):
                for l in range(1, 40, 7):
                    A = np.random.normal(0.0, 1.0, (10, i, j))
                    B = np.random.normal(0.0, 1.0, (10, k, l))
                    AkB = _kp2(A, B)
                    for n in range(0, 10):
                        assert(np.allclose(np.kron(A[n,:,:], B[n,:,:]),
                               AkB[n,:,:]))


def test_P():
    """Test permutation matrix _P(m)"""
    Pm0 = _P(m)
    assert(Pm0.shape == (m**2, m**2))
    assert(np.allclose(Pm0, Pm0.T)) # symmetric
    assert(np.allclose(np.dot(Pm0, Pm0), np.eye(m**2))) # is its own inverse
    Pm = np.broadcast_to(Pm0, (N, m**2, m**2))
    for n in range(0, N):
        assert(np.dot(Pm0, np.kron(X[n,:,0], Y[n,:,0])),
               np.kron(Y[n,:,0], X[n,:,0]))
    # next line is equivalent to the previous 3 lines:
    assert(_dot(Pm, _kp(X, Y)), _kp(Y, X))


def test_K():
    """Test matrix _K(m) against relations in Wiktorsson2001 equation (4.3)"""
    for q in range(2, 10):
      P0 = _P(q)
      K0 = _K(q)
      M = q*(q-1)/2
      Iq2 = np.eye(q**2)
      IM = np.eye(M)
      assert(np.allclose(np.dot(K0, K0.T), IM))
      d = []
      for k in range(1, q+1):
          d.extend([0]*k + [1]*(q-k))
      assert(np.allclose(np.dot(K0.T, K0), np.diag(d)))
      assert(np.allclose(K0.dot(P0).dot(K0.T), np.zeros((M, M))))
      assert(np.allclose(K0.dot(Iq2).dot(K0.T), IM))
      assert(np.allclose((Iq2 - P0).dot(K0.T).dot(K0).dot(Iq2 - P0), Iq2 - P0))


def test_a():
    assert(np.abs(_a(5) - 0.181322955737115) < 1e-15)


def test_Iwik_identities():
    """Test the relations given in Wiktorsson2001 equation (2.1)"""
    dW, A, I = Ikpw(N, h, m)
    NIm = np.broadcast_to(np.eye(m), (N, m, m))
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*NIm))
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2*(I - A), _dot(dW, _t(dW)) - h*NIm))
