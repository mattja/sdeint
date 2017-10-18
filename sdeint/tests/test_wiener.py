"""Still need to write tests for the statistics of offdiagonal I, J.
"""

import pytest
import numpy as np
from sdeint.wiener import (deltaW, _t, _dot, Ikpw, Jkpw, Iwik, Jwik, _vec, 
                           _unvec, _kp, _kp2, _P, _K, _a)

numpy_version = list(map(int, np.version.short_version.split('.')))
if numpy_version >= [1,10,0]:
    broadcast_to = np.broadcast_to
else:
    from sdeint._broadcast import broadcast_to


s = np.random.randint(2**32)
print('Testing using random seed %d' % s)
np.random.seed(s)

N = 10000
h = 0.002
m = 8


def test_vec_unvec():
    A = np.arange(10*3*3).reshape((10, 3, 3))
    vecA = _vec(A)
    assert(vecA.shape == (10, 9, 1))
    assert(np.allclose(_unvec(vecA) - A, np.zeros((10, 3, 3))))


def test_kp():
    """Test our special case Kronecker tensor product function _kp() by
    comparing it against the built-in numpy function np.kron()
    """
    X = np.random.normal(0.0, 1.0, (N, m, 1))
    Y = np.random.normal(0.0, 1.0, (N, m, 1))
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
    X = np.random.normal(0.0, 1.0, (N, m, 1))
    Y = np.random.normal(0.0, 1.0, (N, m, 1))
    assert(Pm0.shape == (m**2, m**2))
    assert(np.allclose(Pm0, Pm0.T)) # symmetric
    assert(np.allclose(np.dot(Pm0, Pm0), np.eye(m**2))) # is its own inverse
    Pm = broadcast_to(Pm0, (N, m**2, m**2))
    for n in range(0, N):
        assert(np.allclose(np.dot(Pm0, np.kron(X[n,:,0], Y[n,:,0])),
                           np.kron(Y[n,:,0], X[n,:,0])))
    # next line is equivalent to the previous 3 lines:
    assert(np.allclose(_dot(Pm, _kp(X, Y)), _kp(Y, X)))


def test_K():
    """Test matrix _K(m) against relations in Wiktorsson2001 equation (4.3)"""
    for q in range(2, 10):
      P0 = _P(q)
      K0 = _K(q)
      M = q*(q-1)//2
      Iqs = np.eye(q**2)
      IM = np.eye(M)
      assert(np.allclose(np.dot(K0, K0.T), IM))
      d = []
      for k in range(1, q+1):
          d.extend([0]*k + [1]*(q-k))
      assert(np.allclose(np.dot(K0.T, K0), np.diag(d)))
      assert(np.allclose(K0.dot(P0).dot(K0.T), np.zeros((M, M))))
      assert(np.allclose(K0.dot(Iqs).dot(K0.T), IM))
      assert(np.allclose((Iqs - P0).dot(K0.T).dot(K0).dot(Iqs - P0), Iqs - P0))


def test_a():
    assert(np.abs(_a(5) - 0.181322955737115) < 1e-15)


def test_Ikpw_Jkpw_identities():
    """Test the relations given in Wiktorsson2001 equation (2.1)"""
    dW = deltaW(N, m, h).reshape((N, m, 1))
    A, I = Ikpw(dW, h)
    assert(A.shape == (N, m, m) and I.shape == (N, m, m))
    Im = broadcast_to(np.eye(m), (N, m, m))
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*Im))
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2.0*(I - A), _dot(dW, _t(dW)) - h*Im))
    # and tests for Stratonovich case
    A, J = Jkpw(dW, h)
    assert(A.shape == (N, m, m) and J.shape == (N, m, m))
    assert(np.allclose(J + _t(J), _dot(dW, _t(dW))))
    assert(np.allclose(2.0*(J - A), _dot(dW, _t(dW))))


def test_Iwik_Jwik_identities():
    dW = deltaW(N, m, h).reshape((N, m, 1))
    Atilde, I = Iwik(dW, h)
    M = m*(m-1)//2
    assert(Atilde.shape == (N, M, 1) and I.shape == (N, m, m))
    Im = broadcast_to(np.eye(m), (N, m, m))
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*Im))
    # can get A from Atilde: (Wiktorsson2001 equation between (4.3) and (4.4))
    Ims = broadcast_to(np.eye(m*m), (N, m*m, m*m))
    Pm = broadcast_to(_P(m), (N, m*m, m*m))
    Km = broadcast_to(_K(m), (N, M, m*m))
    A = _unvec(_dot(_dot((Ims - Pm), _t(Km)), Atilde))
    # now can test this A against the identities of Wiktorsson eqn (2.1)
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2.0*(I - A), _dot(dW, _t(dW)) - h*Im))
    # and tests for Stratonovich case
    Atilde, J = Jwik(dW, h)
    assert(Atilde.shape == (N, M, 1) and J.shape == (N, m, m))
    assert(np.allclose(J + _t(J), _dot(dW, _t(dW))))
    A = _unvec(_dot(_dot((Ims - Pm), _t(Km)), Atilde))
    assert(np.allclose(2.0*(J - A), _dot(dW, _t(dW))))
