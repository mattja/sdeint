"""Still need to write adequate tests.
"""

import pytest
import numpy as np
from sdeint.wiener import _t, _dot, Ikpw

s = np.random.randint(2**32)
print('Testing using random seed %d' % s)
np.random.seed(s)


def test_Ikpw_identities():
    """Test the relations given in Wiktorsson2001 equation (2.1)"""
    N = 10000
    h = 0.002
    m = 8
    dW, A, I = Ikpw(N, h, m)
    NIm = np.broadcast_to(np.eye(m), (N, m, m))
    assert(np.allclose(I + _t(I), _dot(dW, _t(dW)) - h*NIm))
    assert(np.allclose(A, -_t(A)))
    assert(np.allclose(2*(I - A), _dot(dW, _t(dW)) - h*NIm))
