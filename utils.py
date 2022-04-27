import jax
import jax.numpy as np
import functools
import operator
from functools import partial


def simpson(y, dx=1.0, axis: int = -1):
    """Simpson integrator over regularly spaced grid"""

    def basic(a):
        return 1 / 3 * np.sum(a[..., 0:-1:2] + 4 * a[..., 1::2] + a[..., 2::2], axis=-1)

    y = np.moveaxis(y, axis, -1)
    N = y.shape[-1]
    if N % 2 == 0:
        ans = 0.5 * dx * (basic(y[..., :-1]) + basic(y[..., 1:]))
        ans += 0.25 * dx * (y[..., -2] + y[..., -1] + y[..., 0] + y[..., 1])
    else:
        ans = dx * basic(y)
    return ans


def volume_avg(a):
    return np.sum(a) / functools.reduce(operator.mul, a.shape)


def l1norm(a):
    return volume_avg(np.abs(a))


def pin(a):
    return a - volume_avg(a)


def fold(scan_fun):
    def fold0(iterations):
        def fun(*carry):
            carry, trace = jax.lax.scan(scan_fun, carry, None, length=iterations)
            return carry, trace

        return fun

    return fold0


def binary_ab2pm(mu_a, mu_b):
    """Two component potential transform from \mu_a,\mu_b -> \mu_plus, \mu_minus"""
    mu_p = (mu_b + mu_a) / 2
    mu_m = (mu_b - mu_a) / 2
    return mu_p, mu_m


def binary_pm2ab(mu_p, mu_m):
    """Two component potential transform from \mu_plus,\mu_minus -> \mu_a, \mu_b"""
    mu_a = mu_p - mu_m
    mu_b = mu_p + mu_m
    return mu_a, mu_b


def grid(*, L, N):
    """Create Real/Recip grid for pseudo-spectral solvers on real data"""

    def _halfk(L, num: int):
        x = np.linspace(0.0, L, num=num, endpoint=False)
        k = np.fft.rfftfreq(num, d=x[1] - x[0]) * 2 * np.pi
        return x, k

    def _fullk(L, num: int):
        x = np.linspace(0.0, L, num=num, endpoint=False)
        k = np.fft.fftfreq(num, d=x[1] - x[0]) * 2 * np.pi
        return x, k

    def _mesh(*arrays):
        return np.stack(np.meshgrid(*arrays, indexing="ij"), axis=-1)

    ls = np.array(L)
    ns = np.array(N)
    dim = np.size(ls)
    assert dim == np.size(ns)
    if dim == 1:
        x, k = _halfk(ls, ns)
        dx, dk = x[1] - x[0], k[1] - k[0]
        print(f"x  : [{x[0]:.3f}, {x[-1]+dx:.3f}), dx={dx:.3e} ({len(x)} points)")
        print(f"k  : [{k[0]:.3f}, {k[-1]+dk:.3f}), dk={dk:.3e} ({len(k)} points)")
        return np.fft.rfft, partial(np.fft.irfft, n=ns), x, k, k**2
    elif dim == 2:
        x, kx = _fullk(ls[0], ns[0])
        y, ky = _halfk(ls[1], ns[1])
        X = _mesh(x, y, indexing="ij")
        K = _mesh(kx, ky, indexing="ij")
        return np.fft.rfftn, partial(np.fft.irfftn, s=ns), X, K, np.linalg.norm(K, axis=-1) ** 2
    elif dim == 3:
        x, kx = _fullk(ls[0], ns[0])
        y, ky = _fullk(ls[1], ns[1])
        z, kz = _halfk(ls[2], ns[2])
        X = _mesh(x, y, z, indexing="ij")
        K = _mesh(kx, ky, kz, indexing="ij")
        return np.fft.rfftn, partial(np.fft.irfftn, s=ns), X, K, np.linalg.norm(K, axis=-1) ** 2
    else:
        print("Wrong dimension")
        assert False
