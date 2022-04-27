import jax.numpy as np

Coeff = np.array(
    [
        1,
        1 / 2,
        1 / 6,
        1 / 24,
        1 / 120,
        1 / 720,
        1 / 5040,
        1 / 40320,
        1 / 362880,
        1 / 3628800,
        1 / 39916800,
        1 / 479001600,
        1 / 6227020800,
        1 / 87178291200,
        1 / 1307674368000,
        1 / 20922789888000,
        1 / 355687428096000,
        1 / 6402373705728000,
        1 / 121645100408832000,
        1 / 2432902008176640000,
        1 / 51090942171709440000,
    ]
)


def phi1(z):
    ans = np.zeros_like(z)
    z2 = z * z
    z3 = z2 * z
    large = np.abs(z) > 4e-2
    small = np.logical_not(large)

    ans = ans.at[large].set(np.expm1(z[large]) / z[large])
    ans = ans.at[small].set(
        Coeff[0]
        + Coeff[1] * z[small]
        + Coeff[2] * z2[small]
        + Coeff[3] * z3[small]
        + Coeff[4] * z2[small] ** 2
        + Coeff[5] * z2[small] * z3[small]
        + Coeff[6] * z3[small] ** 2
    )
    return ans


def phi2(z):
    ans = np.zeros_like(z)
    z2 = z * z
    z3 = z2 * z
    z5 = z2 * z3
    z7 = z5 * z2
    z8 = z7 * z
    small = np.abs(z) < 0.1
    large = np.abs(z) > 1.0
    medium = np.logical_not(np.logical_or(small, large))

    ans = ans.at[large].set((np.expm1(z[large]) - z[large]) / z2[large])
    ans = ans.at[medium].set(
        Coeff[1]
        + z[medium] * Coeff[2]
        + z2[medium] * Coeff[3]
        + z3[medium] * Coeff[4]
        + z2[medium] * z2[medium] * Coeff[5]
        + z5[medium] * Coeff[6]
        + z3[medium] * z3[medium] * Coeff[7]
        + z7[medium] * Coeff[8]
        + z8[medium] * Coeff[9]
        + z8[medium] * z[medium] * Coeff[10]
        + z5[medium] * z5[medium] * Coeff[11]
        + z8[medium] * z3[medium] * Coeff[12]
        + z7[medium] * z5[medium] * Coeff[13]
        + z8[medium] * z5[medium] * Coeff[14]
        + z7[medium] * z7[medium] * Coeff[15]
        + z8[medium] * z7[medium] * Coeff[16]
        + z8[medium] * z8[medium] * Coeff[17]
    )
    ans = ans.at[small].set(
        Coeff[1]
        + z[small] * Coeff[2]
        + z2[small] * Coeff[3]
        + z3[small] * Coeff[4]
        + z2[small] * z2[small] * Coeff[5]
        + z5[small] * Coeff[6]
        + z3[small] * z3[small] * Coeff[7]
        + z7[small] * Coeff[8]
        + z8[small] * Coeff[9]
    )
    return ans


def phi3(z):
    ans = np.zeros_like(z)
    z2 = z * z
    z3 = z2 * z
    z5 = z2 * z3
    z7 = z5 * z2
    z8 = z7 * z
    z16 = z8 * z8
    small = np.abs(z) < 0.1
    large = np.abs(z) > 1.6
    medium = np.logical_not(np.logical_or(small, large))

    ans = ans.at[large].set((np.expm1(z[large]) - 0.5 * z2[large] - z[large]) / z3[large])
    ans = ans.at[medium].set(
        Coeff[2]
        + z[medium] * Coeff[3]
        + z2[medium] * Coeff[4]
        + z3[medium] * Coeff[5]
        + z2[medium] * z2[medium] * Coeff[6]
        + z5[medium] * Coeff[7]
        + z3[medium] * z3[medium] * Coeff[8]
        + z7[medium] * Coeff[9]
        + z8[medium] * Coeff[10]
        + z8[medium] * z[medium] * Coeff[11]
        + z5[medium] * z5[medium] * Coeff[12]
        + z8[medium] * z3[medium] * Coeff[13]
        + z7[medium] * z5[medium] * Coeff[14]
        + z8[medium] * z5[medium] * Coeff[15]
        + z7[medium] * z7[medium] * Coeff[16]
        + z8[medium] * z7[medium] * Coeff[17]
        + z16[medium] * Coeff[18]
        + z16[medium] * z[medium] * Coeff[19]
        + z16[medium] * z2[medium] * Coeff[20]
    )
    ans = ans.at[small].set(
        Coeff[2]
        + z[small] * Coeff[3]
        + z2[small] * Coeff[4]
        + z3[small] * Coeff[5]
        + z2[small] * z2[small] * Coeff[6]
        + z5[small] * Coeff[7]
        + z3[small] * z3[small] * Coeff[8]
        + z7[small] * Coeff[9]
        + z8[small] * Coeff[10]
    )
    return ans


def phi(z, *, n: int = 1):
    assert n > 0 and n < 4
    if n == 1:
        return phi1(z)
    elif n == 2:
        return phi2(z)
    else:
        return phi3(z)
