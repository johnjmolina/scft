from utils import binary_pm2ab, pin

"""Relaxation routines to find \mu* saddle point solutions
    Euler : simple explicit euler 
    SIS   : semi-implicit seidel method
"""


def binary_euler(*, solver, dt, trj):
    """
    Args:
        solver : mu_a -> mu_b -> (Q, q, phi, (dmu_p, dmu_m))
        dt     : time step
        trj    : trj snapshot function mu_a -> mu_b -> trj_value
    Returns:
        Suitable scan function that can be fed to lax.scan mu_p -> mu_m -> (Q, q, phi, (dmu_p, dmu_m))
    Note:
        Only thermodynamic forces are used
    """

    def scan_fun(carry, _):
        mu_p, mu_m = carry
        _, _, _, (dmu_p, dmu_m) = solver(*binary_pm2ab(mu_p, mu_m))

        mu_p = mu_p + dt * dmu_p
        mu_m = mu_m - dt * dmu_m

        mu_p = pin(mu_p)
        mu_m = pin(mu_m)

        return (mu_p, mu_m), trj(dmu_p, dmu_m)

    return scan_fun


def binary_sis(*, fft, ifft, Gs, solver, dt, trj):
    """
    Args:
        fft,ifft: fourier transform pair
        Gs     : Debye factors for mu_plus, mu_minus updates
        solver : mu_a -> mu_b -> (Q, q, phi, (dmu_p, dmu_m))
        dt     : time step
        trj    : trj snapshot function mu_a -> mu_b -> trj_value
    Returns:
        Suitable scan function that can be fed to lax.scan mu_p -> mu_m -> (Q, q, phi,(dmu_p,dmu_m))
    Note:
        Only thermodynamic forces are used
    """

    def scan_fun(carry, _):
        mu_p, mu_m = carry
        _, _, _, (dmu_p, _) = solver(*binary_pm2ab(mu_p, mu_m))

        dmu_p = fft(dmu_p)
        mu_p = mu_p + dt * ifft(dmu_p / (1 + dt * Gs[0]))
        _, _, _, (dmu_p, dmu_m) = solver(*binary_pm2ab(mu_p, mu_m))
        mu_m = mu_m - dt * dmu_m / (1 + dt * Gs[1])

        mu_p = pin(mu_p)
        mu_m = pin(mu_m)

        return (mu_p, mu_m), trj(dmu_p, dmu_m)

    return scan_fun
