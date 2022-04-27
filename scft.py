import jax
import jax.numpy as np
import phi
from utils import simpson, grid, volume_avg, l1norm, pin


def single_Debye(k2):
    """Single chain Debye function g
    g(k) = 2 (exp(-k^2) - 1 + k^2) / k^4 = 2 phi_2(-k^2) = 2 phi_2(-k^2)
    Args:
        k2 : (minus) diffusion operator in k-space
    """
    return 2 * phi.phi(-k2, n=2)


def diblock_Debye(f, k2):
    """Diblock copolymer Debye function
    g(k) = g_aa + 2 g_ab + g_bb
    Args:
        f : fraction of a-component
        k2: (minus) diffusion operator in k-space
    """
    za, zb = -f * k2, -(1 - f) * k2
    gaa = 2 * f**2 * phi.phi(za, n=2)
    gab = f * (1 - f) * phi.phi(za, n=1) * phi.phi(zb, n=1)
    gbb = 2 * (1 - f) ** 2 * phi.phi(zb, n=2)
    return gaa + 2 * gab + gbb


def partitionQ(q):
    """Canonical Partition function given single-chain propagator
    Q = 1/V \int \dd{r} q(N, r)
    """
    return volume_avg(q[-1, ...])


def mdeSplit(fft, ifft, k2, ds):
    """Expnential splitting method to solve modified diffusion equation using pseudo-spectral method
    Args:
        fft: real fft
        ifft: inverse real fft
        ds : countour step
        k2 : (minus) laplace operator in fourier space
    Returns:
        (init,apply) functions
        init : W -> Pw = exp(-ds / 2 * W)
        apply: Pw -> q_n -> q_n+1 = Pw Pd Pw q_n
        where Pw is the potential propagator in real-space, Pd the diffusion propagator in k-space
    """
    Pd = np.exp(-ds * k2)

    def initpotential(Wx):
        """Propagator for chemical potential term"""
        return np.exp(-ds / 2 * Wx)

    def evolver(Pmu, qn):
        """State evolver for Pmu -> q(s) -> q(s+ds)
        Args:
            Pmu : propagator for chemical potential term (obtained from init function)
            qn  : current state qn at countour length s
        Returns:
            updated state qn+1 at countour length s+ds
        """
        q1 = Pmu * qn
        q2 = Pd * fft(q1)
        q3 = Pmu * ifft(q2)
        return q3

    return initpotential, evolver


def diblock(*, init, evolve, chi, Na: int, Nb: int):
    """Generate solution to diblock copolymer melt scft equations
    Args:
        init   : init function for potential term in propagator
        evolve : chain propagator
        chi    : chi parameter
        Na     : number of A segments
        Nb     : number of B segments
    Returns :
        (ds, f) : countour step,  A/B fraction
        (diff,diff_c) : single chain propagator and complementary propagators, mu_a -> mu_b -> q|qc
        solution : mu_a -> mu_b -> (Q, (q,qc), (phi_a, phi_b), (dHmu_p, dHmu_m))
    """
    ds = 1.0 / (Na + Nb)
    f = Na * ds
    print(f"f  : {f:.3f} ")
    print(f"Ns : {Na} A + {Nb} B = {Na + Nb} segments ")

    def diffusion(Ni: int, Nj: int):
        """Generator for diffusion solver for either q (A->B) or qc (B->A)
        Args:
            Ni : contour segments of first part
            Nj : contour segments of second part
        Returns:
            solver function mu_i -> mu_j -> [[q(s,r)]]
        """

        def scan_fun(carry, _):
            Pmu, qn = carry
            qnew = evolve(Pmu, qn)
            return (Pmu, qnew), qnew

        def fun(mu_i, mu_j):
            """Diffusion solver for q|qc : mu_a->mu_b->[[q|qc(s,r)]]"""
            q0 = np.ones_like(mu_i)
            carry = (init(mu_i), q0)
            (_, qx), qi = jax.lax.scan(scan_fun, carry, None, length=Ni)
            carry = (init(mu_j), qx)
            _, qj = jax.lax.scan(scan_fun, carry, None, length=Nj)
            return np.vstack([q0, qi, qj])

        return fun

    diff_q = diffusion(Na, Nb)  # standard solver
    diff_qc = diffusion(Nb, Na)  # conjugate solver

    def solution(mu_a, mu_b):
        """SCFT solution for diblock copolymer melts at given mu fields mu_a and mu_b
        Args:
            mu_a : external field for A
            mu_b : external field for B
        Returns:
            Q                : partition function
            (q,qc)           : chain/conjugate chain propagator
            (phi_a, phi_b)   : local volume fractions for a,b
            (dHmu_p, dHmu_m) : thermodynamic forces (functional derivatives of H wrt \mu_\plus and \mu_\minus)
        """
        q = diff_q(mu_a, mu_b)
        qc = diff_qc(mu_b, mu_a)
        Q = partitionQ(q)
        qq = q * qc[::-1, ...]

        phi_a = simpson(qq[: Na + 1, ...], dx=ds, axis=0) / Q  # int_0^f ds q(s) qc(1-s)
        phi_b = simpson(qq[Na:, ...], dx=ds, axis=0) / Q  # int_f^1 ds q(s) qc(1-s)

        dHmu_p = phi_a + phi_b - 1
        dHmu_m = (2 * f - 1) + (mu_b - mu_a) / chi + phi_b - phi_a
        return Q, (q, qc), (phi_a, phi_b), (dHmu_p, dHmu_m)

    return (diff_q, diff_qc), solution


def binarymelt(*, init, evolve, chi, f, N: int):
    """Generate solution to binary homopolymer melt scft equations
    Args:
        init   : init function for potential term in propagator
        evolve : chain propagator
        chi    : chi parameter
        f      : fraction of A chains
        N      : number of segments (assumed same for both chains)
        L      : box size
        Ngrid  : number of real-space collocation points
    Returns :
        (ds, f) : contour step, A/B fraction
        (diff,diff_c) : single chain propagator mu_a | mu_b -> q|qc
        solution : mu_a -> mu_b -> ((Q_a,Q_b), (q_a,q_b), (phi_a, phi_b), (dFmu_p, dFmu_m))"""
    ds = 1.0 / N

    def diffusion():
        """Generator for diffusion solver for single chain in external field
        Returns:
            solver function mu -> [[q(s,r)]]
        """

        def scan_fun(carry, _):
            Pmu, qn = carry
            qnew = evolve(Pmu, qn)
            return (Pmu, qnew), qnew

        def fun(mu):
            """Diffusion solver for q : mu -> [[q(s,r)]]"""
            q0 = np.ones_like(mu)
            carry = (init(mu), q0)
            carry, qi = jax.lax.scan(scan_fun, carry, None, length=N)
            return np.vstack([q0, qi])

        return fun

    diff_q = diffusion()

    def solution(mu_a, mu_b):
        """SCFT solution for binary homopolymer melts at given mu fields mu_a and mu_b
        Args:
            mu_a : external field for A
            mu_b : external field for B
        Returns:
            (Q_a,Q_b)        : partition functions
            (q_a,q_b)        : chain propagators
            (phi_a, phi_b)   : local volume fractions for a,b
            (dFmu_p, dFmu_m) : thermodynamic forces (functional derivatives of F wrt \mu_\plus and \mu_\minus)
        """
        q_a = diff_q(mu_a)
        q_b = diff_q(mu_b)
        Q_a = partitionQ(q_a)
        Q_b = partitionQ(q_b)

        phi_a = simpson(q_a * q_a[::-1, ...], dx=ds, axis=0) * f / Q_a
        phi_b = simpson(q_b * q_b[::-1, ...], dx=ds, axis=0) * (1 - f) / Q_b
        dFmu_p = phi_a + phi_b - 1
        dFmu_m = (2 * f - 1) + (mu_b - mu_a) / chi + phi_b - phi_a
        return (Q_a, Q_b), (q_a, q_b), (phi_a, phi_b), (dFmu_p, dFmu_m)

    return diff_q, solution
