"""Tests for pyOPEdefs against exactly-known CFT results."""
import sympy as sp
from sympy import Rational

import opedefs as od
from opedefs import (One, OPE, OPEPole, NO, d, OPEData, OPEJacobi,
                     jacobi_satisfied, bosonic, fermionic, define_ope,
                     set_ope_options, reset, QUANTUM, CLASSICAL)


def opdata(*poles_desc):
    """Build an OPEData from poles listed highest-first (0 allowed)."""
    conv = []
    for p in poles_desc:
        conv.append(p if isinstance(p, od.Op) else One * sp.sympify(p))
    return OPEData(list(reversed(conv)))


# ----------------------------------------------------------------------
def test_virasoro_basics():
    reset()
    c = sp.Symbol('c')
    T = bosonic('T')
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])

    tt = OPE(T, T)
    assert tt.max_pole == 4
    assert tt.pole(4) == c / 2 * One
    assert tt.pole(3) == od.Op()
    assert tt.pole(2) == 2 * T
    assert tt.pole(1) == d(T)

    # OPE with derivative arguments: T(z)T'(w) = d/dw of the TT OPE:
    #   2c/(z-w)^5 + 4T/(z-w)^3 + 3T'/(z-w)^2 + T''/(z-w)
    tdt = OPE(T, d(T))
    assert tdt.pole(5) == 2 * c * One
    assert tdt.pole(4) == od.Op()
    assert tdt.pole(3) == 4 * T
    assert tdt.pole(2) == 3 * d(T)
    assert tdt.pole(1) == d(T, 2)

    # and T'(z)T(w): pole shift with (-1)^i (j)_i
    dtt = OPE(d(T), T)
    assert dtt.pole(5) == -2 * c * One
    assert dtt.pole(3) == -4 * T
    assert dtt.pole(2) == -d(T)
    print("test_virasoro_basics OK")


def test_virasoro_jacobi():
    reset()
    c = sp.Symbol('c')
    T = bosonic('T')
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])
    assert jacobi_satisfied(T, T, T)
    print("test_virasoro_jacobi OK")


def test_sugawara_free_boson():
    """J(z)J(w) ~ 1/(z-w)^2,  T = NO(J,J)/2  ->  c = 1."""
    reset()
    J = bosonic('J')
    define_ope(J, J, [One, 0])
    T = NO(J, J) / 2

    tj = OPE(T, J)
    assert tj == opdata(J, d(J))               # J/(z-w)^2 + J'/(z-w)

    jt = OPE(J, T)
    assert jt == opdata(J, 0)                  # J/(z-w)^2 exactly

    tt = OPE(T, T)
    assert tt == opdata(Rational(1, 2) * One, 0, 2 * T, d(T))   # c = 1
    assert jacobi_satisfied(J, J, J)
    print("test_sugawara_free_boson OK")


def test_free_fermion():
    """psi(z)psi(w) ~ 1/(z-w),  T = -NO(psi, d psi)/2  ->  c = 1/2."""
    reset()
    psi = fermionic('psi')
    define_ope(psi, psi, [One])

    # NO(psi, psi) = 0 for a free fermion
    assert NO(psi, psi) == od.Op()

    T = -NO(psi, d(psi)) / 2
    tpsi = OPE(T, psi)
    assert tpsi == opdata(Rational(1, 2) * psi, d(psi))   # h = 1/2

    tt = OPE(T, T)
    assert tt == opdata(Rational(1, 4) * One, 0, 2 * T, d(T))   # c = 1/2
    assert jacobi_satisfied(psi, psi, psi)
    print("test_free_fermion OK")


def test_bc_ghosts():
    """b, c ghosts (lambda = 2):  T = -2 NO(b, dc) - NO(db, c),  c_gh = -26."""
    reset()
    b, cg = fermionic('b', 'cg')
    define_ope(b, cg, [One])
    T = -2 * NO(b, d(cg)) - NO(d(b), cg)

    tb = OPE(T, b)
    assert tb == opdata(2 * b, d(b))           # weight 2
    tc = OPE(T, cg)
    assert tc == opdata(-cg, d(cg))            # weight -1

    tt = OPE(T, T)
    assert tt == opdata(-13 * One, 0, 2 * T, d(T))   # c = -26
    assert jacobi_satisfied(b, cg, b)
    assert jacobi_satisfied(T, b, cg)
    print("test_bc_ghosts OK")


def test_TT_composite_and_quasiprimary():
    """T(z) NO(T,T)(w) and the quasiprimary Lambda = NO(T,T) - 3/10 T''.

    Exact Virasoro-module results:
        L2  :TT: = (8+c) T,     L1 :TT: = 3 T',   L2 d^2T = 12 T
        T(z)Lambda(w) = (5c+22)/5 T /(z-w)^4 + 4 Lambda /(z-w)^2
                        + Lambda' /(z-w)      (poles 6,5,3 vanish).
    """
    reset()
    c = sp.Symbol('c')
    T = bosonic('T')
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])

    W = NO(T, T)
    tw = OPE(T, W)
    assert tw.pole(6) == 3 * c * One
    assert tw.pole(5) == od.Op()
    assert tw.pole(4) == (c + 8) * T
    assert tw.pole(3) == 3 * d(T)
    assert tw.pole(2) == 4 * W
    assert tw.pole(1) == d(W)

    Lam = W - Rational(3, 10) * d(T, 2)
    tl = OPE(T, Lam).simplify()
    assert tl.pole(6) == od.Op()
    assert tl.pole(5) == od.Op()
    assert tl.pole(4) == (5 * c + 22) / 5 * T
    assert tl.pole(3) == od.Op()               # quasiprimary
    assert tl.pole(2) == 4 * Lam
    assert tl.pole(1) == d(Lam)
    print("test_TT_composite_and_quasiprimary OK")


def test_commutation_consistency():
    """OPE(B,A) computed by the package must satisfy the commutation formula
    against OPE(A,B), including fermion signs."""
    reset()
    psi1, psi2 = fermionic('psi1', 'psi2')
    k = sp.Symbol('k')
    define_ope(psi1, psi2, [k * One])
    ab = OPE(psi1, psi2)
    ba = OPE(psi2, psi1)
    assert ab.pole(1) == k * One
    assert ba.pole(1) == k * One               # (-1)^{|A||B|} * (-1)^1 * ... = +k
    # NO reordering identity: NO(psi2, psi1) = -NO(psi1, psi2) + d-corrections
    lhs = NO(psi2, psi1)
    rhs = -NO(psi1, psi2)                      # [psi1 psi2]_1 = k One, d(One) = 0
    assert lhs == rhs
    print("test_commutation_consistency OK")


def test_no_associativity_rearrangement():
    """NO(NO(J,J),J) vs NO(J,NO(J,J)) differ by known derivative terms."""
    reset()
    J = bosonic('J')
    define_ope(J, J, [One, 0])
    lhs = NO(NO(J, J), J)
    rhs = NO(J, NO(J, J))
    diff = (lhs - rhs).simplify()
    # [[JJ]0 J]0 - [J [JJ]0]0 = sum_{l>0} 1/l! ( d^l J [JJ]_l + d^l J [JJ]_l )
    #                        = 2 * (1/2!) d^2 J = d^2 J
    assert diff == d(J, 2)
    print("test_no_associativity_rearrangement OK")


def test_affine_su2_sugawara():
    """su(2)_k currents, component-wise; Sugawara T with c = 3k/(k+2)."""
    reset()
    k = sp.Symbol('k')
    J3, Jp, Jm = bosonic('J3', 'Jp', 'Jm')
    # conventions:  J3 J3 ~ (k/2)/(z-w)^2 ;  J3 J± ~ ±J±/(z-w)
    # J+ J- ~ k/(z-w)^2 + 2 J3/(z-w)
    define_ope(J3, J3, [k / 2 * One, 0])
    define_ope(J3, Jp, [Jp])
    define_ope(J3, Jm, [-1 * Jm])
    define_ope(Jp, Jm, [k * One, 2 * J3])
    define_ope(Jp, Jp, [])
    define_ope(Jm, Jm, [])

    assert jacobi_satisfied(J3, Jp, Jm)
    assert jacobi_satisfied(Jp, Jm, Jp)

    # Sugawara:  T = 1/(k+2) ( NO(J3,J3) + (NO(Jp,Jm)+NO(Jm,Jp))/2 )
    T = (NO(J3, J3) + Rational(1, 2) * (NO(Jp, Jm) + NO(Jm, Jp))) / (k + 2)

    for cur in (J3, Jp, Jm):
        tj = OPE(T, cur).simplify(sp.cancel)
        assert tj.pole(3) == od.Op()
        assert tj.pole(2) == cur
        assert tj.pole(1) == d(cur)

    tt = OPE(T, T).simplify(sp.cancel)
    csug = 3 * k / (k + 2)
    assert tt.pole(4) == csug / 2 * One
    assert tt.pole(3) == od.Op()
    assert tt.pole(2) == (2 * T).simplify(sp.cancel)
    assert tt.pole(1) == d(T).simplify(sp.cancel)
    print("test_affine_su2_sugawara OK")


def test_classical_mode():
    """Poisson brackets: classical Sugawara has no central term."""
    reset()
    set_ope_options(method=CLASSICAL)
    J = bosonic('J')
    define_ope(J, J, [One, 0])
    T = NO(J, J) / 2
    tt = OPE(T, T)
    assert tt == opdata(2 * T, d(T))           # no c/2 (z-w)^{-4} classically
    assert jacobi_satisfied(T, T, T)
    set_ope_options(method=QUANTUM)
    print("test_classical_mode OK")


def test_negative_and_zero_poles():
    reset()
    c = sp.Symbol('c')
    T = bosonic('T')
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])
    assert OPEPole(0, T, T) == NO(T, T)
    assert OPEPole(-1, T, T) == NO(d(T), T)
    assert OPEPole(-2, T, T) == NO(d(T, 2), T) / 2
    print("test_negative_and_zero_poles OK")


def test_lambda_lambda_central_term():
    """Lambda(z)Lambda(w) central term c(5c+22)/10 (composite x composite)."""
    reset()
    c = sp.Symbol('c')
    T = bosonic('T')
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])
    Lam = NO(T, T) - Rational(3, 10) * d(T, 2)
    ll = OPE(Lam, Lam).simplify(sp.cancel)
    # pole 8: <Lambda Lambda> = c(5c+22)/10 (famous).  pole 6: from the exact
    # Ward identity for the quasiprimary 3pt function <T Lambda Lambda>,
    # a * c/2 = 4 * c(5c+22)/10  =>  a = 4(5c+22)/5;  pole 5 = a/2 * dT.
    assert ll.pole(8) == c * (5 * c + 22) / 10 * One
    assert ll.pole(7) == od.Op()
    assert ll.pole(6) == 4 * (5 * c + 22) / 5 * T
    assert ll.pole(5) == 2 * (5 * c + 22) / 5 * d(T)
    print("test_lambda_lambda_central_term OK")


def test_n1_super_virasoro_free_fields():
    """G = NO(J, psi), T = NO(J,J)/2 - NO(psi, dpsi)/2 : N=1 SCA with c=3/2."""
    reset()
    J = bosonic('J')
    psi = fermionic('psi')
    define_ope(J, J, [One, 0])
    define_ope(psi, psi, [One])
    define_ope(J, psi, [])
    G = NO(J, psi)
    T = NO(J, J) / 2 - NO(psi, d(psi)) / 2

    gg = OPE(G, G)
    assert gg.pole(3) == One                   # 2c/3 = 1  ->  c = 3/2
    assert gg.pole(2) == od.Op()
    assert gg.pole(1) == 2 * T
    tg = OPE(T, G)
    assert tg.pole(2) == Rational(3, 2) * G    # weight 3/2
    assert tg.pole(1) == d(G)
    assert jacobi_satisfied(G, G, G)
    assert jacobi_satisfied(T, G, psi)
    print("test_n1_super_virasoro_free_fields OK")


def test_indexed_free_bosons_symbolic_N():
    """N free bosons via a pattern rule; Sugawara gives c = N (symbolic N)."""
    reset()
    N = sp.Symbol('N', positive=True)
    a, b1, b2, i, j = od.idx('a b1 b2 i j')
    J = od.bosonic_family('J')
    define_ope(J(i), J(j), [od.Delta(i, j) * One, 0])

    assert OPE(J(1), J(1)).pole(2) == One
    assert OPE(J(1), J(2)).max_pole == 0
    assert OPE(J(a), J(b1)).pole(2) == od.Delta(a, b1) * One

    T = od.dsum(NO(J(a), J(a)), a, N) / 2
    assert OPE(T, J(b1)) == opdata(J(b1), d(J(b1)))
    tt = OPE(T, T).simplify(sp.expand)
    assert tt.pole(4) == N / 2 * One            # c = N
    assert tt.pole(3) == od.Op()
    assert tt.pole(2) == 2 * T
    assert tt.pole(1) == d(T)
    assert jacobi_satisfied(J(b1), J(b2), J(a))
    assert jacobi_satisfied(T, J(b1), J(b2))
    assert jacobi_satisfied(T, T, T)
    print("test_indexed_free_bosons_symbolic_N OK")


def test_indexed_free_fermions_symbolic_N():
    """N free fermions: c = N/2 (symbolic N), fermionic family signs."""
    reset()
    N = sp.Symbol('N', positive=True)
    a, b, i, j = od.idx('a b i j')
    psi = od.fermionic_family('psi')
    define_ope(psi(i), psi(j), [od.Delta(i, j) * One])

    assert NO(psi(a), psi(a)) == od.Op()
    assert NO(psi(1), psi(2)) == -NO(psi(2), psi(1))

    T = -od.dsum(NO(psi(a), d(psi(a))), a, N) / 2
    tt = OPE(T, T).simplify(sp.expand)
    assert tt.pole(4) == N / 4 * One            # c = N/2
    assert tt.pole(2) == 2 * T
    assert tt.pole(1) == d(T)
    assert OPE(T, psi(b)).pole(2) == psi(b) / 2
    print("test_indexed_free_fermions_symbolic_N OK")


def test_indexed_su2_pattern_rule():
    """su(2)_k via one pattern rule with Levi-Civita structure constants."""
    reset()
    k = sp.Symbol('k')
    a, i, j, l = od.idx('a i j l')
    J = od.bosonic_family('J')
    define_ope(J(i), J(j),
               [k / 2 * od.Delta(i, j) * One,
                od.dsum(sp.I * od.Eps(i, j, l) * J(l), l, 3)])

    assert OPE(J(1), J(2)) == opdata(sp.I * J(3))
    assert OPE(J(2), J(1)) == opdata(-sp.I * J(3))
    assert jacobi_satisfied(J(1), J(2), J(3))

    T = od.dsum(NO(J(a), J(a)), a, 3) / (k + 2)
    tt = OPE(T, T).simplify(sp.cancel)
    assert tt.pole(4) == 3 * k / (2 * (k + 2)) * One    # c = 3k/(k+2)
    assert tt.pole(3) == od.Op()
    assert tt.pole(2) == (2 * T).simplify(sp.cancel)
    assert tt.pole(1) == d(T).simplify(sp.cancel)
    for comp in (1, 2, 3):
        tj = OPE(T, J(comp)).simplify(sp.cancel)
        assert tj.pole(3) == od.Op() and tj.pole(2) == J(comp) \
            and tj.pole(1) == d(J(comp))
    assert jacobi_satisfied(T, J(1), J(2))
    print("test_indexed_su2_pattern_rule OK")


def test_mixed_pattern_and_commutation():
    """OPE[T, phi[i_]] style rule; phi(z)T(w) follows by commutation."""
    reset()
    c, h = sp.symbols('c h')
    i, a = od.idx('i a')
    Tv = bosonic('Tv')
    phi = od.bosonic_family('phi')
    define_ope(Tv, Tv, [c / 2 * One, 0, 2 * Tv, d(Tv)])
    define_ope(Tv, phi(i), [h * phi(i), d(phi(i))])
    assert OPE(Tv, phi(a)) == opdata(h * phi(a), d(phi(a)))
    pt = OPE(phi(a), Tv)
    assert pt.pole(2) == h * phi(a)
    assert pt.pole(1) == (h - 1) * d(phi(a))
    print("test_mixed_pattern_and_commutation OK")


def test_dummy_sum_machinery():
    """Delta contraction, nested sums, scalar extraction, residual sums."""
    reset()
    N = sp.Symbol('N', positive=True)
    a, b, x, y, i, j = od.idx('a b x y i j')
    J = od.bosonic_family('J')
    define_ope(J(i), J(j), [od.Delta(i, j) * One, 0])

    # sum_b Delta(a,b) NO(J(a),J(b)) = NO(J(a),J(a)), then sum over a
    X = od.dsum(od.dsum(od.Delta(a, b) * NO(J(a), J(b)), b, N), a, N)
    assert X == od.dsum(NO(J(a), J(a)), a, N)
    # sum over an absent index multiplies by the dimension
    assert od.dsum(J(1), a, N) == N * J(1)
    # scalar extraction: canonical form of scalar multiples
    f = sp.Function('f')
    assert 3 * od.dsum(f(y) * J(y), y, N) == od.dsum(3 * f(y) * J(y), y, N)
    # residual symbolic sums and their OPEs
    U = od.dsum(f(y) * J(y), y, N)
    assert OPE(J(x), U).pole(2) == f(x) * One
    assert d(U) == od.dsum(f(y) * d(J(y)), y, N)
    T = od.dsum(NO(J(a), J(a)), a, N) / 2
    tu = OPE(T, U).simplify()
    assert tu.pole(2) == U and tu.pole(1) == d(U)
    print("test_dummy_sum_machinery OK")


if __name__ == "__main__":
    test_virasoro_basics()
    test_virasoro_jacobi()
    test_sugawara_free_boson()
    test_free_fermion()
    test_bc_ghosts()
    test_TT_composite_and_quasiprimary()
    test_commutation_consistency()
    test_no_associativity_rearrangement()
    test_affine_su2_sugawara()
    test_classical_mode()
    test_negative_and_zero_poles()
    test_lambda_lambda_central_term()
    test_n1_super_virasoro_free_fields()
    test_indexed_free_bosons_symbolic_N()
    test_indexed_free_fermions_symbolic_N()
    test_indexed_su2_pattern_rule()
    test_mixed_pattern_and_commutation()
    test_dummy_sum_machinery()
    print("\nALL TESTS PASSED")
