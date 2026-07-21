"""examples.py -- pyOPEdefs demonstration.

Mirrors the standard examples of the original OPEdefs documentation
(Thielemans, hep-th/9506159, ch. 3).
"""
import sympy as sp
from sympy import Rational
from opedefs import *

# ======================================================================
# 1. Virasoro algebra
#    Mathematica:  Bosonic[T]
#                  OPE[T,T] = MakeOPE[{c/2 One, 0, 2 T, T'}]
# ======================================================================
print("=" * 70)
print("1. Virasoro")
reset()
c = sp.Symbol('c')
T = bosonic('T')
define_ope(T, T, [c/2*One, 0, 2*T, d(T)])

print("OPE(T,T)       =", OPE(T, T))
print("OPE(T,T'')     =", OPE(T, d(T, 2)))
print("OPE(T,NO(T,T)) =", OPE(T, NO(T, T)))

Lam = NO(T, T) - Rational(3, 10)*d(T, 2)
print("\nQuasiprimary Lambda = NO(T,T) - 3/10 T'':")
print("OPE(T,Lambda)  =", OPE(T, Lam).simplify())
print("Central term of OPE(Lambda,Lambda):",
      OPE(Lam, Lam).simplify(sp.cancel).pole(8), "  [= c(5c+22)/10 One]")
print("Jacobi(T,T,T) satisfied:", jacobi_satisfied(T, T, T))

# ======================================================================
# 2. Sugawara construction, su(2)_k
# ======================================================================
print("\n" + "=" * 70)
print("2. su(2)_k currents and the Sugawara stress tensor")
reset()
k = sp.Symbol('k')
J3, Jp, Jm = bosonic('J3', 'Jp', 'Jm')
define_ope(J3, J3, [k/2*One, 0])
define_ope(J3, Jp, [Jp])
define_ope(J3, Jm, [-1*Jm])
define_ope(Jp, Jm, [k*One, 2*J3])

Tsug = (NO(J3, J3) + Rational(1, 2)*(NO(Jp, Jm) + NO(Jm, Jp))) / (k + 2)
tt = OPE(Tsug, Tsug).simplify(sp.cancel)
print("OPE(T,J3)  =", OPE(Tsug, J3).simplify(sp.cancel))
print("c/2 from OPE(T,T) pole 4:", tt.pole(4), "  [c = 3k/(k+2)]")

# ======================================================================
# 3. bc ghost system (lambda = 2), c = -26
# ======================================================================
print("\n" + "=" * 70)
print("3. bc ghosts")
reset()
b, cg = fermionic('b', 'c')
define_ope(b, cg, [One])
Tgh = -2*NO(b, d(cg)) - NO(d(b), cg)
print("OPE(T,b) =", OPE(Tgh, b))
print("OPE(T,c) =", OPE(Tgh, cg))
print("OPE(T,T) =", OPE(Tgh, Tgh), "  [pole 4 = -13 One  =>  c = -26]")

# ======================================================================
# 4. Classical (Poisson bracket) computations
#    Mathematica:  SetOPEOptions[OPEMethod, ClassicalOPEs]
# ======================================================================
print("\n" + "=" * 70)
print("4. Classical mode")
reset()
set_ope_options(method=CLASSICAL)
J = bosonic('J')
define_ope(J, J, [One, 0])
Tcl = NO(J, J)/2
print("{T,T} =", OPE(Tcl, Tcl), "  (no central term classically)")
set_ope_options(method=QUANTUM)

# ======================================================================
# 5. Series form / poles
# ======================================================================
print("\n" + "=" * 70)
print("5. Output forms")
reset()
c = sp.Symbol('c')
T = bosonic('T')
define_ope(T, T, [c/2*One, 0, 2*T, d(T)])
ope_tt = OPE(T, T)
print("series :", ope_tt.series_str())
print("pole 2 :", ope_tt.pole(2))
print("OPEPole(0,T,T) = NO(T,T) :", OPEPole(0, T, T))
print("OPEPole(-1,T,T)          :", OPEPole(-1, T, T))

# ======================================================================
# 6. Indexed families and pattern OPE rules (the Delta`/Dummies` analogue)
#    Mathematica:  Bosonic[J[_]]
#                  OPE[J[i_],J[j_]] = MakeOPE[{k Delta[i,j] One, ...}]
# ======================================================================
print("\n" + "=" * 70)
print("6. Indexed families: su(2)_k from ONE pattern rule")
reset()
k = sp.Symbol('k')
a, i, j, l = idx('a i j l')
J = bosonic_family('J')
define_ope(J(i), J(j), [k/2*Delta(i, j)*One,
                        dsum(sp.I*Eps(i, j, l)*J(l), l, 3)])
print("OPE(J(1),J(2)) =", OPE(J(1), J(2)))
Tsu2 = dsum(NO(J(a), J(a)), a, 3) / (k + 2)
print("OPE(T,T) pole 4 =", OPE(Tsu2, Tsu2).simplify(sp.cancel).pole(4),
      "  [c = 3k/(k+2)]")

# ======================================================================
# 7. Formal dummy sums with a SYMBOLIC dimension N
# ======================================================================
print("\n" + "=" * 70)
print("7. N free bosons, symbolic N:  c = N")
reset()
N = sp.Symbol('N', positive=True)
a, b, i, j = idx('a b i j')
J = bosonic_family('J')
define_ope(J(i), J(j), [Delta(i, j)*One, 0])
T = dsum(NO(J(a), J(a)), a, N) / 2       # Sugawara, formal sum over a
print("T           =", T)
print("OPE(T,J(b)) =", OPE(T, J(b)))     # Delta contraction: primary of h=1
print("OPE(T,T)    =", OPE(T, T).simplify(sp.expand))
print("Jacobi(T,T,T):", jacobi_satisfied(T, T, T))
