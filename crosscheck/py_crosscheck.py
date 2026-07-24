"""py_crosscheck.py -- compare pyOPEdefs against the original Mathematica
OPEdefs, computation by computation.

Workflow:
    1. wolframscript -file mma_crosscheck.wls path/to/OPEdefs.m
       -> writes mma_results.txt
    2. python py_crosscheck.py mma_results.txt
       -> recomputes everything with pyOPEdefs and diffs each pole.

Self-test of the parser/pipeline (no Mathematica needed):
    python py_crosscheck.py --selftest

The comparison is convention-independent: each Mathematica expression is
re-evaluated *through the pyOPEdefs constructors* (NO, d, ...), so both
sides are brought to the same canonical form before Op.__eq__ is applied.
"""
import re
import sys
import sympy as sp
from sympy import Rational

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from opedefs import *
import opedefs as od


# ----------------------------------------------------------------------
# The same battery, in pyOPEdefs.  Each entry: name -> (kind, object)
#   kind "ope": OPEData whose poles are compared one by one
#   kind "op" : a single operator expression
# ----------------------------------------------------------------------
def build_battery():
    cases = {}
    c, k = sp.symbols("c k")

    # 1. Virasoro
    reset()
    T = bosonic("T")
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])
    Lam = NO(T, T) - Rational(3, 10) * d(T, 2)
    cases["vir.TT"] = ("ope", OPE(T, T))
    cases["vir.T_TT"] = ("ope", OPE(T, NO(T, T)))
    cases["vir.T_Lam"] = ("ope", OPE(T, Lam))
    cases["vir.Lam_Lam"] = ("ope", OPE(Lam, Lam))
    cases["vir.T_dT"] = ("ope", OPE(T, d(T)))
    cases["vir.NOTT"] = ("op", NO(T, T))
    ns1 = {"T": T}

    # 2. free boson
    reset()
    J = bosonic("J")
    define_ope(J, J, [One, 0])
    Tfb = NO(J, J) / 2
    cases["fb.T_J"] = ("ope", OPE(Tfb, J))
    cases["fb.J_T"] = ("ope", OPE(J, Tfb))
    cases["fb.T_T"] = ("ope", OPE(Tfb, Tfb))
    cases["fb.NOassoc"] = ("op", NO(NO(J, J), J) - NO(J, NO(J, J)))
    ns2 = {"J": J}

    # 3. free fermion
    reset()
    psi = fermionic("psi")
    define_ope(psi, psi, [One])
    Tff = -NO(psi, d(psi)) / 2
    cases["ff.T_psi"] = ("ope", OPE(Tff, psi))
    cases["ff.T_T"] = ("ope", OPE(Tff, Tff))
    cases["ff.NOpsipsi"] = ("op", NO(psi, psi))
    ns3 = {"psi": psi}

    # 4. bc ghosts
    reset()
    bb, cg = fermionic("bb", "cg")
    define_ope(bb, cg, [One])
    Tgh = -2 * NO(bb, d(cg)) - NO(d(bb), cg)
    cases["bc.T_b"] = ("ope", OPE(Tgh, bb))
    cases["bc.T_c"] = ("ope", OPE(Tgh, cg))
    cases["bc.T_T"] = ("ope", OPE(Tgh, Tgh))
    cases["bc.NOrev"] = ("op", NO(cg, bb))
    ns4 = {"bb": bb, "cg": cg}

    # 5. su(2)_k
    reset()
    J3, Jp, Jm = bosonic("J3", "Jp", "Jm")
    define_ope(J3, J3, [k / 2 * One, 0])
    define_ope(J3, Jp, [Jp])
    define_ope(J3, Jm, [-1 * Jm])
    define_ope(Jp, Jm, [k * One, 2 * J3])
    Tsu = (NO(J3, J3) + (NO(Jp, Jm) + NO(Jm, Jp)) / 2) / (k + 2)
    cases["su2.T_J3"] = ("ope", OPE(Tsu, J3))
    cases["su2.T_Jp"] = ("ope", OPE(Tsu, Jp))
    cases["su2.T_T"] = ("ope", OPE(Tsu, Tsu))
    ns5 = {"J3": J3, "Jp": Jp, "Jm": Jm}

    # 6. N=2 SCA
    reset()
    TT2, JJ2 = bosonic("TT2", "JJ2")
    Gp, Gm = fermionic("Gp", "Gm")
    define_ope(TT2, TT2, [c / 2 * One, 0, 2 * TT2, d(TT2)])
    define_ope(TT2, JJ2, [JJ2, d(JJ2)])
    define_ope(TT2, Gp, [Rational(3, 2) * Gp, d(Gp)])
    define_ope(TT2, Gm, [Rational(3, 2) * Gm, d(Gm)])
    define_ope(JJ2, JJ2, [c / 3 * One, 0])
    define_ope(JJ2, Gp, [Gp])
    define_ope(JJ2, Gm, [-1 * Gm])
    define_ope(Gp, Gm, [2 * c / 3 * One, 2 * JJ2, 2 * TT2 + d(JJ2)])
    cases["n2.GmGp"] = ("ope", OPE(Gm, Gp))
    cases["n2.T_Gp"] = ("ope", OPE(TT2, Gp))
    cases["n2.Gp_T"] = ("ope", OPE(Gp, TT2))
    ns6 = {"TT2": TT2, "JJ2": JJ2, "Gp": Gp, "Gm": Gm}

    # 7. W3
    reset()
    TW, W = bosonic("TW", "W")
    define_ope(TW, TW, [c / 2 * One, 0, 2 * TW, d(TW)])
    define_ope(TW, W, [3 * W, d(W)])
    LamW = NO(TW, TW) - Rational(3, 10) * d(TW, 2)
    bet = 16 / (22 + 5 * c)
    define_ope(W, W, [c / 3 * One, 0, 2 * TW, d(TW),
                      2 * bet * LamW + Rational(3, 10) * d(TW, 2),
                      bet * d(LamW) + Rational(1, 15) * d(TW, 3)])
    cases["w3.T_W"] = ("ope", OPE(TW, W))
    cases["w3.W_W"] = ("ope", OPE(W, W))
    cases["w3.W_dW"] = ("ope", OPE(W, d(W)))
    cases["w3.T_Lam"] = ("ope", OPE(TW, LamW))
    ns7 = {"TW": TW, "W": W}

    # merged evaluation namespace for parsing Mathematica expressions.
    # NOTE: operators live in *different* reset() worlds above; parsing a
    # Mathematica expression only needs the Op objects themselves, which
    # stay valid, but NO()/OPE() lookups need the right tables.  We
    # therefore rebuild every world ONCE more into a single shared world:
    reset()
    c, k = sp.symbols("c k")
    T = bosonic("T")
    define_ope(T, T, [c / 2 * One, 0, 2 * T, d(T)])
    J = bosonic("J")
    define_ope(J, J, [One, 0])
    psi = fermionic("psi")
    define_ope(psi, psi, [One])
    bb, cg = fermionic("bb", "cg")
    define_ope(bb, cg, [One])
    J3, Jp, Jm = bosonic("J3", "Jp", "Jm")
    define_ope(J3, J3, [k / 2 * One, 0])
    define_ope(J3, Jp, [Jp])
    define_ope(J3, Jm, [-1 * Jm])
    define_ope(Jp, Jm, [k * One, 2 * J3])
    TT2, JJ2 = bosonic("TT2", "JJ2")
    Gp, Gm = fermionic("Gp", "Gm")
    define_ope(TT2, TT2, [c / 2 * One, 0, 2 * TT2, d(TT2)])
    define_ope(TT2, JJ2, [JJ2, d(JJ2)])
    define_ope(TT2, Gp, [Rational(3, 2) * Gp, d(Gp)])
    define_ope(TT2, Gm, [Rational(3, 2) * Gm, d(Gm)])
    define_ope(JJ2, JJ2, [c / 3 * One, 0])
    define_ope(JJ2, Gp, [Gp])
    define_ope(JJ2, Gm, [-1 * Gm])
    define_ope(Gp, Gm, [2 * c / 3 * One, 2 * JJ2, 2 * TT2 + d(JJ2)])
    TW, W = bosonic("TW", "W")
    define_ope(TW, TW, [c / 2 * One, 0, 2 * TW, d(TW)])
    define_ope(TW, W, [3 * W, d(W)])
    LamW = NO(TW, TW) - Rational(3, 10) * d(TW, 2)
    define_ope(W, W, [c / 3 * One, 0, 2 * TW, d(TW),
                      2 * bet * LamW + Rational(3, 10) * d(TW, 2),
                      bet * d(LamW) + Rational(1, 15) * d(TW, 3)])

    # ... and recompute the battery inside this shared world so that the
    # canonical basis elements of "cases" and of parsed expressions agree.
    Lam = NO(T, T) - Rational(3, 10) * d(T, 2)
    Tfb = NO(J, J) / 2
    Tff = -NO(psi, d(psi)) / 2
    Tgh = -2 * NO(bb, d(cg)) - NO(d(bb), cg)
    Tsu = (NO(J3, J3) + (NO(Jp, Jm) + NO(Jm, Jp)) / 2) / (k + 2)
    cases = {
        "vir.TT": ("ope", OPE(T, T)),
        "vir.T_TT": ("ope", OPE(T, NO(T, T))),
        "vir.T_Lam": ("ope", OPE(T, Lam)),
        "vir.Lam_Lam": ("ope", OPE(Lam, Lam)),
        "vir.T_dT": ("ope", OPE(T, d(T))),
        "vir.NOTT": ("op", NO(T, T)),
        "fb.T_J": ("ope", OPE(Tfb, J)),
        "fb.J_T": ("ope", OPE(J, Tfb)),
        "fb.T_T": ("ope", OPE(Tfb, Tfb)),
        "fb.NOassoc": ("op", NO(NO(J, J), J) - NO(J, NO(J, J))),
        "ff.T_psi": ("ope", OPE(Tff, psi)),
        "ff.T_T": ("ope", OPE(Tff, Tff)),
        "ff.NOpsipsi": ("op", NO(psi, psi)),
        "bc.T_b": ("ope", OPE(Tgh, bb)),
        "bc.T_c": ("ope", OPE(Tgh, cg)),
        "bc.T_T": ("ope", OPE(Tgh, Tgh)),
        "bc.NOrev": ("op", NO(cg, bb)),
        "su2.T_J3": ("ope", OPE(Tsu, J3)),
        "su2.T_Jp": ("ope", OPE(Tsu, Jp)),
        "su2.T_T": ("ope", OPE(Tsu, Tsu)),
        "n2.GmGp": ("ope", OPE(Gm, Gp)),
        "n2.T_Gp": ("ope", OPE(TT2, Gp)),
        "n2.Gp_T": ("ope", OPE(Gp, TT2)),
        "w3.T_W": ("ope", OPE(TW, W)),
        "w3.W_W": ("ope", OPE(W, W)),
        "w3.W_dW": ("ope", OPE(W, d(W))),
        "w3.T_Lam": ("ope", OPE(TW, LamW)),
    }

    ns = {
        "NO": NO, "One": One, "d": d,
        "Derivative": (lambda n: (lambda X: d(X, n))),
        "Rational": Rational, "I": sp.I, "Sqrt": sp.sqrt,
        "c": sp.Symbol("c"), "k": sp.Symbol("k"),
        "T": T, "J": J, "psi": psi, "bb": bb, "cg": cg,
        "J3": J3, "Jp": Jp, "Jm": Jm,
        "TT2": TT2, "JJ2": JJ2, "Gp": Gp, "Gm": Gm, "TW": TW, "W": W,
    }
    return cases, ns


# ----------------------------------------------------------------------
# Mathematica InputForm  ->  evaluatable pyOPEdefs expression
# ----------------------------------------------------------------------
_PRIMES = [(re.compile(r"([A-Za-z][A-Za-z0-9]*)'''"), r"d(\1,3)"),
           (re.compile(r"([A-Za-z][A-Za-z0-9]*)''"), r"d(\1,2)"),
           (re.compile(r"([A-Za-z][A-Za-z0-9]*)'"), r"d(\1,1)")]


def mma_to_py(s):
    s = s.strip()
    for pat, rep in _PRIMES:
        s = pat.sub(rep, s)
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("^", "**")
    # exact rationals:  3/10 -> Rational(3,10)   (only literal int/int;
    # never right after "**", where the integer is an exponent: c**2/2)
    s = re.sub(r"(?<!\*\*)(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])",
               r"Rational(\1,\2)", s)
    return s


def parse_mma(s, ns):
    py = mma_to_py(s)
    val = eval(py, {"__builtins__": {}}, ns)
    if not isinstance(val, od.Op):          # a pure scalar means val * One
        val = sp.sympify(val) * One
    return val


# ----------------------------------------------------------------------
def compare(results_path):
    cases, ns = build_battery()
    lines = [ln.rstrip("\n") for ln in open(results_path) if ln.strip()]
    npass = nfail = 0
    fails = []
    seen = set()
    for ln in lines:
        parts = ln.split("|")
        if parts[0] == "MAX":
            _, name, n = parts
            kind, obj = cases[name]
            ok = (obj.max_pole == int(n))
        elif parts[0] == "CASE":
            _, name, q, expr = parts[0], parts[1], parts[2], "|".join(parts[3:])
            kind, obj = cases[name]
            ok = (obj.pole(int(q)) == parse_mma(expr, ns))
            name = "%s pole %s" % (name, q)
        elif parts[0] == "OP":
            _, name, expr = parts[0], parts[1], "|".join(parts[2:])
            kind, obj = cases[name]
            ok = (obj == parse_mma(expr, ns))
        else:
            continue
        seen.add(parts[1])
        if ok:
            npass += 1
        else:
            nfail += 1
            fails.append(name)
    missing = sorted(set(cases) - seen)
    print("compared %d items:  PASS %d   FAIL %d" % (npass + nfail, npass, nfail))
    for f in fails:
        print("  MISMATCH:", f)
    if missing:
        print("  (cases not present in the Mathematica file: %s)" % ", ".join(missing))
    return nfail == 0 and not fails


# ----------------------------------------------------------------------
# Self-test: serialize our own results in Mathematica InputForm style,
# then round-trip them through the parser and comparator.
# ----------------------------------------------------------------------
def _to_mma(X):
    def basis_str(b):
        if isinstance(b, od._OneB):
            return "One"
        if isinstance(b, od._F):
            if b.n == 0:
                return b.op.name
            return "Derivative[%d][%s]" % (b.n, b.op.name)
        if isinstance(b, od._NOb):
            return "NO[%s, %s]" % (basis_str(b.l), basis_str(b.r))
        raise TypeError(b)
    parts = []
    for b, coeff in sorted(X.terms.items(), key=lambda t: str(t[0])):
        cs = sp.printing.sstr(sp.nsimplify(coeff))
        cs = cs.replace("**", "^")
        parts.append("(%s)*%s" % (cs, basis_str(b)))
    return " + ".join(parts) if parts else "0"


def selftest():
    cases, ns = build_battery()
    with open("selftest_results.txt", "w") as f:
        for name, (kind, obj) in cases.items():
            if kind == "ope":
                f.write("MAX|%s|%d\n" % (name, obj.max_pole))
                for q in range(1, obj.max_pole + 1):
                    f.write("CASE|%s|%d|%s\n" % (name, q, _to_mma(obj.pole(q))))
            else:
                f.write("OP|%s|%s\n" % (name, _to_mma(obj)))
    ok = compare("selftest_results.txt")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    path = sys.argv[1] if len(sys.argv) > 1 else "mma_results.txt"
    sys.exit(0 if compare(path) else 1)
