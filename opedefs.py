"""
opedefs.py -- pyOPEdefs
=======================

A pure-Python / SymPy reimplementation of **OPEdefs 3.1**, Kris Thielemans'
Mathematica package for computing Operator Product Expansions of composite
operators in meromorphic conformal field theory (chiral vertex algebras).

Given the OPEs of a set of "basic" fields (generators), OPEs of arbitrarily
complicated normal-ordered composites are computed automatically, and
normal-ordered products are reduced to a canonical standard form.

The algorithms implemented here are the ones documented in:

  * K. Thielemans, "A Mathematica package for computing operator product
    expansions", Int. J. Mod. Phys. C2 (1991) 787.
  * K. Thielemans, "An Algorithmic Approach to Operator Product Expansions,
    W-Algebras and W-Strings", PhD thesis, KU Leuven (1994),
    arXiv:hep-th/9506159.  (In particular eqs. (3.3.1)-(3.3.19).)

Please cite these references when using this package in publications.
This is an independent reimplementation; all credit for the algorithm
belongs to K. Thielemans.

Conventions
-----------
An OPE is stored as the list of its pole parts,

    A(z) B(w) = sum_{q=1}^{N}  [AB]_q(w) / (z-w)^q  +  regular,

with `[AB]_0 = NO(A,B)` the (point-splitting) normal-ordered product.
Coefficients are arbitrary SymPy expressions (central charges, levels, ...).

Quick start
-----------
    >>> import sympy as sp
    >>> from opedefs import *
    >>> c = sp.Symbol('c')
    >>> T = bosonic('T')
    >>> define_ope(T, T, [c/2*One, 0, 2*T, d(T)])   # highest pole first
    >>> OPE(T, T)
    << 4|| c/2*One ||3|| 0 ||2|| 2*T ||1|| T' >>
    >>> OPE(T, NO(T, T)).pole(4)
    (c + 8)*T

Both quantum OPEs (default) and classical Poisson-bracket computations
(`set_ope_options(method=CLASSICAL)`) are supported.

Differences w.r.t. the Mathematica original:
  * Pattern-indexed operator families ``J[i_]`` with the symbolic ``Delta``
    package are not supported; declare components individually (e.g. with
    ``bosonic('J1','J2','J3')``) or generate them in a Python loop.
  * Symbolic (non-integer) parities (``OPEOperator``) are not supported.
  * ``OPESave`` is unnecessary: use ``pickle`` on your own results if needed.
"""

from __future__ import annotations

import sympy as sp
from sympy import S, Rational

__version__ = "0.1.0"
__all__ = [
    "One", "bosonic", "fermionic", "define_ope",
    "OPE", "OPEPole", "NO", "d", "Op", "OPEData",
    "OPEJacobi", "OPESimplify", "MaxPole",
    "GetOperators", "GetCoefficients",
    "set_ope_options", "reset", "clear_caches",
    "QUANTUM", "CLASSICAL",
]

QUANTUM = "quantum"
CLASSICAL = "classical"


# ----------------------------------------------------------------------
# Global state (operator registry, OPE table, caches, options)
# ----------------------------------------------------------------------
class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.names = {}
        self.counter = 0
        self.ope_table = {}          # (BasicOperator, BasicOperator) -> tuple[Op] ascending
        self.family_ops = {}         # (family_pos, index_expr) -> _BasicOperator
        self.pattern_rules = {}      # (keyA, keyB) -> (patA, patB, tuple[Op]) ; key = family | op
        self.dummy_counter = 0       # for fresh dummy indices while computing
        self.method = QUANTUM
        self.no_ordering = -1        # <0: higher derivatives to the left (default)
        self.clear_caches()

    def clear_caches(self):
        self.ope_cache = {}          # (_Basis, _Basis) -> tuple[Op]
        self.no_cache = {}           # (_Basis, _Basis) -> Op


_state = _State()


def reset():
    """Forget all declared operators, OPEs, options and caches."""
    _state.reset()


def clear_caches():
    """Clear cached intermediate results (cf. ClearOPESavedValues[])."""
    _state.clear_caches()


def set_ope_options(method=None, no_ordering=None):
    """Set global options (cf. SetOPEOptions).

    method       : QUANTUM (default) or CLASSICAL (Poisson brackets).
    no_ordering  : ordering of derivatives of the same field inside NO;
                   -1 (default) puts higher derivatives to the left,
                   +1 puts lower derivatives to the left.
    """
    if method is not None:
        if method not in (QUANTUM, CLASSICAL):
            raise ValueError("method must be QUANTUM or CLASSICAL")
        _state.method = method
        _state.clear_caches()
    if no_ordering is not None:
        _state.no_ordering = int(no_ordering)
        _state.clear_caches()


def _csimp(c):
    """Normalise a scalar coefficient; must map 0-equivalent exprs to 0."""
    try:
        return sp.cancel(sp.together(c))
    except Exception:
        return sp.expand(c)


# ----------------------------------------------------------------------
# Basis elements of the operator algebra
# ----------------------------------------------------------------------
class _BasicOperator:
    __slots__ = ("name", "parity", "pos", "family", "index")

    def __init__(self, name, parity, pos, family=None, index=None):
        self.name = name
        self.parity = parity      # 0 = boson, 1 = fermion
        self.pos = pos            # declaration order (fixes NO ordering)
        self.family = family      # _Family for indexed operators, else None
        self.index = index        # sympy index expression, else None

    def __repr__(self):
        return self.name


class _Basis:
    """Abstract canonical basis element."""
    __slots__ = ()


class _OneB(_Basis):
    """The unit operator (only a zero mode)."""
    __slots__ = ()
    parity = 0

    def __hash__(self):
        return 0x0135

    def __eq__(self, other):
        return isinstance(other, _OneB)

    def __repr__(self):
        return "One"


_ONE = _OneB()


class _F(_Basis):
    """n-th derivative of a basic operator."""
    __slots__ = ("op", "n", "_h")

    def __init__(self, op, n=0):
        self.op = op
        self.n = n
        self._h = hash((op.pos, n))

    @property
    def parity(self):
        return self.op.parity

    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return isinstance(other, _F) and other.op is self.op and other.n == self.n

    def __repr__(self):
        if self.n == 0:
            return self.op.name
        if self.n <= 3:
            return self.op.name + "'" * self.n
        return "%s^(%d)" % (self.op.name, self.n)


class _NOb(_Basis):
    """Canonical normal-ordered product NO(left, right); left is a _F."""
    __slots__ = ("l", "r", "parity", "_h")

    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.parity = (l.parity + _parity(r)) % 2
        self._h = hash(("NO", l, r))

    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return isinstance(other, _NOb) and other.l == self.l and other.r == self.r

    def __repr__(self):
        return "NO[%r, %r]" % (self.l, self.r)


def _parity(b):
    return b.parity


def _swap_sign(x, y):
    """(-1)^{|x||y|}."""
    return S.NegativeOne if (_parity(x) % 2) and (_parity(y) % 2) else S.One


def _idx_key(op):
    """Deterministic sort key of an operator's index (families share pos)."""
    if op.index is None:
        return ()
    return sp.default_sort_key(sp.sympify(op.index))


def _sort_key(b):
    if isinstance(b, _OneB):
        return (0,)
    if isinstance(b, _F):
        return (1, b.op.pos, _idx_key(b.op), b.n)
    if isinstance(b, _NOb):
        return (2,) + _sort_key(b.l) + _sort_key(b.r)
    # _ISum
    return (3, sp.default_sort_key(b.dim)) + tuple(
        _sort_key(ib) + (sp.default_sort_key(ic),) for ib, ic in b.inner)


# ----------------------------------------------------------------------
# Op : linear combination of basis elements with SymPy coefficients
# ----------------------------------------------------------------------
class Op:
    """A (finite) linear combination of fields with SymPy coefficients."""
    __slots__ = ("terms",)

    def __init__(self, terms=None):
        t = {}
        if terms:
            for b, c in terms.items():
                c = _csimp(sp.sympify(c))
                if c != 0:
                    if b in t:
                        c = _csimp(t[b] + c)
                        if c == 0:
                            del t[b]
                            continue
                    t[b] = c
        self.terms = t

    # -- constructors ------------------------------------------------
    @staticmethod
    def _basis(b):
        o = Op()
        o.terms = {b: S.One}
        return o

    # -- predicates --------------------------------------------------
    @property
    def is_zero(self):
        return not self.terms

    def parity(self):
        """Grassmann parity (0/1); raises if the expression is not homogeneous."""
        if not self.terms:
            return 0
        ps = {(_parity(b) % 2) for b in self.terms}
        if len(ps) > 1:
            raise ValueError("operator of mixed parity")
        return ps.pop()

    # -- arithmetic --------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Op):
            t = dict(self.terms)
            for b, c in other.terms.items():
                nc = _csimp(t.get(b, 0) + c)
                if nc == 0:
                    t.pop(b, None)
                else:
                    t[b] = nc
            r = Op()
            r.terms = t
            return r
        if other == 0:
            return self
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        r = Op()
        r.terms = {b: -c for b, c in self.terms.items()}
        return r

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, scalar):
        if isinstance(scalar, Op):
            raise TypeError("Use NO(A, B) for the normal-ordered product; "
                            "'*' only multiplies by scalars.")
        s = sp.sympify(scalar)
        t = {}
        for b, c in self.terms.items():
            nc = _csimp(c * s)
            if nc != 0:
                t[b] = nc
        r = Op()
        r.terms = t
        return r

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (S.One / sp.sympify(scalar))

    def d(self, n=1):
        """n-th derivative."""
        return d(self, n)

    # -- comparison --------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, Op):
            if other == 0:
                return self.is_zero
            return NotImplemented
        diff = self - other
        for c in diff.terms.values():
            if sp.simplify(c) != 0:
                return False
        return True

    def __hash__(self):
        raise TypeError("Op is not hashable")

    # -- inspection --------------------------------------------------
    def coefficient(self, other):
        """Coefficient of a single basis field `other` (an Op with one term)."""
        (b,) = other.terms.keys()
        return self.terms.get(b, S.Zero)

    def simplify(self, func=sp.expand):
        r = Op()
        r.terms = {}
        for b, c in self.terms.items():
            nc = func(c)
            if nc != 0:
                r.terms[b] = nc
        return r

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for b in sorted(self.terms, key=_sort_key):
            c = self.terms[b]
            cs = sp.sstr(c)
            if c == 1:
                s = repr(b)
            elif c == -1:
                s = "-" + repr(b)
            elif isinstance(c, sp.Add) or ("/" in cs and "(" not in cs and not c.is_Rational):
                s = "(%s)*%r" % (cs, b)
            elif isinstance(c, sp.Add):
                s = "(%s)*%r" % (cs, b)
            else:
                s = "%s*%r" % (cs, b)
            parts.append(s)
        out = parts[0]
        for p in parts[1:]:
            out += " - " + p[1:] if p.startswith("-") else " + " + p
        return out


# ----------------------------------------------------------------------
# Declarations
# ----------------------------------------------------------------------
def _declare(name, parity):
    if name in _state.names:
        raise ValueError("operator %r already declared "
                         "(use reset() to start over)" % name)
    op = _BasicOperator(name, parity, _state.counter)
    _state.counter += 1
    _state.names[name] = op
    _state.clear_caches()
    return Op._basis(_F(op, 0))


def bosonic(*names):
    """Declare bosonic operators (cf. Bosonic).  Returns Op(s)."""
    ops = [_declare(n, 0) for n in names]
    return ops[0] if len(ops) == 1 else tuple(ops)


def fermionic(*names):
    """Declare fermionic operators (cf. Fermionic).  Returns Op(s)."""
    ops = [_declare(n, 1) for n in names]
    return ops[0] if len(ops) == 1 else tuple(ops)


One = Op._basis(_ONE)


def _as_single_field(x, what):
    if not isinstance(x, Op) or len(x.terms) != 1:
        raise TypeError("%s must be a single declared operator" % what)
    ((b, c),) = x.terms.items()
    if not isinstance(b, _F) or b.n != 0 or c != 1:
        raise TypeError("%s must be an underived basic operator "
                        "with unit coefficient" % what)
    return b.op


def define_ope(A, B, poles):
    """Define the OPE of two *basic* operators (cf. OPE[A,B] = MakeOPE[{...}]).

    `poles` lists the operators at the poles from the HIGHEST order pole
    down to the first order pole (include the zero entries!), exactly as in
    the list form of MakeOPE.  Scalar entries are interpreted as multiples
    of the unit operator One.

    Examples:
        define_ope(T, T, [c/2*One, 0, 2*T, d(T)])           # Virasoro
        i, j = idx('i j')
        define_ope(J(i), J(j), [k*Delta(i, j)*One, 0])      # pattern rule,
                                                            # cf. OPE[J[i_],J[j_]]

    An index which is a bare SymPy Symbol acts as a *pattern* matching any
    index (the analogue of ``i_`` in the Mathematica package); concrete
    indices (numbers, strings, ...) define the OPE of those components only.
    """
    a = _as_single_field(A, "first argument")
    b = _as_single_field(B, "second argument")
    conv = []
    for p in poles:
        if isinstance(p, Op):
            conv.append(p)
        else:
            conv.append(One * sp.sympify(p))
    conv.reverse()               # store ascending: pole 1 first
    while conv and conv[-1].is_zero:
        conv.pop()

    def _pat(op):
        if op.family is not None and isinstance(sp.sympify(op.index), sp.Symbol):
            return op.family, sp.sympify(op.index)
        return op, None

    ka, pa = _pat(a)
    kb, pb = _pat(b)
    if pa is not None or pb is not None:
        _state.pattern_rules[(ka, kb)] = (pa, pb, tuple(conv))
    else:
        _state.ope_table[(a, b)] = tuple(conv)
    _state.clear_caches()


# ----------------------------------------------------------------------
# Pole-list helpers (internal representation: tuple/list of Op, ascending)
# ----------------------------------------------------------------------
def _pl_trim(pl):
    pl = list(pl)
    while pl and pl[-1].is_zero:
        pl.pop()
    return pl


def _pl_add(*pls):
    n = max((len(p) for p in pls), default=0)
    out = [Op() for _ in range(n)]
    for p in pls:
        for i, x in enumerate(p):
            out[i] = out[i] + x
    return _pl_trim(out)


def _pl_pole(pl, q):
    if 1 <= q <= len(pl):
        return pl[q - 1]
    return Op()


# ----------------------------------------------------------------------
# Ordering (cf. OPEOrder / NOOrder)
# ----------------------------------------------------------------------
def _no_order(x, y):
    """>0 if (x, y) is correctly ordered inside NO, 0 if tie, <0 otherwise.

    x, y are _F leaves (possibly derived)."""
    r = y.op.pos - x.op.pos
    if r != 0:
        return r
    if x.op is not y.op:
        # same indexed family, different index: canonical order by index key
        return 1 if _idx_key(x.op) < _idx_key(y.op) else -1
    return _state.no_ordering * (y.n - x.n)


# ----------------------------------------------------------------------
# The OPE engine  (eqs. (3.3.1)-(3.3.4), (3.3.12)-(3.3.19) of the thesis)
# ----------------------------------------------------------------------
def _ope_b(a, b):
    """OPE of two canonical basis elements -> tuple of Op (poles, ascending)."""
    key = (a, b)
    hit = _state.ope_cache.get(key)
    if hit is not None:
        return hit
    res = tuple(_pl_trim(_ope_b_compute(a, b)))
    _state.ope_cache[key] = res
    return res


def _ope_b_compute(a, b):
    # The unit operator is regular with everything.
    if isinstance(a, _OneB) or isinstance(b, _OneB):
        return ()

    # OPE(d^i A, B):  [d^i A  B]_{j+i} = (-1)^i (j)_i [AB]_j
    if isinstance(a, _F) and a.n > 0:
        i = a.n
        AB = _ope_b(_F(a.op, 0), b)
        m = len(AB)
        res = [Op() for _ in range(m + i)]
        for j in range(1, m + 1):
            res[j + i - 1] = (S.NegativeOne ** i * sp.rf(j, i)) * AB[j - 1]
        return res

    # OPE(A, d^i B):
    # [A d^i B]_j = sum_k C(i,k) (j-k)_k  d^{i-k} [AB]_{j-k}
    if isinstance(b, _F) and b.n > 0:
        i = b.n
        AB = _ope_b(a, _F(b.op, 0))
        m = len(AB)
        res = []
        for j in range(1, m + i + 1):
            acc = Op()
            for k in range(max(0, j - m), min(i, j - 1) + 1):
                acc = acc + (sp.binomial(i, k) * sp.rf(j - k, k)) * d(AB[j - k - 1], i - k)
            res.append(acc)
        return res

    # OPE(A, NO(B, C))
    if isinstance(b, _NOb):
        return _ope_comp_R(a, b.l, b.r)

    # OPE(NO(A, B), C) with B non-composite
    if isinstance(a, _NOb):
        if not isinstance(a.r, _NOb):
            return _ope_comp_L(a.l, a.r, b)
        # deeply-nested first argument: use the commutation formula
        return _ope_commuted(a, b)

    # both are underived basic fields
    if a.op.pos > b.op.pos:
        return _ope_commuted(a, b)
    return _lookup_basic(a, b)


def _commute_from(BA, s):
    """[AB]_q = s * sum_{l>=q} (-1)^l/(l-q)!  d^{l-q} [BA]_l   (eq. 3.3.3)."""
    mx = len(BA)
    res = []
    for q in range(1, mx + 1):
        acc = Op()
        for l in range(q, mx + 1):
            acc = acc + (S.NegativeOne ** l / sp.factorial(l - q)) * d(BA[l - 1], l - q)
        res.append(s * acc)
    return res


def _ope_commuted(a, b):
    return _commute_from(_ope_b(b, a), _swap_sign(a, b))


def _ope_of_expr_and_basis(X, C):
    """pole list of OPE(X, C), X an Op (linear in the first slot)."""
    parts = []
    for t, c in X.terms.items():
        pl = _ope_b(t, C)
        parts.append([c * p for p in pl])
    return _pl_add(*parts) if parts else []


def _ope_of_basis_and_expr(Bb, X):
    """pole list of OPE(Bb, X), X an Op (linear in the second slot)."""
    parts = []
    for t, c in X.terms.items():
        pl = _ope_b(Bb, t)
        parts.append([c * p for p in pl])
    return _pl_add(*parts) if parts else []


def _no_expr(X, Y):
    """NO(X, Y) for Op arguments (bilinear; scalar coefficients commute)."""
    acc = Op()
    for bx, cx in X.terms.items():
        for by, cy in Y.terms.items():
            acc = acc + (cx * cy) * _no_b(bx, by)
    return acc


def _ope_comp_R(A, B, C):
    """OPE(A, NO(B, C))    [eq. (3.3.13) / OPECompositeHelpRQ].

    [A [BC]_0]_q = (-1)^{|A||B|} [B [AC]_q]_0 + [[AB]_q C]_0
                   + sum_{l=1}^{q-1} C(q-1, l) [[AB]_{q-l} C]_l    (quantum)
    """
    s = _swap_sign(A, B)
    AB = _ope_b(A, B)
    AC = AB if B == C else _ope_b(A, C)
    mAB, mAC = len(AB), len(AC)
    OpB, OpC = Op._basis(B), Op._basis(C)

    if _state.method == CLASSICAL:
        res = []
        for q in range(1, max(mAB, mAC) + 1):
            res.append(s * _no_expr(OpB, _pl_pole(AC, q))
                       + _no_expr(_pl_pole(AB, q), OpC))
        return res

    ABC = [_ope_of_expr_and_basis(AB[m - 1], C) for m in range(1, mAB + 1)]
    maxq = max([mAC, mAB] + [len(ABC[m - 1]) + m for m in range(1, mAB + 1)] + [0])
    res = []
    for q in range(1, maxq + 1):
        acc = s * _no_expr(OpB, _pl_pole(AC, q)) + _no_expr(_pl_pole(AB, q), OpC)
        for l in range(max(1, q - mAB), q):
            m = q - l
            acc = acc + sp.binomial(q - 1, l) * _pl_pole(ABC[m - 1], l)
        res.append(acc)
    return res


def _ope_comp_L(A, B, C):
    """OPE(NO(A, B), C)    [eq. (3.3.18) / OPECompositeHelpLQ].

    [[AB]_0 C]_q = sum_{l>=0} 1/l! [d^l A [BC]_{l+q}]_0
                 + (-1)^{|A||B|} sum_{l>=0} 1/l! [d^l B [AC]_{l+q}]_0
                 + (-1)^{|A||B|} sum_{l=1}^{q-1} [B [AC]_{q-l}]_l    (quantum)
    """
    s = _swap_sign(A, B)
    AC = _ope_b(A, C)
    BC = AC if A == B else _ope_b(B, C)
    mAC, mBC = len(AC), len(BC)

    p1 = []
    for q in range(1, mBC + 1):
        acc = Op()
        for l in range(0, mBC - q + 1):
            acc = acc + _no_expr(Op._basis(_F(A.op, A.n + l)), BC[l + q - 1]) \
                / sp.factorial(l)
        p1.append(acc)

    p2 = []
    for q in range(1, mAC + 1):
        acc = Op()
        for l in range(0, mAC - q + 1):
            acc = acc + _no_expr(Op._basis(_F(B.op, B.n + l)), AC[l + q - 1]) \
                / sp.factorial(l)
        p2.append(s * acc)

    if _state.method == CLASSICAL:
        return _pl_add(p1, p2)

    BAC = [_ope_of_basis_and_expr(B, AC[m - 1]) for m in range(1, mAC + 1)]
    maxq3 = max([len(BAC[m - 1]) + m for m in range(1, mAC + 1)] + [0])
    p3 = []
    for q in range(1, maxq3 + 1):
        acc = Op()
        for l in range(max(1, q - mAC), q):
            m = q - l
            acc = acc + _pl_pole(BAC[m - 1], l)
        p3.append(s * acc)

    return _pl_add(p1, p2, p3)


# ----------------------------------------------------------------------
# Normal ordering  (eqs. (3.3.5)-(3.3.11))
# ----------------------------------------------------------------------
def _no_commute(x, y):
    """NO(x,y) - (-1)^{|x||y|} NO(y,x) = -sum_{m>=1} (-1)^m/m! d^m [xy]_m."""
    if _state.method == CLASSICAL:
        return Op()
    XY = _ope_b(x, y)
    acc = Op()
    for m in range(1, len(XY) + 1):
        acc = acc + (S.NegativeOne ** (m + 1) / sp.factorial(m)) * d(XY[m - 1], m)
    return acc


def _no_comp_R(Bb, A, C):
    """NO(Bb, NO(A, C)) with A to be moved left  [NOCompositeHelpR]."""
    s = _swap_sign(A, Bb)
    inner = _no_expr(Op._basis(A), _no_b(Bb, C))
    if _state.method == CLASSICAL:
        return s * inner
    return s * inner + _no_expr(_no_commute(Bb, A), Op._basis(C))


def _no_comp_L(A, B, Cb):
    """NO(NO(A, B), Cb)  [eq. (3.3.19) / NOCompositeHelpLQ]."""
    s = _swap_sign(A, B)
    first = _no_expr(Op._basis(A), _no_b(B, Cb))
    if _state.method == CLASSICAL:
        return first
    AC = _ope_b(A, Cb)
    BC = AC if A == B else _ope_b(B, Cb)
    acc = first
    for l in range(1, len(BC) + 1):
        acc = acc + _no_expr(Op._basis(_F(A.op, A.n + l)), BC[l - 1]) / sp.factorial(l)
    for l in range(1, len(AC) + 1):
        acc = acc + s * _no_expr(Op._basis(_F(B.op, B.n + l)), AC[l - 1]) / sp.factorial(l)
    return acc


def _no_b(a, b):
    key = (a, b)
    hit = _state.no_cache.get(key)
    if hit is not None:
        return hit
    res = _no_b_compute(a, b)
    _state.no_cache[key] = res
    return res


def _no_b_compute(a, b):
    if isinstance(a, _OneB):
        return Op._basis(b)
    if isinstance(b, _OneB):
        return Op._basis(a)

    if isinstance(a, _NOb):
        A, B2 = a.l, a.r
        if isinstance(b, _NOb):
            C, D = b.l, b.r
            if _no_order(A, C) <= 0:
                return _no_comp_R(a, C, D)
            return _no_comp_L(A, B2, b)
        # NO(NO(A,B2), b) with b a leaf
        if _no_order(A, b) > 0:
            return _no_comp_L(A, B2, b)
        s = _swap_sign(b, a)
        return s * (_no_b(b, a) - _no_commute(b, a))

    # a is a leaf
    if isinstance(b, _NOb):
        A2, C2 = b.l, b.r
        if a == A2 and (a.parity % 2) == 1:
            # NO(A, NO(A, C)) = 1/2 NO(NOCommute(A,A), C)  for fermionic A
            return Rational(1, 2) * _no_expr(_no_commute(a, a), Op._basis(C2))
        if _no_order(A2, a) > 0:
            return _no_comp_R(a, A2, C2)
        return Op._basis(_NOb(a, b))

    # both leaves
    if a == b and (a.parity % 2) == 1:
        return Rational(1, 2) * _no_commute(a, a)
    if _no_order(b, a) > 0:
        s = _swap_sign(b, a)
        return s * (_no_b(b, a) - _no_commute(b, a))
    return Op._basis(_NOb(a, b))


def NO(*args):
    """Normal-ordered product (point splitting), reduced to standard form.

    NO(A, B, C) == NO(A, NO(B, C)).  Arguments are Ops (multi-linear)."""
    if len(args) < 2:
        raise TypeError("NO needs at least two arguments")
    if len(args) > 2:
        return NO(args[0], NO(*args[1:]))
    X, Y = args
    if not isinstance(X, Op) or not isinstance(Y, Op):
        raise TypeError("NO arguments must be Ops")
    acc = Op()
    for bind1, b1, c1 in _open_terms(X):
        for bind2, b2, c2 in _open_terms(Y):
            piece = (c1 * c2) * _no_b(b1, b2)
            binders = bind1 + bind2
            if binders:
                piece = _rebind(piece, binders)
            acc = acc + piece
    return acc


# ----------------------------------------------------------------------
# Derivative
# ----------------------------------------------------------------------
def _d_basis(b):
    if isinstance(b, _OneB):
        return Op()
    if isinstance(b, _F):
        return Op._basis(_F(b.op, b.n + 1))
    if isinstance(b, _ISum):
        # d commutes with the index sum
        return _map_under_sum(b, lambda X: d(X, 1))
    # NO(l, r):  Leibniz, then re-canonicalise
    return _no_expr(_d_basis(b.l), Op._basis(b.r)) \
        + _no_expr(Op._basis(b.l), _d_basis(b.r))


def d(X, n=1):
    """n-th derivative of an operator expression."""
    if not isinstance(X, Op):
        raise TypeError("d() takes an Op")
    if n < 0:
        raise ValueError("n must be >= 0")
    for _ in range(n):
        acc = Op()
        for b, c in X.terms.items():
            acc = acc + c * _d_basis(b)
        X = acc
    return X


# ----------------------------------------------------------------------
# OPEData : user-facing OPE result
# ----------------------------------------------------------------------
class OPEData:
    """The list of pole parts of an OPE.  pole(q) is the operator at 1/(z-w)^q."""
    __slots__ = ("_poles",)

    def __init__(self, poles_ascending):
        self._poles = tuple(_pl_trim(list(poles_ascending)))

    @property
    def max_pole(self):
        return len(self._poles)

    def pole(self, q):
        return _pl_pole(self._poles, q)

    def poles(self):
        """List of poles from highest down to first order (MakeOPE order)."""
        return list(reversed(self._poles))

    def __add__(self, other):
        if isinstance(other, OPEData):
            return OPEData(_pl_add(self._poles, other._poles))
        if other == 0:
            return self
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, scalar):
        return OPEData([scalar * p for p in self._poles])

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def simplify(self, func=sp.expand):
        return OPEData([p.simplify(func) for p in self._poles])

    def map(self, f):
        """Apply f to every pole (cf. OPEMap)."""
        return OPEData([f(p) for p in self._poles])

    def __eq__(self, other):
        if not isinstance(other, OPEData):
            return NotImplemented
        top = max(self.max_pole, other.max_pole)
        return all(self.pole(q) == other.pole(q) for q in range(1, top + 1))

    @property
    def is_zero(self):
        return not self._poles

    def __repr__(self):
        if not self._poles:
            return "<< regular >>"
        bits = []
        for q in range(self.max_pole, 0, -1):
            bits.append("%d|| %r" % (q, self.pole(q)))
        return "<< " + " ||".join(bits) + " >>"

    def series_str(self, z="z", w="w"):
        """Human-readable series form  A(z)B(w) = ... + O(1)."""
        if not self._poles:
            return "O((%s-%s)^0)" % (z, w)
        parts = []
        for q in range(self.max_pole, 0, -1):
            p = self.pole(q)
            if p.is_zero:
                continue
            parts.append("(%r)/(%s-%s)^%d" % (p, z, w, q))
        parts.append("O((%s-%s)^0)" % (z, w))
        return " + ".join(parts)


def OPE(X, Y):
    """OPE of two operator expressions -> OPEData (bilinear; index sums are
    expanded, computed, and re-summed with automatic Delta contraction)."""
    if not isinstance(X, Op) or not isinstance(Y, Op):
        raise TypeError("OPE arguments must be Ops")
    parts = []
    for bind1, b1, c1 in _open_terms(X):
        for bind2, b2, c2 in _open_terms(Y):
            pl = [(c1 * c2) * p for p in _ope_b(b1, b2)]
            binders = bind1 + bind2
            if binders:
                pl = [_rebind(p, binders) for p in pl]
            parts.append(pl)
    return OPEData(_pl_add(*parts) if parts else [])


def OPEPole(q, X, Y):
    """The operator at the q-th order pole of OPE(X, Y).

    q > 0 : pole of order q;   q == 0 : NO(X, Y);
    q < 0 : coefficient in the regular part, 1/(-q)! NO(d^{-q}X, Y)."""
    if q > 0:
        return OPE(X, Y).pole(q)
    if q == 0:
        return NO(X, Y)
    return NO(d(X, -q), Y) / sp.factorial(-q)


def MaxPole(ope):
    return ope.max_pole


def OPESimplify(x, func=sp.expand):
    """Collect equal operators and apply `func` to their coefficients."""
    if isinstance(x, OPEData):
        return x.simplify(func)
    if isinstance(x, Op):
        return x.simplify(func)
    if isinstance(x, (list, tuple)):
        return type(x)(OPESimplify(e, func) for e in x)
    raise TypeError


def GetOperators(x):
    """List of basis fields occurring in an Op or OPEData (as Ops)."""
    if isinstance(x, OPEData):
        out = []
        for p in x._poles:
            out.extend(GetOperators(p))
        seen, uniq = set(), []
        for o in out:
            (b,) = o.terms.keys()
            if b not in seen:
                seen.add(b)
                uniq.append(o)
        return uniq
    return [Op._basis(b) for b in sorted(x.terms, key=_sort_key)]


def GetCoefficients(x):
    """List of the coefficients of all fields in an Op or OPEData."""
    if isinstance(x, OPEData):
        out = []
        for p in x._poles:
            out.extend(GetCoefficients(p))
        return out
    return [x.terms[b] for b in sorted(x.terms, key=_sort_key)]


# ----------------------------------------------------------------------
# Jacobi identities (associativity check; cf. OPEJacobi)
# ----------------------------------------------------------------------
def OPEJacobi(A, B, C, func=sp.expand):
    """Return the matrix J[m][n] of Jacobi combinations

        J(m,n) = [A [BC]_m]_n - (-1)^{|A||B|} [B [AC]_n]_m
                 - sum_{p=1}^{n} C(n-1, p-1) [[AB]_p C]_{m+n-p}

    (m, n >= 1).  All entries vanish iff the OPEs are associative
    ("Jacobi identities" of the operator algebra)."""
    sign = S.NegativeOne ** (A.parity() * B.parity())
    AB, AC, BC = OPE(A, B), OPE(A, C), OPE(B, C)
    AnBC = {m: OPE(A, BC.pole(m)) for m in range(1, BC.max_pole + 1)}
    BnAC = {n: OPE(B, AC.pole(n)) for n in range(1, AC.max_pole + 1)}
    ABnC = {p: OPE(AB.pole(p), C) for p in range(1, AB.max_pole + 1)}

    def mp(dd):
        return max([v.max_pole for v in dd.values()] + [0])

    maxn = max(mp(AnBC), AC.max_pole, AB.max_pole)
    maxm = max(BC.max_pole, mp(BnAC), mp(ABnC))
    J = []
    for m in range(1, maxm + 1):
        row = []
        for n in range(1, maxn + 1):
            acc = AnBC.get(m, OPEData([])).pole(n) \
                - sign * BnAC.get(n, OPEData([])).pole(m)
            for p in range(1, n + 1):
                acc = acc - sp.binomial(n - 1, p - 1) \
                    * ABnC.get(p, OPEData([])).pole(m + n - p)
            row.append(acc.simplify(func))
        J.append(row)
    return J


def jacobi_satisfied(A, B, C):
    """True iff all Jacobi combinations of (A, B, C) vanish."""
    return all(entry == Op() for row in OPEJacobi(A, B, C) for entry in row)


__all__.append("jacobi_satisfied")


# ======================================================================
# Indexed operator families, pattern OPE rules, Delta symbol, dummy sums
# (the analogue of  Bosonic[J[i_]],  the Delta` package and  Dummies`)
# ======================================================================

#: The (symmetric) Kronecker delta symbol, cf. Delta[i,j] of the Delta`
#: package.  Delta(i, i) == 1, Delta(1, 2) == 0, Delta(i, j) stays symbolic.
Delta = sp.KroneckerDelta

#: Totally antisymmetric symbol (useful for structure constants).
Eps = sp.LeviCivita


def idx(names):
    """Convenience: idx('i j k') -> SymPy symbols usable as indices/patterns."""
    return sp.symbols(names)


class _Family:
    """An indexed family of basic operators, e.g. J(i) (cf. J[i_])."""
    __slots__ = ("name", "parity", "pos")

    def __init__(self, name, parity, pos):
        self.name = name
        self.parity = parity
        self.pos = pos

    def __repr__(self):
        return "%s(_)" % self.name


class IndexedFamily:
    """User-facing handle: calling it with an index returns the component Op."""
    __slots__ = ("_fam",)

    def __init__(self, fam):
        self._fam = fam

    @property
    def name(self):
        return self._fam.name

    def __call__(self, index):
        op = _family_op(self._fam, sp.sympify(index))
        return Op._basis(_F(op, 0))

    def __repr__(self):
        return "%s(_)" % self._fam.name


def _family_op(fam, index):
    key = (fam.pos, index)
    op = _state.family_ops.get(key)
    if op is None:
        op = _BasicOperator("%s(%s)" % (fam.name, index), fam.parity,
                            fam.pos, family=fam, index=index)
        _state.family_ops[key] = op
    return op


def _declare_family(name, parity):
    if name in _state.names:
        raise ValueError("operator %r already declared "
                         "(use reset() to start over)" % name)
    fam = _Family(name, parity, _state.counter)
    _state.counter += 1
    _state.names[name] = fam
    _state.clear_caches()
    return IndexedFamily(fam)


def bosonic_family(*names):
    """Declare indexed bosonic families (cf. Bosonic[J[_]]).

    >>> J = bosonic_family('J');  J(1), J(sp.Symbol('a'))
    """
    fams = [_declare_family(n, 0) for n in names]
    return fams[0] if len(fams) == 1 else tuple(fams)


def fermionic_family(*names):
    """Declare indexed fermionic families (cf. Fermionic[psi[_]])."""
    fams = [_declare_family(n, 1) for n in names]
    return fams[0] if len(fams) == 1 else tuple(fams)


# ----------------------------------------------------------------------
# Index substitution
# ----------------------------------------------------------------------
def _subs_basis(b, m):
    """Substitute index symbols in a basis element; returns an Op
    (substitution can trigger re-canonicalisation of NO products)."""
    if not m:
        return Op._basis(b)
    if isinstance(b, _OneB):
        return One
    if isinstance(b, _F):
        if b.op.family is None:
            return Op._basis(b)
        new = sp.sympify(b.op.index).subs(m)
        if new == b.op.index:
            return Op._basis(b)
        return Op._basis(_F(_family_op(b.op.family, new), b.n))
    if isinstance(b, _NOb):
        return _no_expr(_subs_basis(b.l, m), _subs_basis(b.r, m))
    # _ISum : substitute in the summand, avoiding capture of the bound symbol
    bsym = _BOUND(b.depth)
    m2 = {k: v for k, v in m.items() if k != bsym}
    fresh = _fresh_sym()
    m2[bsym] = fresh
    acc = Op()
    for ib, ic in b.inner:
        acc = acc + sp.sympify(ic).subs(m2) * _subs_basis(ib, m2)
    return dsum(acc, fresh, sp.sympify(b.dim).subs({k: v for k, v in m.items()
                                                    if k != bsym}))


def _subs_op(X, m):
    acc = Op()
    for b, c in X.terms.items():
        acc = acc + sp.sympify(c).subs(m) * _subs_basis(b, m)
    return acc


# ----------------------------------------------------------------------
# OPE lookup with pattern rules
# ----------------------------------------------------------------------
def _match_rule(aop, bop):
    ka = aop.family if aop.family is not None else aop
    kb = bop.family if bop.family is not None else bop
    rule = _state.pattern_rules.get((ka, kb))
    if rule is None:
        return None
    pa, pb, poles = rule
    m = {}
    if pa is not None:
        m[pa] = sp.sympify(aop.index)
    if pb is not None:
        if pa is not None and pb == pa:
            if sp.sympify(aop.index) != sp.sympify(bop.index):
                return None          # diagonal-only rule J(i), J(i)
        else:
            m[pb] = sp.sympify(bop.index)
    return [_subs_op(p, m) for p in poles]


def _lookup_basic(a, b):
    """OPE of two underived basic fields via the tables (patterns included)."""
    t = _state.ope_table.get((a.op, b.op))
    if t is not None:
        return t
    r = _match_rule(a.op, b.op)
    if r is not None:
        return r
    s = _swap_sign(a, b)
    t = _state.ope_table.get((b.op, a.op))
    if t is not None:
        return _commute_from(t, s)
    r = _match_rule(b.op, a.op)
    if r is not None:
        return _commute_from(r, s)
    # all non-defined OPEs of basic fields are regular
    return ()


# ----------------------------------------------------------------------
# Formal sums over indices (dummy indices; the analogue of Dummies`)
# ----------------------------------------------------------------------
def _BOUND(k):
    return sp.Symbol('_d%d' % k)


def _fresh_sym():
    _state.dummy_counter += 1
    return sp.Symbol('_x%d' % _state.dummy_counter)


class _ISum(_Basis):
    """Formal sum over an index:  sum_{s=1..dim} (summand);  the bound
    symbol is the reserved symbol _d<depth>."""
    __slots__ = ("dim", "inner", "depth", "parity", "_h")

    def __init__(self, dim, inner, depth):
        self.dim = dim
        self.inner = inner            # tuple of (basis, coeff), sorted
        self.depth = depth
        ps = {(_parity(ib) % 2) for ib, _ in inner}
        if len(ps) > 1:
            raise ValueError("index sum over operators of mixed parity")
        self.parity = ps.pop() if ps else 0
        self._h = hash(("ISum", dim, depth, inner))

    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return (isinstance(other, _ISum) and other.dim == self.dim and
                other.depth == self.depth and other.inner == self.inner)

    def __repr__(self):
        body = Op()
        body.terms = dict(self.inner)
        return "Sum[%s=1..%s]{ %r }" % (_BOUND(self.depth), self.dim, body)


def _isum_depth(b):
    if isinstance(b, _ISum):
        return b.depth
    if isinstance(b, _NOb):
        return max(_isum_depth(b.l), _isum_depth(b.r))
    return 0


def _basis_free_symbols(b):
    if isinstance(b, _OneB):
        return set()
    if isinstance(b, _F):
        if b.op.index is None:
            return set()
        return set(sp.sympify(b.op.index).free_symbols)
    if isinstance(b, _NOb):
        return _basis_free_symbols(b.l) | _basis_free_symbols(b.r)
    # _ISum
    out = set(sp.sympify(b.dim).free_symbols)
    for ib, ic in b.inner:
        out |= _basis_free_symbols(ib) | set(sp.sympify(ic).free_symbols)
    out.discard(_BOUND(b.depth))
    return out


def dsum(X, sym, rng):
    """Sum an operator expression over an index (cf. SumDummy of Dummies`).

    * ``rng`` an integer (or list of index values): the sum is expanded
      explicitly, e.g. ``dsum(NO(J(a), J(a)), a, 3)``.
    * ``rng`` a symbol (symbolic dimension N): the sum is kept formal,
      with automatic contraction of Kronecker deltas,
          sum_a Delta(a, b) X(a) = X(b),      sum_a Delta(a, a) = N,
      and terms independent of the index are multiplied by N.
      Free indices appearing elsewhere are assumed to lie in the range.
    """
    if not isinstance(X, Op):
        raise TypeError("dsum takes an Op")
    sym = sp.sympify(sym)
    if not isinstance(sym, sp.Symbol):
        raise TypeError("the summation index must be a Symbol")
    if isinstance(rng, (list, tuple, range)):
        acc = Op()
        for v in rng:
            acc = acc + _subs_op(X, {sym: sp.sympify(v)})
        return acc
    rng = sp.sympify(rng)
    if rng.is_Integer:
        return dsum(X, sym, list(range(1, int(rng) + 1)))
    return _formal_sum(X, sym, rng)


def _formal_sum(X, sym, dim):
    out = Op()
    bound = []                                   # (basis, coeff-piece)
    for b, c in X.terms.items():
        for piece in sp.Add.make_args(sp.expand(c)):
            dl = None
            for cand in piece.atoms(sp.KroneckerDelta):
                if sym in cand.args:
                    other = cand.args[1] if cand.args[0] == sym else cand.args[0]
                    if sym not in sp.sympify(other).free_symbols:
                        dl, oth = cand, other
                        break
            if dl is not None:
                out = out + (piece / dl).subs(sym, oth) * _subs_basis(b, {sym: oth})
            elif (sym not in piece.free_symbols
                  and sym not in _basis_free_symbols(b)):
                out = out + (dim * piece) * Op._basis(b)
            else:
                bound.append((b, piece))
    if not bound:
        return out
    depth = 1 + max(_isum_depth(b) for b, _ in bound)
    bsym = _BOUND(depth)
    if bsym != sym:
        S = Op()
        for b, piece in bound:
            S = S + sp.sympify(piece).subs(sym, bsym) * _subs_basis(b, {sym: bsym})
        # renaming can re-canonicalise; re-run (now sym == bsym, terminates)
        return out + _formal_sum(S, bsym, dim)
    S = Op()
    for b, piece in bound:
        S = S + piece * Op._basis(b)
    items = sorted(S.terms.items(), key=lambda t: _sort_key(t[0]))
    if not items:
        return out
    # canonical form: scalar multiples of the same sum must coincide, so
    # pull out the part of the leading coefficient that is independent of
    # the bound symbol:  sum_a (2 f(a) X_a) == 2 * sum_a f(a) X_a
    lead = sp.sympify(items[0][1]).as_independent(sym, as_Add=False)[0]
    if lead == 0:
        lead = S.One
    frozen = tuple((b, _csimp(c / lead)) for b, c in items)
    return out + lead * Op._basis(_ISum(dim, frozen, depth))


def _open_term(b, c):
    """Expand top-level index sums: -> list of (binders, basis, coeff),
    binders = [(fresh_symbol, dim), ...] innermost first."""
    if not isinstance(b, _ISum):
        return [([], b, c)]
    fresh = _fresh_sym()
    bsym = _BOUND(b.depth)
    res = []
    for ib, ic in b.inner:
        sub = (c * sp.sympify(ic).subs(bsym, fresh)) * _subs_basis(ib, {bsym: fresh})
        for nb, nc in sub.terms.items():
            for binds, b2, c2 in _open_term(nb, nc):
                res.append((binds + [(fresh, b.dim)], b2, c2))
    return res


def _open_terms(X):
    out = []
    for b, c in X.terms.items():
        out.extend(_open_term(b, c))
    return out


def _rebind(P, binders):
    for s, dim in binders:
        P = dsum(P, s, dim)
    return P


def _map_under_sum(b, f):
    """Apply an Op->Op linear map under the top-level sum of an _ISum."""
    fresh = _fresh_sym()
    bsym = _BOUND(b.depth)
    X = Op()
    for ib, ic in b.inner:
        X = X + sp.sympify(ic).subs(bsym, fresh) * _subs_basis(ib, {bsym: fresh})
    return dsum(f(X), fresh, b.dim)


__all__ += ["bosonic_family", "fermionic_family", "dsum", "Delta", "Eps", "idx"]
