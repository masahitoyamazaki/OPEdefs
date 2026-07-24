# Cross-check against the original Mathematica OPEdefs

Battery of 27 computations (110 compared items: max pole orders + every
individual pole) spanning: Virasoro (incl. NO[T,T], quasiprimary Lambda,
Lambda x Lambda), free-boson Sugawara, free fermion, bc ghosts, su(2)_k,
the N=2 superconformal algebra, and Zamolodchikov W_3 with
beta = 16/(22+5c) -- all with symbolic c, k.

## How to run

1. On a machine with Mathematica and the original `OPEdefs.m`:

       wolframscript -file mma_crosscheck.wls /path/to/OPEdefs.m

   (or, inside a Mathematica session in this directory:
   `$ScriptCommandLine={"","OPEdefs.m"}; Get["mma_crosscheck.wls"]`)
   This writes `mma_results.txt`.

2. Then:

       python py_crosscheck.py mma_results.txt

   Every pole of every case is re-evaluated through the pyOPEdefs
   constructors and compared with `Op.__eq__`, so the comparison is
   independent of either package's normal-ordering display convention
   and of how coefficients happen to be written.

## Pipeline self-test (no Mathematica required)

       python py_crosscheck.py --selftest

serializes pyOPEdefs' own results in Mathematica InputForm style, then
round-trips them through the parser and comparator. This validates the
translation layer (Derivative[n][X], primes, `^` vs `**`, exact
rationals) but NOT the physics -- for that, run the real Mathematica side.

## Notes

* Operator names avoid collisions with coefficient symbols: the ghosts
  are `bb`, `cg` (not `b`, `c`), since `c` is the central charge.
* Indexed-family/Delta computations are not part of this battery: they
  would require the separate Delta`/Dummies` packages on the Mathematica
  side. Their Python implementation is instead validated in
  `test_opedefs.py` against exact CFT results (c = N, c = 3k/(k+2), ...).
* If a mismatch is reported, the failing case name + pole is printed;
  compare `OPESimplify[OPEPole[q][...], Factor]` by hand on both sides.
