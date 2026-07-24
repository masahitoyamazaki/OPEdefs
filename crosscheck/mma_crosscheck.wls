(* ::Package:: *)
(* mma_crosscheck.wls -- run the reference battery with the ORIGINAL
   Mathematica OPEdefs and write machine-readable results.

   Usage:   wolframscript -file mma_crosscheck.wls [path/to/OPEdefs.m]
   Output:  mma_results.txt   (lines:  CASE|name|q|expr   and  MAX|name|n)

   The expression strings are written so that the Python comparator can
   evaluate them directly (InputForm, with NO[..] and Derivative[n][X]). *)

pkg = If[Length[$ScriptCommandLine] >= 2, $ScriptCommandLine[[2]], "OPEdefs.m"];
Get[pkg];

out = OpenWrite["mma_results.txt"];
emit[name_, ope_] := Module[{n = MaxPole[ope], q, e},
  WriteString[out, "MAX|", name, "|", ToString[n], "\n"];
  Do[
    e = OPESimplify[OPEPole[q][ope], Factor];
    WriteString[out, "CASE|", name, "|", ToString[q], "|",
      ToString[InputForm[e]], "\n"],
    {q, 1, n}]];
emitop[name_, expr_] := WriteString[out, "OP|", name, "|",
  ToString[InputForm[OPESimplify[expr, Factor]]], "\n"];

(* ---------- 1. Virasoro, symbolic c ---------- *)
Bosonic[T];
OPE[T, T] = MakeOPE[{c/2 One, 0, 2 T, T'}];
emit["vir.TT", OPE[T, T]];
emit["vir.T_TT", OPE[T, NO[T, T]]];
Lam = NO[T, T] - 3/10 T'';
emit["vir.T_Lam", OPE[T, Lam]];
emit["vir.Lam_Lam", OPE[Lam, Lam]];
emit["vir.T_dT", OPE[T, T']];
emitop["vir.NOTT", NO[T, T]];

(* ---------- 2. free boson Sugawara ---------- *)
Bosonic[J];
OPE[J, J] = MakeOPE[{One, 0}];
Tfb = NO[J, J]/2;
emit["fb.T_J", OPE[Tfb, J]];
emit["fb.J_T", OPE[J, Tfb]];
emit["fb.T_T", OPE[Tfb, Tfb]];
emitop["fb.NOassoc", NO[NO[J, J], J] - NO[J, NO[J, J]]];

(* ---------- 3. free fermion ---------- *)
Fermionic[psi];
OPE[psi, psi] = MakeOPE[{One}];
Tff = -NO[psi, psi']/2;
emit["ff.T_psi", OPE[Tff, psi]];
emit["ff.T_T", OPE[Tff, Tff]];
emitop["ff.NOpsipsi", NO[psi, psi]];

(* ---------- 4. bc ghosts (lambda = 2) ---------- *)
Fermionic[bb, cg];
OPE[bb, cg] = MakeOPE[{One}];
Tgh = -2 NO[bb, cg'] - NO[bb', cg];
emit["bc.T_b", OPE[Tgh, bb]];
emit["bc.T_c", OPE[Tgh, cg]];
emit["bc.T_T", OPE[Tgh, Tgh]];
emitop["bc.NOrev", NO[cg, bb]];

(* ---------- 5. su(2)_k, complex basis ---------- *)
Bosonic[J3, Jp, Jm];
OPE[J3, J3] = MakeOPE[{k/2 One, 0}];
OPE[J3, Jp] = MakeOPE[{Jp}];
OPE[J3, Jm] = MakeOPE[{-Jm}];
OPE[Jp, Jm] = MakeOPE[{k One, 2 J3}];
Tsu = (NO[J3, J3] + (NO[Jp, Jm] + NO[Jm, Jp])/2)/(k + 2);
emit["su2.T_J3", OPE[Tsu, J3]];
emit["su2.T_Jp", OPE[Tsu, Jp]];
emit["su2.T_T", OPE[Tsu, Tsu]];

(* ---------- 6. N=2 superconformal ---------- *)
Bosonic[TT2, JJ2]; Fermionic[Gp, Gm];
OPE[TT2, TT2] = MakeOPE[{c/2 One, 0, 2 TT2, TT2'}];
OPE[TT2, JJ2] = MakeOPE[{JJ2, JJ2'}];
OPE[TT2, Gp] = MakeOPE[{3/2 Gp, Gp'}];
OPE[TT2, Gm] = MakeOPE[{3/2 Gm, Gm'}];
OPE[JJ2, JJ2] = MakeOPE[{c/3 One, 0}];
OPE[JJ2, Gp] = MakeOPE[{Gp}];
OPE[JJ2, Gm] = MakeOPE[{-Gm}];
OPE[Gp, Gm] = MakeOPE[{2 c/3 One, 2 JJ2, 2 TT2 + JJ2'}];
emit["n2.GmGp", OPE[Gm, Gp]];
emit["n2.T_Gp", OPE[TT2, Gp]];
emit["n2.Gp_T", OPE[Gp, TT2]];

(* ---------- 7. Zamolodchikov W3 ---------- *)
Bosonic[TW, W];
OPE[TW, TW] = MakeOPE[{c/2 One, 0, 2 TW, TW'}];
OPE[TW, W] = MakeOPE[{3 W, W'}];
LamW = NO[TW, TW] - 3/10 TW'';
bet = 16/(22 + 5 c);
OPE[W, W] = MakeOPE[{c/3 One, 0, 2 TW, TW',
   2 bet LamW + 3/10 TW'', bet LamW' + 1/15 Derivative[3][TW]}];
emit["w3.T_W", OPE[TW, W]];
emit["w3.W_W", OPE[W, W]];
emit["w3.W_dW", OPE[W, W']];
emit["w3.T_Lam", OPE[TW, LamW]];

Close[out];
Print["wrote mma_results.txt"];
