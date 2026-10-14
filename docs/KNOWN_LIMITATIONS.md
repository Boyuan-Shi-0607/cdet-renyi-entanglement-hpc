# Known limitations of the unchanged production core

This migration does not silently repair the numerical implementation.
The historical findings in `tests/AUDIT_REPORT.md` remain relevant; those
results are dated evidence, not proof of all inputs/platforms.

1. The binary looks for `../params.txt` before `params.txt`. The migrated
   chunk/example launchers guard against shadowing, but direct binary or
   legacy PBS use requires the same check by the caller.
2. Chunk validation checks the listed files, not a general proof of TSV
   coverage, non-overlap and destination bounds. Use freshly generated,
   unmodified manifests; production bounds/coverage hardening remains open.
3. The chunk identity does not include every precision setting, including
   `GREEN_TABLE_DROP_BITS`. Do not reuse tables under changed parameters.
4. Long-double non-LAPACK fallback tolerances are not uniformly type-aware.
   Double load data with LAPACK and long-double integrands are distinct
   choices; do not interpret all-long-double as automatically more accurate.
5. The parser permits problematic branch-epsilon values and an envelope
   range wider than some mask helpers. Restrict experiments to tested orders
   and sensible positive corner tolerances until these checks are hardened.
6. Runtime caches assume one immutable parameter set per process.
7. The legacy per-order printed error ignores numerator/denominator
   covariance. Publication analysis must use whole-chain resampling and
   retain cross-order covariance. Final checkpoints are cumulative summaries,
   not a full trajectory for autocorrelation/thermalization diagnostics.
8. Statistical tests at short chain lengths can fluctuate. Passing them does
   not prove asymptotic convergence, absence of a sign problem, or accuracy
   of a finite-order resummation.

The three DIAGONAL_SIMPSON settings found in older parameter files are not
used by this core. `TOL_FPM` controls determinant fallback; the z-transform
safety settings can filter small connected terms. They are not interchangeable
convergence controls. See the precision table in the historical audit.
