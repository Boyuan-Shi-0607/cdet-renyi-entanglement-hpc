# CDet numerical implementation audit

Audit target:

- branch: `hpc_version_mem_chunks_mcmcmc_optimized_`
- commit inspected: `2532ac2e19bd7d62826e6a384f400edffb7e1913`
- production files were read only; all new code is under `tests/`

## Executive result

The implemented second-Renyi graded-replica Green function, stable diagonal
scaling, twisted-minus-two-ordinary normalization, connected-determinant
recursion, and set heat-bath MCMCMC estimator agree with the theory notes.
The tests found no numerical overflow, out-of-bounds access, undefined
behavior, missing sector, or resident/chunk disagreement on valid generated
manifests.

The strongest end-to-end checks are:

1. Stable sector kernels agree with the direct unscaled theoretical formula,
   satisfy the equal-time jump and graded temporal boundary condition, and
   remain finite at beta 200.
2. A deliberately awkward non-square `3x2` chunk build covers all four table
   kinds, both triangle branches, all four replica-index pairs, all sites, and
   all Chebyshev nodes. Every loaded coefficient and corner agrees with a
   direct resident build. The chunked and resident order-1 reductions agree.
3. Every all-subset hot-path value for five six-vertex envelopes agrees with a
   fresh integrand evaluation at the subset's actual order.
4. Fast principal minors agree with pivoted determinants, including forced
   fallback cases. CDet recursion agrees with an independent set-partition
   cumulant through order seven.
5. Synthetic set heat-bath chains with variable retained order recover known
   equal integrals at orders 1 through 4 for both uniform and histogram seeds.
6. ASan and UBSan report no problem in the full audit suite.

There are several hardening gaps described below. The most important is that
the production chunk validator validates files listed in the TSV manifests,
but does not itself prove that those TSV ranges form an exact, non-overlapping
cover. The generated planner is correct and the audit test proves its current
output is complete; a damaged or hand-edited manifest is the remaining risk.

## Theory-to-code mapping

### Free graded-replica kernel

The theory note defines

\[
q_m=\frac{(2m+1)\pi}{2},\qquad
S_m=V^T\Omega_m^{-1}V,
\]

and the exact eigenbasis kernels

\[
K_m^>=D^{1-u}(D+S_m)^{-1}D^v,
\]

\[
K_m^<=-H^u(H+S_m^{-1})^{-1}H^{1-v}.
\]

`ReplicaGreenCalculator::stable_sector_kernel_eigenbasis` implements the
diagonally scaled versions in `replica_free_greens.h`. It uses

\[
\theta_+=\frac{u+v}{2},\qquad
\theta_-=1-\frac{u+v}{2},
\]

and constructs only exponentials with non-positive exponents. The lower branch
uses `conj(mixing(b,a))`, exactly `S_m^dagger=S_m^{-1}`. The sector labels use
the conjugate Fourier ordering relative to one common notation, but the sector
reconstruction and boundary condition are internally consistent. This was
checked numerically rather than inferred only from matching symbols.

For double precision with `CDET_USE_LAPACK=1`, the one-body Hamiltonian is
diagonalized by `dsyev`, the complex middle system is solved by `zgesv`, the
condition estimate uses the exact inverse one-norm available from the diagonal
right-hand side (or `zgecon` for a general right-hand side), and basis changes
use `dgemm`. These are direct dense solvers; there is no iterative convergence
tolerance in the LAPACK path.

For non-double build types, or builds without LAPACK, the code uses the internal
Jacobi eigensolver and pivoted complex Gauss-Jordan solve.

### Normalized observable

The implementation builds four logical tables:

- twisted up and twisted down, with the requested subregion;
- ordinary up and ordinary down, with an empty subregion.

Their free chemical potentials are

\[
\mu_\uparrow^{(0)}=\mu-\alpha_\uparrow,
\qquad
\mu_\downarrow^{(0)}=\mu-\alpha_\downarrow.
\]

At every subset/order, `cdet_integrand_evaluator.h` returns

\[
C_k^{\mathrm{normalized}}
=C_k^{\mathrm{twisted},n=2}-2C_k^{\mathrm{ordinary},n=1},
\]

which matches `ln Z_2[A] = ln Z_tilde_2[A] - 2 ln Z_1`.

### MCMCMC

For each visible subset, the implementation forms

\[
\Phi_S=\lambda_{|S|}C_{|S|}(V_S)Q_{\bar S|S},
\qquad
W=\sum_S\frac{|\Phi_S|}{\binom{N}{|S|}},
\]

samples the retained subset from the corresponding local heat bath, and records

\[
A_u(V)=\frac{1}{\binom{N}{u}}
\sum_{|S|=u}\frac{\Phi_S}{\lambda_u W}.
\]

The division by `lambda_u` occurs when the production running statistics are
updated. The learned histogram seed is not a uniform approximation: its exact
conditional generation density is included in every flux. Positive uniform
mixing gives the proposal full support and therefore preserves ergodicity.

The prose near the end of `docs/mcmcmc_theory_note.tex` still calls the current
implementation the uniform-only baseline. That sentence is stale: the code now
implements both uniform and learned factorized-histogram seeds.

## Precision and convergence controls

| Control | Code effect | Bias or failure behavior | Audit conclusion |
|---|---|---|---|
| `LOAD_DATA_REAL` | Arithmetic used to diagonalize `h`, form scaled sector matrices, and solve them | Changes roundoff; double uses LAPACK only when enabled | Core load-data precision selector |
| `VECTOR_REAL` | Stored Chebyshev coefficients and hot-path Clenshaw arithmetic | Quantizes the Green table to this type | Must equal `LOAD_DATA_REAL` in current chunk workflow |
| `INTEGRAND_REAL` | Principal minors and CDet subset recursion | Changes cancellation accuracy | Independent of table storage precision |
| `CHEB_NU`, `CHEB_NV` | Fixed 2D Chebyshev degrees on each triangle | Interpolation truncation; no adaptive stopping test | Degree 25 gave max error `1.02e-10`, RMS `1.82e-11` in the audited beta-8 `3x3` probe |
| `GREEN_TABLE_DROP_BITS` | Truncates coefficient/corner mantissas after construction | Deliberate table quantization | Applied consistently in resident and chunk paths |
| `BRANCH_EPSILON` | Routes points with the triangle denominator below the threshold to the exact `(0,0)` corner table | Prevents `0/0`; it is not a solver convergence tolerance | `1e-13` is safe in double for beta of order one; parser should still require `0 < epsilon << beta` |
| `LOAD_DATA_CONDITION_CHECK` | `off`, warning (`auto`), or exception (`strict`) after estimating `rcond` | Does not alter the computed solution | Default `auto` is diagnostic only |
| `LOAD_DATA_CONDITION_RCOND_WARN` | Warning/strict threshold, default `1e-12` | Only controls reporting or strict failure | Size sweep remained at `rcond >= 1.87e-2` |
| `LOAD_DATA_CONDITION_MAX_WARNINGS` | Limits warning count | No numerical effect | Diagnostic only |
| `TOL_FPM` | Sets `fallback_epsilon = TOL_FPM * max_abs(A)` for fast-recursion pivots | A small pivot causes a complete fallback to pivoted determinant evaluation; it does not zero the result | Primarily speed versus robustness, not an iterative residual tolerance |
| `FPM_GLOBAL_SHIFT` | Uniform diagonal shift used by the fast principal-minor identity; `auto` uses `max_abs(A)/n^2` | Algebraically removed afterward | Explicit and auto shifts matched the reference minors |
| `ZTRANSFORM_EPSILON0` | Base roundoff scale; auto is `2e-16` for double, long-double epsilon, `1e-31` for double-double | Used by an actual cancellation filter | This can deliberately set very small connected terms to zero |
| `ZTRANSFORM_SAFETY_FACTOR` | Multiplies `ZTRANSFORM_EPSILON0` | Filters when `abs(C) < threshold * sum(abs(rhs terms))` | This is the main CDet-recursion cutoff and should be stability-scanned |
| internal solve pivot threshold | Fallback solver declares a pivot singular at `real_tolerance_from_digits(8)` | Hard failure | `1e-12` for both double and long double, `1e-24` for double-double; see hardening issue below |
| internal Jacobi threshold | Stops non-LAPACK symmetric diagonalization | Hard failure if max sweeps are exhausted | Double uses `8 eps * scale`; long double currently falls back to `1e-12` |
| histogram negative-roundoff clamp | Conditional-density subtraction in MCMCMC clamps values in `(-1e-12,0)` to zero | Hidden fixed roundoff guard | Safe in tested order range, but should be made type-aware or checked for larger negative values |

The legacy keys `DIAGONAL_SIMPSON_ABS_TOL`,
`DIAGONAL_SIMPSON_REL_TOL`, and `DIAGONAL_SIMPSON_MAX_DEPTH` are not parsed or
used by this branch. The diagonal observable is integrated exactly from the
Chebyshev line coefficients. The ED `XI_STEPS`, stencil radius, and ED order
control only the optional exact-diagonalization finite-difference benchmark,
not production CDet convergence.

## Size-sweep result

The diagnostic used beta 8, effective free chemical potential `-0.4`, periodic
boundaries, and an origin-square region with approximately fixed area fraction.
It reconstructed the scaled middle matrix and computed its exact one-norm
reciprocal condition number.

| Lattice | Matrix dimension | upper `rcond` | lower `rcond` | max `abs(M_ab)` | finite kernel |
|---:|---:|---:|---:|---:|:---:|
| `3x3` | 9 | `2.63e-1` | `3.11e-1` | `1.01` | yes |
| `10x10` | 100 | `6.73e-2` | `5.10e-2` | `1.75` | yes |
| `20x20` | 400 | `2.94e-2` | `1.87e-2` | `1.75` | yes |

The two sectors have identical condition numbers because they are conjugate.
The unitarity defect of the sector mixing matrix stayed below `1.8e-15`. Very
small individual entries (down to about `1e-30`) do appear as expected from
diagonal scaling, but they coexist with an order-one matrix norm and healthy
`rcond`; this is decay, not cancellation between an overflowing and an
underflowing exponential. No positive exponent is formed in the implementation.

Generated metrics and heatmaps are in `tests/artifacts/green_kernel/` after
running `tests/run_green_diagnostics.sh`.

## Hardening findings

### 1. Chunk manifest coverage is assumed by production validation

`validate_chunked_outputs_for_real` checks the size of every file named by the
current TSV jobs. It does not check that, for each table/branch, node ranges and
site-pair ranges are nonempty, in bounds, non-overlapping, and exactly cover the
full range. `run_chunked_transform_sitepair_job_for_real` initializes its sample
grid to zero, so an omitted node interval in a damaged TSV can become silent
zeros. `load_coeff_chunks_for_branch` also computes the destination pointer from
the TSV begin offset without an explicit destination-bounds check.

The planner currently emits an exact cover, and `test_chunked_pipeline.cpp`
checks this property independently. The production validator should eventually
perform the same check before writing `READY` and before resident loading.

### 2. `GREEN_TABLE_DROP_BITS` is not part of the chunk manifest identity

The transform stage applies `GREEN_TABLE_DROP_BITS`, but
`current_chunked_manifest_values` does not record it. Reusing the same chunk
directory under a params file with a different requested drop count will pass
the manifest check while loading the already-built precision. The stored data
are internally consistent; the issue is provenance and expectation checking.

### 3. Long-double non-LAPACK tolerances do not use long-double epsilon

`real_tolerance_from_digits` returns `1e-12` for ordinary floating types other
than the explicitly handled double-double and multiprecision types. Therefore
the long-double Jacobi and fallback solve paths use a double-like absolute
threshold and do not exploit full long-double precision. Current HPC double
load-data runs with LAPACK do not use this fallback path.

### 4. `BRANCH_EPSILON` lacks parser range validation

A positive small value has the intended meaning. A zero or negative value can
expose the exact triangle corner to division by zero, while a value comparable
to beta would replace a finite part of the interpolation domain by the corner.
The current parser accepts both. The tested `1e-13` value is numerically
reasonable.

### 5. The advertised maximum envelope order is wider than all helper types

Configuration validation permits envelope order up to 48, but some popcount
paths cast masks to 32-bit `uint32_t`, and `2^N` storage is already impractical
well before 48. Current orders 9 and 10 are unaffected. The validated maximum
should eventually match the actual mask and memory implementation.

### 6. Cached integrand scalars assume one immutable parameter set per process

The thread-local runtime-scalar cache is invalidated only when `N1` or `N2`
changes, not when beta, U, shifts, FPM settings, or branch epsilon change. The
production executable loads one parameter set and never mutates it, so current
runs are correct. A future in-process parameter sweep would need a complete
cache key or explicit reset.

## Reproduction

Fast suite:

```bash
tests/run_tests.sh
```

Sanitizers:

```bash
tests/run_sanitizers.sh
```

Conditioning sweep and plots:

```bash
tests/run_green_diagnostics.sh
```

All three commands completed successfully in this audit.
