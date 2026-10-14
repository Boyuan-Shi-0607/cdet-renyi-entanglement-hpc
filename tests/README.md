# Numerical Audit Tests

This directory contains audit-only tests for the production code on branch
`hpc_version_mem_chunks_mcmcmc_optimized_`. The tests include the production
headers directly; they do not change the implementation in `src/`, `include/`,
or the HPC workflow scripts.

## Fast audit suite

Run:

```bash
tests/run_tests.sh
```

The suite checks:

- `test_green_kernel.cpp`: the stable graded-replica Green kernel against the
  unscaled theoretical formula at moderate beta; the equal-time jump, temporal
  boundary condition, conjugate replica sectors, and the bounded scaled matrix
  at large beta.
- `test_chebyshev_accuracy.cpp`: degree convergence of the piecewise 2D
  Chebyshev table against direct Green-kernel solves at random points, plus
  exact corner-table checks.
- `test_fpm_cdet.cpp`: all fast principal minors against pivoted determinants,
  including forced fallback cases; the connected-determinant recursion against
  an independent set-partition cumulant; and the exact role of
  `BRANCH_EPSILON`.
- `test_chunked_pipeline.cpp`: an end-to-end non-square `3x2` chunked build with
  deliberately uneven node and site-pair chunks. It proves exact manifest
  coverage for all four table kinds and both branches, compares every loaded
  coefficient/corner with a direct resident build, and compares the chunked and
  resident order-1 reductions.
- `test_integrand_all_subsets.cpp`: every subset produced by the MCMCMC hot path
  is compared with an independent call at that subset's actual vertex count.
- `test_mcmcmc_synthetic.cpp`: a variable-retained-order set heat-bath test for
  a synthetic integrand whose exact integral is one at every order. It tests
  both uniform and positive histogram proposals and verifies that all retained
  vertex counts are visited. Set `CDET_AUDIT_LONG=1` for a longer statistical
  run.

The same suite can be rebuilt with AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
tests/run_sanitizers.sh
```

## Green-kernel size and conditioning sweep

Run:

```bash
tests/run_green_diagnostics.sh
```

The default sweep uses `3x3`, `10x10`, and `20x20` lattices at beta 8. It
reconstructs the scaled middle matrices for both branches and both replica
sectors, computes exact one-norm reciprocal condition numbers, checks sector
unitarity and finite Green kernels, and writes magnitude heatmaps. Generated
CSV and PNG files are placed in `tests/artifacts/green_kernel/` and ignored by
Git.

Alternative sizes can be passed after the script name, for example:

```bash
tests/run_green_diagnostics.sh 3 8 12
```

The diagnostics use LAPACK/BLAS (`Accelerate` on macOS, `-llapack -lblas` on
Linux). They are intentionally separate from the fast suite because the exact
condition-number calculation for a `20x20` lattice requires dense solves of
`400x400` complex matrices.
