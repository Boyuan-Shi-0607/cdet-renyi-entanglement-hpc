# CDet Renyi Entanglement HPC

Research code for the second Renyi subsystem entropy of the square-lattice
Hubbard model. It combines fermionic replica Green functions, connected
determinant summation, learned-histogram many-configuration MCMC, chunked
table construction, and per-chain checkpoint/resume.

**Status: local migration candidate, not a published release.** A software
license has not yet been selected. See [release checklist](docs/RELEASE_CHECKLIST.md).
The numerical core is byte-identical to production commit
`a782c8f00d5f510fd53e88b2c125fc66d43d9013`; build scripts and documentation have
been adapted for distribution. See [provenance](docs/PROVENANCE.md).

## Build

Requires a C++20 compiler, Boost headers, Eigen headers and POSIX threads.
BLAS/LAPACK is optional for the double-precision load-data path. Python 3 is
needed for checkpoint checks; plots additionally need numpy/matplotlib.
The imported source does not call GSL or HDF5, so these are no longer mandatory
build dependencies. No precompiled executable is distributed.

```sh
make -j 1
python3 scripts/verify_production_core.py
```

If headers are outside system/Homebrew locations, set `BOOST_INCLUDE_DIR`
and `EIGEN_INCLUDE_DIR`. `make USE_LAPACK=1` uses Accelerate on macOS and
`LAPACK_LIBS` (default `-llapack -lblas`) on Linux. Run `make clean` before
changing compiler, precision macros, LAPACK or SIMD build settings.
Generic CPU code and `USE_CHEB_SIMD=0` are the defaults. `NATIVE_ARCH=1` is
an explicit opt-in: do not copy such binaries to a different CPU model.
The [migration validation record](docs/MIGRATION_VALIDATION.md) lists the
checks actually run and distinguishes local tests from untested HPC deployment.

Alternatively:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 1
```

Make creates `Aphrodite`; CMake creates `build/cdet_entanglement_entropy`.
The convenience example launcher uses Make.

## Small local examples

```sh
bash scripts/run_example.sh 3x3-resident
bash scripts/run_example.sh 3x3-chunked
```

Each command creates a separate directory under `runs/`. Both use resident
MCMCMC; the second constructs its tables in deliberately uneven chunks.
These short two-chain runs are workflow smoke tests, not publication-quality
entropy estimates. Their parameters are under `examples/`; use the printed
run directory to inspect `logs/` and `mcmcmc_checkpoints/`.

The production executable searches `../params.txt` before `params.txt`.
Keep the parent of each run free of `params.txt`. The new example wrapper
and chunk launchers refuse this ambiguous layout; the unchanged C++ binary
itself still has the historical search order.

## Checkpoints and testing

```sh
bash tests/run_tests.sh
python3 tests/test_checkpoint_resume.py
bash tests/run_sanitizers.sh
```

The resume check compares complete checkpoints after 1500 uninterrupted
steps and 137 + 1363 steps with an intermediate process restart.
It preserves RNG state, vertices, adaptation and accumulated statistics.
Do not attempt recovery by advancing an RNG from the seed alone.
See [checkpoint usage](docs/CHECKPOINTS.md), [test scope](tests/README.md)
and [known limitations](docs/KNOWN_LIMITATIONS.md).

## HPC workflow

The PBS generator prepares
`plan -> precompute -> sample -> corners -> transform -> diagonal -> validate -> MCMCMC`.
It does not submit jobs unless explicitly invoked with `--submit`.
Run directories, environment modules and resources are user configuration;
no personal cluster account is required by the scripts.
See [HPC instructions](docs/hpc_chunked_workflow.md).
`run.pbs` and `scripts/submit_hpc_sweep.sh` are retained legacy Imperial
templates, not portable defaults; review their module lists before use.

## Meaning of the output

The normalized coefficients are those of `ln Tr(rho_A^2) / N`, not the
entropy itself. Include the deterministic free term; `S_2/N` has the
opposite sign. At finite temperature this is a subsystem entropy with
thermal contributions, not a pure-state entanglement measure.

All orders from a chain are correlated. The legacy console error formula
does not include the shared-normalization covariance. Use chain-level
jackknife/covariance analysis for publication uncertainties, as in the
separate paper reproduction package. This repository does not silently
replace that analysis or claim that resummation errors are purely statistical.

## Citation and data

Author: Boyuan Shi. Related manuscript:
*Diagrammatic Monte Carlo for Fermionic Renyi Entanglement Entropy*.
`CITATION.cff` provides citation metadata without inventing a DOI or arXiv ID.
The large Green tables, production checkpoints, old drafts and referee
correspondence are intentionally not included. The paper reproduction
archive is a separate artifact; its public persistent link is a release task.
