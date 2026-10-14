# Local migration validation - 2026-10-14

All checks below ran in a separate staging copy. The private source checkout
was not modified. No HPC job, Git push, release or visibility change occurred.

Environment: macOS, AppleClang 15, C++20. Tests used the generic CPU build;
the optional CMake LAPACK configuration linked Apple's Accelerate framework.

| Check | Result |
| --- | --- |
| SHA-256 of `src/` + `include/` | All 25 files equal upstream `a782c8f00d5f510fd53e88b2c125fc66d43d9013` |
| Make build, internal solver | Passed |
| CMake Release build, LAPACK enabled | Passed |
| Six fast audit executables | All passed |
| Same six executables with AddressSanitizer + UndefinedBehaviorSanitizer | All passed; no sanitizer diagnostic |
| Small resident application example | Two chains completed 5000 production steps each |
| Small chunked application example | All 8 stages passed; 40 sample, 32 transform, 8 corner jobs |
| Resident vs chunked example order-1 observable | Absolute difference 2.220446049250313e-16 |
| End-to-end checkpoint test, Make binary | 1500 vs 137 + 1363 production steps: two final checkpoints byte-identical |
| Same checkpoint test, CMake/LAPACK binary | Passed |
| Source shell syntax | Passed |
| Generated PBS scripts | Eight scripts pass `bash -n`; checkpoint exports preserved |
| Parent params.txt shadow guard | Rejects before workflow creation |
| Existing-run overwrite guard | Rejects workflow regeneration |
| Current import sensitive-path/token initial scan | No personal account/path or tested credential marker found |

The numerical tests themselves remain the upstream tests. Portability edits
affect dependency discovery in their launchers, not their assertions or the
production headers. The new resume test executes the unchanged application.
The chunked example launcher was also tested from a checkout path containing
spaces; it now uses relative Make targets in that default local path.

The small examples are not substitutes for long-chain error/convergence
studies. No full Green-kernel size sweep, Linux run, PBS scheduler execution
or large-memory production run was performed during migration. CMake's macOS
linker emitted a harmless duplicate `-lm`/`-ldl` warning. Known core limitations
are documented separately, not claimed fixed by these checks.

Reproduction commands:

```sh
python3 scripts/verify_production_core.py
make -j 1
bash tests/run_tests.sh
bash tests/run_sanitizers.sh
bash scripts/run_example.sh 3x3-resident
bash scripts/run_example.sh 3x3-chunked
python3 tests/test_checkpoint_resume.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCDET_USE_LAPACK=ON
cmake --build build -j 1
python3 tests/test_checkpoint_resume.py --binary build/cdet_entanglement_entropy
```
