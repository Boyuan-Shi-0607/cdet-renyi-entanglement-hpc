# Production provenance

- Origin repository: `CDet_double_double_entanglement_entropy` (private).
- Origin branch: `hpc_version_mem_chunks_mcmcmc_optimized_`.
- Origin commit: `a782c8f00d5f510fd53e88b2c125fc66d43d9013`.
- Local migration: 2026-10-14.
- Paper source snapshot name:
  `cdet_lru_cache_condcheck_pointerguard_checkpoint_strtod_20260601_155300`.

Before migration, `src/main.cpp`, all 24 files under `include/`, and the four
upstream scripts were compared byte-for-byte with the paper snapshot. They
matched. The snapshot directory date is a label, not a substitute for a hash:
the corrected hexadecimal vertex parser is present in the selected code.

The numerical core (`src/`, `include/`) is unchanged. The original hashes
of all selected upstream files are in `provenance/upstream_files.json`.
Run `python3 scripts/verify_production_core.py` to verify all 25 core files.
The original default parameter file is kept as
`provenance/params_upstream.txt`; it is an old development example, not a
claim to represent all paper production runs.

Packaging changes are separate from scientific changes:

- Make/CMake discover dependencies without a personal HPC installation;
  generic CPU builds are default and native-CPU optimization is opt-in.
- Remove unused GSL/HDF5 requirements from the build interface.
- Add small resident/chunked examples and an end-to-end resume check.
- Parameterize cluster paths and modules in the chunked PBS generator,
  reject parent-parameter shadowing and accidental run regeneration.
- Make test include paths/Python selection configurable.
- Correct the stale uniform-only sentence in the MCMCMC theory note.
- Add usage, citation, limitations and release-decision documentation.

The private Git history and editor state were not imported. Authors and
third-party attribution must still be retained when a license is chosen.
No production job, table, checkpoint or source file on HPC was modified.
