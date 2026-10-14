# Chunked load data and resident MCMCMC on PBS

The production stages are:

```text
plan -> precompute_calculators array -> sample_nodes array -> corners array
     -> transform_sitepairs array -> diagonal_reduce -> validate -> MCMCMC
```

Small local equivalent:

```sh
bash scripts/run_example.sh 3x3-chunked
```

The example has 40 node-sample jobs, 32 transform jobs and 8 corner jobs.
Uneven final chunks exercise range handling without needing a large lattice.
These are subprocesses locally, PBS array subjobs on the cluster.

## Prepare cluster jobs

Install a C++20 toolchain, Boost and Eigen. For double-precision table
construction, a tested BLAS/LAPACK library is recommended. Choose your site's
module names and memory/walltime limits; no Imperial account/path is assumed.
Each run needs a NEW absolute directory without whitespace (a Make target
restriction). Its parent must not contain params.txt.

```sh
PARAMS_FILE="$PWD/examples/3x3-chunked/params.txt" \
RUN_DIR="/path/to/scratch/my-new-run" \
HPC_MODULES="your/compiler/module your/boost/module your/eigen/module" \
bash scripts/submit_hpc_chunked_workflow.sh
```

Omit `HPC_MODULES` when the batch environment already provides the toolchain.
Optional `LAPACK_MODULES` is also loaded in each job. The script resolves its
source directory from its own location; `SOURCE_DIR` can override it.

The command above ONLY generates scripts. Review params.txt,
chunked_resource_summary.txt and all eight PBS files before submission.
For a fresh directory, adding `--submit` prepares and submits with dependencies.
Do not rerun the generator over an existing run: it refuses to overwrite
params.txt. To submit scripts prepared without `--submit`, follow the ordered
qsub commands printed by the generator. This repository does not submit any
job merely by being cloned, built or tested.

## Resource and numerical settings

- `LOAD_DATA_WORKFLOW=load_chunks` selects loading the completed tables in
  the final stage; each earlier job exports its own construction stage.
- `MCMCMC_TABLE_BACKEND=resident` keeps the final tables in memory.
- `LOAD_DATA_REAL` and `VECTOR_REAL` must match for this chunked workflow.
- `*_NCPUS`, `*_MEM_GB` and `*_WALLTIME` select PBS resources per stage.
- `*_ARRAY_LIMIT` is concurrent subjobs, not the number of chunks.
- `LOAD_DATA_NODE_CHUNK_SIZE` and `LOAD_DATA_SITEPAIR_CHUNK_SIZE` determine
  job counts; zero requests automatic sizing by the target file/memory keys.
- `MCMCMC_CHAIN_COUNT` controls chains and normally matches MCMCMC_NCPUS.
- `USE_LAPACK`, `LAPACK_LIBS` and `USE_CHEB_SIMD` are build selections.
  The generic-CPU Make default avoids accidental login/compute ISA mismatch.

The planner's memory estimates are not a reservation guarantee. Check site
limits and measured peak memory before scaling up. The imported research
workflow has TB-scale storage costs at large sizes; start with the small example.

## Resume only MCMCMC

See [CHECKPOINTS.md](CHECKPOINTS.md). Preserve the physical parameters,
tables, histogram settings and seed. Set the cumulative production target
and `MCMCMC_RESUME=require` in params.txt, and change the PBS export
`CDET_MCMCMC_RESUME` to `require`. After confirming the previous job has
stopped and backing up its log, submit only `08_mcmcmc.pbs`.

Missing or malformed chunks must not be worked around by creating a READY
file manually. The historical validator still has coverage/identity limits;
see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Historical helpers

`run.pbs` and `scripts/submit_hpc_sweep.sh` retain the old Imperial module
stack and example sweeps for reference. They are not the primary portable
workflow, do not include every guard above, and must be reviewed before use.
Do not submit them unchanged to an arbitrary cluster.
