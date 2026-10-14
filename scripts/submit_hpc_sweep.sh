#!/usr/bin/env bash
set -euo pipefail

# Generate one independent PBS job directory per parameter case.
#
# Typical use on the login node:
#   bash scripts/submit_hpc_sweep.sh
#   bash scripts/submit_hpc_sweep.sh --submit
#
# Override these without editing the script if needed:
#   SOURCE_DIR=/path/to/cdet-renyi-entanglement-hpc \
#   DATA_ROOT=/path/to/scratch/cdet_data_collection \
#   bash scripts/submit_hpc_sweep.sh --submit

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to an absolute scratch directory}"
SWEEP_ROOT="${SWEEP_ROOT:-$DATA_ROOT/$(date +%Y%m%d_%H%M%S)}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-Aphrodite}"

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=true
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: $0 [--submit]"
    exit 2
fi

# Format:
#   add_case case_name ncpus mem walltime <<'PARAMS'
#   ...paste a complete params.txt here...
#   PARAMS
#
# This intentionally stores a full params.txt for every case.  That makes each
# run directory self-contained and makes sweeps easy to edit by copy/paste.
CASE_NAMES=()
CASE_NCPUS=()
CASE_MEMS=()
CASE_WALLTIMES=()
CASE_PARAMS=()

add_case() {
    if [[ "$#" -ne 4 ]]; then
        echo "add_case usage: add_case case_name ncpus mem walltime <<'PARAMS'"
        exit 2
    fi
    CASE_NAMES+=("$1")
    CASE_NCPUS+=("$2")
    CASE_MEMS+=("$3")
    CASE_WALLTIMES+=("$4")
    CASE_PARAMS+=("$(cat)")
}

add_case default 64 16gb 24:00:00 <<'PARAMS'
N1=2
N2=2
CHEB_NU=24
CHEB_NV=24
BRANCH_EPSILON=1e-15
TOL_FPM=1e-2
FPM_GLOBAL_SHIFT=auto
MU=1.0
U=3.0
BETA=2.1
ALPHA_UP=0.7
ALPHA_DOWN=0.7
PBC=true
SUBREGION_MASK=first_row
SUBREGION_SQUARE_RADIUS=0
INTERNAL_INTEGRAND_THREADS=1
TABLE_BUILD_THREADS=8
GREEN_TABLE_DROP_BITS=0
LOAD_DATA_REAL=double
VECTOR_REAL=double
INTEGRAND_REAL=double
ZTRANSFORM_EPSILON0=auto
ZTRANSFORM_SAFETY_FACTOR=1e5

MCMCMC_ORDER_MIN=1
MCMCMC_ORDER_MAX=3
MCMCMC_ENVELOPE_ORDER=4
MCMCMC_NORMALIZATION_ORDER=1
MCMCMC_LAMBDAS=1,1,1
MCMCMC_CHAIN_COUNT=4
MCMCMC_BASE_SEED=2233222666
MCMCMC_LAMBDA_WARMUP_STEPS=100000
MCMCMC_LAMBDA_ADAPT_INTERVAL=5000
MCMCMC_LAMBDA_ADAPT_RATE=0.2
MCMCMC_PRODUCTION_STEPS=1000000
MCMCMC_SEED_MODE=histogram
MCMCMC_SEED_HISTOGRAM_LEARNING_STEPS=200000
MCMCMC_SEED_HISTOGRAM_TAU_BINS=30
MCMCMC_SEED_HISTOGRAM_UNIFORM_MIX=0.1
MCMCMC_SEED_HISTOGRAM_PSEUDO_COUNT=0
MCMCMC_SEED_HISTOGRAM_LEARNING_ONLY=false
MCMCMC_SEED_HISTOGRAM_OUTPUT_PREFIX=histograms/current_seed
MCMCMC_COMPARE_UNIFORM_SEED=false

ED_BENCHMARK_ENABLED=false
ED_BENCHMARK_ONLY=false
ED_THREADS=auto
ED_XI_STEPS=1e-2,5e-3,2e-3,1e-3
ED_STENCIL_RADIUS=4
ED_ORDER_MAX=3
ED_DENSE_THRESHOLD=8000
ED_SYMMETRY=auto
ED_PROGRESS=false
ED_BLOCK_SUMMARY=false

PLAIN_MC_BENCHMARK_SAMPLES=100000
PARAMS

add_case dd_tables_dd_integrand 64 16gb 24:00:00 <<'PARAMS'
N1=2
N2=2
CHEB_NU=24
CHEB_NV=24
BRANCH_EPSILON=1e-15
TOL_FPM=1e-2
FPM_GLOBAL_SHIFT=auto
MU=1.0
U=3.0
BETA=2.1
ALPHA_UP=0.7
ALPHA_DOWN=0.7
PBC=true
SUBREGION_MASK=first_row
SUBREGION_SQUARE_RADIUS=0
INTERNAL_INTEGRAND_THREADS=1
TABLE_BUILD_THREADS=8
GREEN_TABLE_DROP_BITS=0
LOAD_DATA_REAL=double_double
VECTOR_REAL=double_double
INTEGRAND_REAL=double_double
ZTRANSFORM_EPSILON0=auto
ZTRANSFORM_SAFETY_FACTOR=1e5

MCMCMC_ORDER_MIN=1
MCMCMC_ORDER_MAX=3
MCMCMC_ENVELOPE_ORDER=4
MCMCMC_NORMALIZATION_ORDER=1
MCMCMC_LAMBDAS=1,1,1
MCMCMC_CHAIN_COUNT=4
MCMCMC_BASE_SEED=2233222666
MCMCMC_LAMBDA_WARMUP_STEPS=100000
MCMCMC_LAMBDA_ADAPT_INTERVAL=5000
MCMCMC_LAMBDA_ADAPT_RATE=0.2
MCMCMC_PRODUCTION_STEPS=1000000
MCMCMC_SEED_MODE=histogram
MCMCMC_SEED_HISTOGRAM_LEARNING_STEPS=200000
MCMCMC_SEED_HISTOGRAM_TAU_BINS=30
MCMCMC_SEED_HISTOGRAM_UNIFORM_MIX=0.1
MCMCMC_SEED_HISTOGRAM_PSEUDO_COUNT=0
MCMCMC_SEED_HISTOGRAM_LEARNING_ONLY=false
MCMCMC_SEED_HISTOGRAM_OUTPUT_PREFIX=histograms/current_seed
MCMCMC_COMPARE_UNIFORM_SEED=false

ED_BENCHMARK_ENABLED=false
ED_BENCHMARK_ONLY=false
ED_THREADS=auto
ED_XI_STEPS=1e-2,5e-3,2e-3,1e-3
ED_STENCIL_RADIUS=4
ED_ORDER_MAX=3
ED_DENSE_THRESHOLD=8000
ED_SYMMETRY=auto
ED_PROGRESS=false
ED_BLOCK_SUMMARY=false

PLAIN_MC_BENCHMARK_SAMPLES=100000
PARAMS

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path"
        exit 1
    fi
}

write_pbs() {
    local pbs_path="$1"
    local case_name="$2"
    local ncpus="$3"
    local mem="$4"
    local walltime="$5"
    local run_dir="$6"

    cat > "$pbs_path" <<PBS
#!/bin/bash
#PBS -N ${JOB_NAME_PREFIX}_${case_name}
#PBS -l walltime=${walltime}
#PBS -l select=1:ncpus=${ncpus}:mem=${mem}
#PBS -j oe

set -euo pipefail

module load tools/prod
module load GCCcore/12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load binutils/2.39-GCCcore-12.2.0
module load GCC/12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0
module load libxml2/2.10.3-GCCcore-12.2.0
module load libpciaccess/0.17-GCCcore-12.2.0
module load hwloc/2.8.0-GCCcore-12.2.0
module load OpenSSL/1.1
module load libevent/2.1.12-GCCcore-12.2.0
module load UCX/1.13.1-GCCcore-12.2.0
module load libfabric/1.16.1-GCCcore-12.2.0
module load PMIx/4.2.2-GCCcore-12.2.0
module load UCC/1.1.0-GCCcore-12.2.0
module load OpenMPI/4.1.4-GCC-12.2.0
module load gompi/2022b
module load Szip/2.1.1-GCCcore-12.2.0
module load HDF5/1.14.0-gompi-2022b

SOURCE_DIR="${SOURCE_DIR}"
RUN_DIR="${run_dir}"
EXECUTABLE="\$RUN_DIR/Aphrodite"
BUILD_DIR="\$RUN_DIR/build_hpc"
LOG_DIR="\$RUN_DIR/logs"
LOG_FILE="\$LOG_DIR/${case_name}_\${PBS_JOBID:-local}.log"

cleanup() {
    rm -rf "\$BUILD_DIR" "\$EXECUTABLE"
}
trap cleanup EXIT

mkdir -p "\$LOG_DIR" "\$RUN_DIR/histograms"
cd "\$RUN_DIR"

echo "Source: \$SOURCE_DIR"
echo "Run dir: \$RUN_DIR"
echo "Job: \${PBS_JOBID:-local}"
echo "Host: \$(hostname)"
echo "Started: \$(date)"
echo "PBS ncpus: ${ncpus}"
echo "Using params.txt:"
sed -n '1,260p' params.txt

make -C "\$SOURCE_DIR" TARGET="\$EXECUTABLE" BUILD_DIR="\$BUILD_DIR" print-config
make -C "\$SOURCE_DIR" TARGET="\$EXECUTABLE" BUILD_DIR="\$BUILD_DIR" -j 1

echo "Running \$EXECUTABLE ..."
"\$EXECUTABLE" > "\$LOG_FILE" 2>&1
RUN_STATUS=\$?

cat "\$LOG_FILE"

echo "Finished: \$(date)"
echo "Exit status: \$RUN_STATUS"
exit "\$RUN_STATUS"
PBS
    chmod +x "$pbs_path"
}

require_file "$SOURCE_DIR/Makefile"

mkdir -p "$SWEEP_ROOT"

echo "Source dir: $SOURCE_DIR"
echo "Sweep root: $SWEEP_ROOT"
echo

for case_index in "${!CASE_NAMES[@]}"; do
    case_name="${CASE_NAMES[$case_index]}"
    ncpus="${CASE_NCPUS[$case_index]}"
    mem="${CASE_MEMS[$case_index]}"
    walltime="${CASE_WALLTIMES[$case_index]}"

    run_dir="$SWEEP_ROOT/$case_name"
    mkdir -p "$run_dir/logs" "$run_dir/histograms"
    printf "%s\n" "${CASE_PARAMS[$case_index]}" > "$run_dir/params.txt"

    pbs_path="$run_dir/run.pbs"
    write_pbs "$pbs_path" "$case_name" "$ncpus" "$mem" "$walltime" "$run_dir"

    echo "Prepared $case_name"
    echo "  params: $run_dir/params.txt"
    echo "  pbs:    $pbs_path"

    if [[ "$SUBMIT" == true ]]; then
        if ! command -v qsub >/dev/null 2>&1; then
            echo "qsub not found. Generated jobs but did not submit."
            exit 1
        fi
        job_id="$(cd "$run_dir" && qsub run.pbs)"
        echo "  qsub:   $job_id"
    else
        echo "  submit: cd $run_dir && qsub run.pbs"
    fi
    echo
done

if [[ "$SUBMIT" == false ]]; then
    echo "Generated only. Re-run with --submit to qsub every case."
fi
