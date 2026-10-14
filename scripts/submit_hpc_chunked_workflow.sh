#!/usr/bin/env bash
set -euo pipefail

# Generate and optionally submit the chunked load-data HPC workflow:
#   plan -> calculator-cache precompute array -> sample array -> corner array
#        -> transform array -> diagonal -> validate -> MCMCMC
#
# Typical use on the login node:
#   PARAMS_FILE=params.txt RUN_DIR=/rds/.../case64 bash scripts/submit_hpc_chunked_workflow.sh
#   PARAMS_FILE=params.txt RUN_DIR=/rds/.../case64 bash scripts/submit_hpc_chunked_workflow.sh --submit

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PARAMS_FILE="${PARAMS_FILE:?Set PARAMS_FILE to a complete parameter file}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a new absolute run directory}"
HPC_MODULES="${HPC_MODULES:-}"
if [[ "$RUN_DIR" != /* || "$RUN_DIR" == *[[:space:]]* || -e "$RUN_DIR/params.txt" ]]; then
    echo "RUN_DIR must be absolute, contain no whitespace and not already contain params.txt; resume with the existing PBS script." >&2
    exit 2
fi
if [[ -e "$(dirname "$RUN_DIR")/params.txt" ]]; then
    echo "A parent params.txt would shadow this run's parameters." >&2
    exit 2
fi

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=true
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: $0 [--submit]"
    exit 2
fi

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "Missing PARAMS_FILE=$PARAMS_FILE"
    exit 1
fi

ENV_PLAN_MEM_GB="${PLAN_MEM_GB:-}"
ENV_PRECOMPUTE_MEM_GB="${PRECOMPUTE_MEM_GB:-}"
ENV_SAMPLE_MEM_GB="${SAMPLE_MEM_GB:-}"
ENV_CORNER_MEM_GB="${CORNER_MEM_GB:-}"
ENV_TRANSFORM_MEM_GB="${TRANSFORM_MEM_GB:-}"
ENV_DIAGONAL_MEM_GB="${DIAGONAL_MEM_GB:-}"
ENV_VALIDATE_MEM_GB="${VALIDATE_MEM_GB:-}"
ENV_MCMCMC_MEM_GB="${MCMCMC_MEM_GB:-}"

param_value() {
    local key="$1"
    local default_value="${2:-}"
    local value
    value="$(awk -F= -v key="$key" '
        $0 !~ /^[[:space:]]*#/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1);
            if ($1 == key) {
                v=$2;
                sub(/[[:space:]]*#.*/, "", v);
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", v);
                print v;
                found=1;
                exit;
            }
        }
        END { if (!found) exit 1; }
    ' "$PARAMS_FILE" 2>/dev/null || true)"
    if [[ -z "$value" ]]; then
        value="$default_value"
    fi
    printf "%s" "$value"
}

param_or_env() {
    local key="$1"
    local env_name="$2"
    local default_value="$3"
    local value
    local env_value
    value="$(param_value "$key" "$default_value")"
    eval "env_value=\"\${${env_name}:-}\""
    if [[ -n "$env_value" ]]; then
        value="$env_value"
    fi
    printf "%s" "$value"
}

param_or_captured_env() {
    local key="$1"
    local captured_env_value="$2"
    local default_value="$3"
    local value
    value="$(param_value "$key" "$default_value")"
    if [[ -n "$captured_env_value" ]]; then
        value="$captured_env_value"
    fi
    printf "%s" "$value"
}

ceil_div() {
    local a="$1"
    local b="$2"
    echo $(( (a + b - 1) / b ))
}

ceil_gb_from_bytes() {
    awk -v bytes="$1" 'BEGIN {
        gb = bytes / (1024.0 * 1024.0 * 1024.0);
        c = int(gb);
        if (gb > c) c += 1;
        if (c < 1) c = 1;
        print c;
    }'
}

JOB_NAME_PREFIX="$(param_or_env JOB_NAME_PREFIX JOB_NAME_PREFIX AphroditeChunk)"
CHUNK_DIR_DEFAULT="$(param_value CHUNK_DIR "$RUN_DIR/chunked_green_tables")"
CHUNK_DIR="$(param_or_env LOAD_DATA_CHUNK_DIR CHUNK_DIR "$CHUNK_DIR_DEFAULT")"

SAMPLE_ARRAY_LIMIT="$(param_or_env SAMPLE_ARRAY_LIMIT SAMPLE_ARRAY_LIMIT 32)"
PRECOMPUTE_ARRAY_LIMIT="$(param_or_env PRECOMPUTE_ARRAY_LIMIT PRECOMPUTE_ARRAY_LIMIT 4)"
CORNER_ARRAY_LIMIT="$(param_or_env CORNER_ARRAY_LIMIT CORNER_ARRAY_LIMIT 8)"
TRANSFORM_ARRAY_LIMIT="$(param_or_env TRANSFORM_ARRAY_LIMIT TRANSFORM_ARRAY_LIMIT 32)"

PLAN_NCPUS="$(param_or_env PLAN_NCPUS PLAN_NCPUS 1)"
PRECOMPUTE_NCPUS="$(param_or_env PRECOMPUTE_NCPUS PRECOMPUTE_NCPUS 8)"
SAMPLE_NCPUS="$(param_or_env SAMPLE_NCPUS SAMPLE_NCPUS 1)"
CORNER_NCPUS="$(param_or_env CORNER_NCPUS CORNER_NCPUS 1)"
TRANSFORM_NCPUS="$(param_or_env TRANSFORM_NCPUS TRANSFORM_NCPUS 1)"
DIAGONAL_NCPUS="$(param_or_env DIAGONAL_NCPUS DIAGONAL_NCPUS 1)"
VALIDATE_NCPUS="$(param_or_env VALIDATE_NCPUS VALIDATE_NCPUS 1)"
MCMCMC_NCPUS="$(param_or_env MCMCMC_NCPUS MCMCMC_NCPUS auto)"

PLAN_WALLTIME="$(param_or_env PLAN_WALLTIME PLAN_WALLTIME 00:20:00)"
PRECOMPUTE_WALLTIME="$(param_or_env PRECOMPUTE_WALLTIME PRECOMPUTE_WALLTIME 01:00:00)"
SAMPLE_WALLTIME="$(param_or_env SAMPLE_WALLTIME SAMPLE_WALLTIME 04:00:00)"
CORNER_WALLTIME="$(param_or_env CORNER_WALLTIME CORNER_WALLTIME 01:00:00)"
TRANSFORM_WALLTIME="$(param_or_env TRANSFORM_WALLTIME TRANSFORM_WALLTIME 02:00:00)"
DIAGONAL_WALLTIME="$(param_or_env DIAGONAL_WALLTIME DIAGONAL_WALLTIME 00:30:00)"
VALIDATE_WALLTIME="$(param_or_env VALIDATE_WALLTIME VALIDATE_WALLTIME 00:30:00)"
MCMCMC_WALLTIME="$(param_or_env MCMCMC_WALLTIME MCMCMC_WALLTIME 24:00:00)"

N1="$(param_value N1)"
N2="$(param_value N2)"
CHEB_NU="$(param_value CHEB_NU)"
CHEB_NV="$(param_value CHEB_NV)"
VECTOR_REAL="$(param_value VECTOR_REAL double)"
LOAD_DATA_REAL="$(param_value LOAD_DATA_REAL "$VECTOR_REAL")"
NODE_CHUNK_SIZE="$(param_value LOAD_DATA_NODE_CHUNK_SIZE 0)"
SITEPAIR_CHUNK_SIZE="$(param_value LOAD_DATA_SITEPAIR_CHUNK_SIZE 0)"
SAMPLE_TARGET_FILE_GB="$(param_value LOAD_DATA_SAMPLE_TARGET_FILE_GB 4.0)"
TRANSFORM_TARGET_MEM_GB="$(param_value LOAD_DATA_TRANSFORM_TARGET_MEM_GB 4.0)"
MCMCMC_TABLE_BACKEND="$(param_or_env MCMCMC_TABLE_BACKEND MCMCMC_TABLE_BACKEND resident)"
MCMCMC_LRU_CACHE_GB="$(param_or_env MCMCMC_LRU_CACHE_GB MCMCMC_LRU_CACHE_GB 64.0)"
MCMCMC_CHECKPOINT_ENABLED="$(param_or_env MCMCMC_CHECKPOINT_ENABLED CDET_MCMCMC_CHECKPOINT_ENABLED false)"
MCMCMC_CHECKPOINT_DIR="$(param_or_env MCMCMC_CHECKPOINT_DIR CDET_MCMCMC_CHECKPOINT_DIR mcmcmc_checkpoints)"
MCMCMC_CHECKPOINT_INTERVAL_STEPS="$(param_or_env MCMCMC_CHECKPOINT_INTERVAL_STEPS CDET_MCMCMC_CHECKPOINT_INTERVAL_STEPS 10000000)"
MCMCMC_RESUME_DEFAULT="off"
case "$(printf '%s' "$MCMCMC_CHECKPOINT_ENABLED" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) MCMCMC_RESUME_DEFAULT="auto" ;;
esac
MCMCMC_RESUME="$(param_or_env MCMCMC_RESUME CDET_MCMCMC_RESUME "$MCMCMC_RESUME_DEFAULT")"
LOAD_DATA_CALCULATOR_CACHE="$(param_or_env LOAD_DATA_CALCULATOR_CACHE CDET_LOAD_DATA_CALCULATOR_CACHE auto)"
USE_LAPACK="$(param_or_env USE_LAPACK USE_LAPACK 0)"
LAPACK_LIBS="$(param_or_env LAPACK_LIBS LAPACK_LIBS "")"
LAPACK_MODULES="$(param_or_env LAPACK_MODULES LAPACK_MODULES "")"
USE_CHEB_SIMD="$(param_or_env USE_CHEB_SIMD USE_CHEB_SIMD 0)"
MCMCMC_TABLE_BACKEND="$(printf '%s' "$MCMCMC_TABLE_BACKEND" | tr '[:upper:]' '[:lower:]')"
case "$MCMCMC_TABLE_BACKEND" in
    resident) ;;
    lru|on_demand|disk|disk_cache) MCMCMC_TABLE_BACKEND="disk_lru" ;;
    disk_lru) ;;
    *)
        echo "Unsupported MCMCMC_TABLE_BACKEND: $MCMCMC_TABLE_BACKEND"
        exit 1
        ;;
esac

if [[ "$LOAD_DATA_REAL" != "$VECTOR_REAL" ]]; then
    echo "Chunked workflow currently requires LOAD_DATA_REAL == VECTOR_REAL."
    echo "Found LOAD_DATA_REAL=$LOAD_DATA_REAL VECTOR_REAL=$VECTOR_REAL"
    exit 1
fi

VECTOR_REAL_LOWER="$(printf '%s' "$VECTOR_REAL" | tr '[:upper:]' '[:lower:]')"
case "$VECTOR_REAL_LOWER" in
    double) REAL_BYTES=8 ;;
    long_double|longdouble) REAL_BYTES=16 ;;
    double_double|doubledouble|dd) REAL_BYTES=16 ;;
    *)
        echo "Unsupported VECTOR_REAL for chunked workflow: $VECTOR_REAL"
        exit 1
        ;;
esac

N_SITE=$(( N1 * N2 ))
SITEPAIR_COUNT=$(( 4 * N_SITE * N_SITE ))
NODE_COUNT=$(( CHEB_NU * CHEB_NV ))

if [[ "$NODE_CHUNK_SIZE" -le 0 ]]; then
    NODE_CHUNK_SIZE="$(awk -v target_gb="$SAMPLE_TARGET_FILE_GB" \
                           -v sitepairs="$SITEPAIR_COUNT" \
                           -v bytes="$REAL_BYTES" \
        'BEGIN {
            target = target_gb * 1024.0 * 1024.0 * 1024.0;
            c = int(target / (sitepairs * bytes));
            if (c < 1) c = 1;
            print c;
        }')"
fi

if [[ "$SITEPAIR_CHUNK_SIZE" -le 0 ]]; then
    SITEPAIR_CHUNK_SIZE="$(awk -v target_gb="$TRANSFORM_TARGET_MEM_GB" \
                                -v nodes="$NODE_COUNT" \
                                -v bytes="$REAL_BYTES" \
        'BEGIN {
            target = target_gb * 1024.0 * 1024.0 * 1024.0;
            c = int(target / (nodes * bytes * 2.5));
            if (c < 1) c = 1;
            print c;
        }')"
fi

NODE_CHUNKS="$(ceil_div "$NODE_COUNT" "$NODE_CHUNK_SIZE")"
SITEPAIR_CHUNKS="$(ceil_div "$SITEPAIR_COUNT" "$SITEPAIR_CHUNK_SIZE")"
SAMPLE_JOBS=$(( 4 * 2 * NODE_CHUNKS ))
PRECOMPUTE_JOBS=4
CORNER_JOBS=8
TRANSFORM_JOBS=$(( 4 * 2 * SITEPAIR_CHUNKS ))

COMPLEX_BYTES=$(( 2 * REAL_BYTES ))
SAMPLE_MEM_BYTES="$(awk -v n="$N_SITE" -v cb="$COMPLEX_BYTES" 'BEGIN { print 16.0 * n * n * cb * 1.5; }')"
TRANSFORM_MEM_BYTES="$(awk -v sp="$SITEPAIR_CHUNK_SIZE" -v nodes="$NODE_COUNT" -v rb="$REAL_BYTES" 'BEGIN { print sp * nodes * rb * 2.5 * 1.5; }')"
RESIDENT_BYTES="$(awk -v sp="$SITEPAIR_COUNT" -v nodes="$NODE_COUNT" -v rb="$REAL_BYTES" 'BEGIN { print ((4*2*sp*nodes*rb) + (4*2*sp*rb)) * 1.35 + 8.0*1024*1024*1024; }')"
DISK_LRU_MCMCMC_BYTES="$(awk -v cache_gb="$MCMCMC_LRU_CACHE_GB" -v sp="$SITEPAIR_COUNT" -v rb="$REAL_BYTES" 'BEGIN { print cache_gb*1024.0*1024.0*1024.0 + (4*2*sp*rb)*1.35 + 8.0*1024.0*1024.0*1024.0; }')"

SAMPLE_MEM_ESTIMATE_GB="$(ceil_gb_from_bytes "$SAMPLE_MEM_BYTES")"
PRECOMPUTE_MEM_ESTIMATE_GB="$SAMPLE_MEM_ESTIMATE_GB"
CORNER_MEM_ESTIMATE_GB="$SAMPLE_MEM_ESTIMATE_GB"
TRANSFORM_MEM_ESTIMATE_GB="$(ceil_gb_from_bytes "$TRANSFORM_MEM_BYTES")"
PRECOMPUTE_MEM_GB="$(param_or_captured_env PRECOMPUTE_MEM_GB "$ENV_PRECOMPUTE_MEM_GB" "$PRECOMPUTE_MEM_ESTIMATE_GB")"
SAMPLE_MEM_GB="$(param_or_captured_env SAMPLE_MEM_GB "$ENV_SAMPLE_MEM_GB" "$SAMPLE_MEM_ESTIMATE_GB")"
CORNER_MEM_GB="$(param_or_captured_env CORNER_MEM_GB "$ENV_CORNER_MEM_GB" "$CORNER_MEM_ESTIMATE_GB")"
TRANSFORM_MEM_GB="$(param_or_captured_env TRANSFORM_MEM_GB "$ENV_TRANSFORM_MEM_GB" "$TRANSFORM_MEM_ESTIMATE_GB")"
DIAGONAL_MEM_GB="$(param_or_captured_env DIAGONAL_MEM_GB "$ENV_DIAGONAL_MEM_GB" 4)"
VALIDATE_MEM_GB="$(param_or_captured_env VALIDATE_MEM_GB "$ENV_VALIDATE_MEM_GB" 4)"
PLAN_MEM_GB="$(param_or_captured_env PLAN_MEM_GB "$ENV_PLAN_MEM_GB" 4)"
if [[ "$MCMCMC_TABLE_BACKEND" == "disk_lru" ]]; then
    MCMCMC_MEM_GB="$(ceil_gb_from_bytes "$DISK_LRU_MCMCMC_BYTES")"
else
    MCMCMC_MEM_GB="$(ceil_gb_from_bytes "$RESIDENT_BYTES")"
fi
MCMCMC_MEM_GB="$(param_or_captured_env MCMCMC_MEM_GB "$ENV_MCMCMC_MEM_GB" "$MCMCMC_MEM_GB")"

if [[ "$MCMCMC_NCPUS" == "auto" ]]; then
    MCMCMC_NCPUS="$(param_value MCMCMC_CHAIN_COUNT 8)"
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/histograms"
cp "$PARAMS_FILE" "$RUN_DIR/params.txt"

write_common_body() {
    cat <<'PBS'
set -euo pipefail

if [[ -n "${HPC_MODULES:-}" ]]; then
    for module_name in $HPC_MODULES; do
        module load "$module_name"
    done
fi

if [[ -n "${LAPACK_MODULES:-}" ]]; then
    for module_name in $LAPACK_MODULES; do
        module load "$module_name"
    done
fi

cd "$RUN_DIR"
if [[ -e ../params.txt ]]; then
    echo "Refusing to run: ../params.txt would shadow the run parameters." >&2
    exit 2
fi
mkdir -p logs histograms
PBS
}

write_single_pbs() {
    local path="$1" name="$2" ncpus="$3" mem_gb="$4" walltime="$5" workflow="$6" log_name="$7"
    cat > "$path" <<PBS
#!/bin/bash
#PBS -N ${JOB_NAME_PREFIX}_${name}
#PBS -l walltime=${walltime}
#PBS -l select=1:ncpus=${ncpus}:mem=${mem_gb}gb
#PBS -j oe

SOURCE_DIR="${SOURCE_DIR}"
RUN_DIR="${RUN_DIR}"
EXECUTABLE="\$RUN_DIR/Aphrodite"
BUILD_DIR="\$RUN_DIR/build_hpc"
export CDET_LOAD_DATA_WORKFLOW="${workflow}"
export CDET_LOAD_DATA_CHUNK_DIR="${CHUNK_DIR}"
export CDET_LOAD_DATA_NODE_CHUNK_SIZE="${NODE_CHUNK_SIZE}"
export CDET_LOAD_DATA_SITEPAIR_CHUNK_SIZE="${SITEPAIR_CHUNK_SIZE}"
export CDET_LOAD_DATA_SAMPLE_TARGET_FILE_GB="${SAMPLE_TARGET_FILE_GB}"
export CDET_LOAD_DATA_TRANSFORM_TARGET_MEM_GB="${TRANSFORM_TARGET_MEM_GB}"
export CDET_LOAD_DATA_CALCULATOR_CACHE="${LOAD_DATA_CALCULATOR_CACHE}"
export CDET_MCMCMC_TABLE_BACKEND="${MCMCMC_TABLE_BACKEND}"
export CDET_MCMCMC_LRU_CACHE_GB="${MCMCMC_LRU_CACHE_GB}"
export CDET_MCMCMC_CHECKPOINT_ENABLED="${MCMCMC_CHECKPOINT_ENABLED}"
export CDET_MCMCMC_CHECKPOINT_DIR="${MCMCMC_CHECKPOINT_DIR}"
export CDET_MCMCMC_CHECKPOINT_INTERVAL_STEPS="${MCMCMC_CHECKPOINT_INTERVAL_STEPS}"
export CDET_MCMCMC_RESUME="${MCMCMC_RESUME}"
export LAPACK_MODULES="${LAPACK_MODULES}"
export HPC_MODULES="${HPC_MODULES}"
PBS
    write_common_body >> "$path"
    cat >> "$path" <<PBS
if [[ "${workflow}" == "plan" ]]; then
    make -C "\$SOURCE_DIR" TARGET="\$EXECUTABLE" BUILD_DIR="\$BUILD_DIR" USE_LAPACK="${USE_LAPACK}" LAPACK_LIBS="${LAPACK_LIBS}" USE_CHEB_SIMD="${USE_CHEB_SIMD}" -j 1
fi

echo "Started: \$(date)"
echo "Workflow: ${workflow}"
status=0
"\$EXECUTABLE" > "logs/${log_name}" 2>&1 || status=\$?
cat "logs/${log_name}"
echo "Finished: \$(date)"
exit "\$status"
PBS
    chmod +x "$path"
}

write_array_pbs() {
    local path="$1" name="$2" ncpus="$3" mem_gb="$4" walltime="$5" workflow="$6" jobs="$7" limit="$8" log_prefix="$9"
    cat > "$path" <<PBS
#!/bin/bash
#PBS -N ${JOB_NAME_PREFIX}_${name}
#PBS -l walltime=${walltime}
#PBS -l select=1:ncpus=${ncpus}:mem=${mem_gb}gb
#PBS -J 1-${jobs}%${limit}
#PBS -j oe

RUN_DIR="${RUN_DIR}"
export CDET_LOAD_DATA_WORKFLOW="${workflow}"
export CDET_LOAD_DATA_CHUNK_DIR="${CHUNK_DIR}"
export CDET_LOAD_DATA_NODE_CHUNK_SIZE="${NODE_CHUNK_SIZE}"
export CDET_LOAD_DATA_SITEPAIR_CHUNK_SIZE="${SITEPAIR_CHUNK_SIZE}"
export CDET_LOAD_DATA_SAMPLE_TARGET_FILE_GB="${SAMPLE_TARGET_FILE_GB}"
export CDET_LOAD_DATA_TRANSFORM_TARGET_MEM_GB="${TRANSFORM_TARGET_MEM_GB}"
export CDET_LOAD_DATA_CALCULATOR_CACHE="${LOAD_DATA_CALCULATOR_CACHE}"
export CDET_LOAD_DATA_JOB_INDEX="\${PBS_ARRAY_INDEX}"
export CDET_MCMCMC_TABLE_BACKEND="${MCMCMC_TABLE_BACKEND}"
export CDET_MCMCMC_LRU_CACHE_GB="${MCMCMC_LRU_CACHE_GB}"
export CDET_MCMCMC_CHECKPOINT_ENABLED="${MCMCMC_CHECKPOINT_ENABLED}"
export CDET_MCMCMC_CHECKPOINT_DIR="${MCMCMC_CHECKPOINT_DIR}"
export CDET_MCMCMC_CHECKPOINT_INTERVAL_STEPS="${MCMCMC_CHECKPOINT_INTERVAL_STEPS}"
export CDET_MCMCMC_RESUME="${MCMCMC_RESUME}"
export LAPACK_MODULES="${LAPACK_MODULES}"
export HPC_MODULES="${HPC_MODULES}"
PBS
    write_common_body >> "$path"
    cat >> "$path" <<PBS
echo "Started: \$(date)"
echo "Workflow: ${workflow}"
echo "Array index: \${PBS_ARRAY_INDEX}"
status=0
"\$RUN_DIR/Aphrodite" > "logs/${log_prefix}_\${PBS_ARRAY_INDEX}.log" 2>&1 || status=\$?
cat "logs/${log_prefix}_\${PBS_ARRAY_INDEX}.log"
echo "Finished: \$(date)"
exit "\$status"
PBS
    chmod +x "$path"
}

write_single_pbs "$RUN_DIR/01_plan.pbs" plan "$PLAN_NCPUS" "$PLAN_MEM_GB" "$PLAN_WALLTIME" plan plan.log
write_array_pbs "$RUN_DIR/02_precompute_calculators.pbs" precompute "$PRECOMPUTE_NCPUS" "$PRECOMPUTE_MEM_GB" "$PRECOMPUTE_WALLTIME" precompute_calculators "$PRECOMPUTE_JOBS" "$PRECOMPUTE_ARRAY_LIMIT" precompute
write_array_pbs "$RUN_DIR/03_sample_nodes.pbs" sample "$SAMPLE_NCPUS" "$SAMPLE_MEM_GB" "$SAMPLE_WALLTIME" sample_nodes "$SAMPLE_JOBS" "$SAMPLE_ARRAY_LIMIT" sample
write_array_pbs "$RUN_DIR/04_corners.pbs" corners "$CORNER_NCPUS" "$CORNER_MEM_GB" "$CORNER_WALLTIME" corners "$CORNER_JOBS" "$CORNER_ARRAY_LIMIT" corners
write_array_pbs "$RUN_DIR/05_transform.pbs" transform "$TRANSFORM_NCPUS" "$TRANSFORM_MEM_GB" "$TRANSFORM_WALLTIME" transform_sitepairs "$TRANSFORM_JOBS" "$TRANSFORM_ARRAY_LIMIT" transform
write_single_pbs "$RUN_DIR/06_diagonal.pbs" diagonal "$DIAGONAL_NCPUS" "$DIAGONAL_MEM_GB" "$DIAGONAL_WALLTIME" diagonal_reduce diagonal.log
write_single_pbs "$RUN_DIR/07_validate.pbs" validate "$VALIDATE_NCPUS" "$VALIDATE_MEM_GB" "$VALIDATE_WALLTIME" validate validate.log
write_single_pbs "$RUN_DIR/08_mcmcmc.pbs" mcmcmc "$MCMCMC_NCPUS" "$MCMCMC_MEM_GB" "$MCMCMC_WALLTIME" load_chunks mcmcmc.log

cat > "$RUN_DIR/chunked_resource_summary.txt" <<SUMMARY
N1=$N1
N2=$N2
CHEB_NU=$CHEB_NU
CHEB_NV=$CHEB_NV
VECTOR_REAL=$VECTOR_REAL
mcmcmc_table_backend=$MCMCMC_TABLE_BACKEND
mcmcmc_lru_cache_gb=$MCMCMC_LRU_CACHE_GB
mcmcmc_checkpoint_enabled=$MCMCMC_CHECKPOINT_ENABLED
mcmcmc_checkpoint_dir=$MCMCMC_CHECKPOINT_DIR
mcmcmc_checkpoint_interval_steps=$MCMCMC_CHECKPOINT_INTERVAL_STEPS
mcmcmc_resume=$MCMCMC_RESUME
load_data_calculator_cache=$LOAD_DATA_CALCULATOR_CACHE
use_lapack=$USE_LAPACK
lapack_libs=$LAPACK_LIBS
lapack_modules=$LAPACK_MODULES
use_cheb_simd=$USE_CHEB_SIMD
node_count=$NODE_COUNT
replica_sitepair_count=$SITEPAIR_COUNT
node_chunk_size=$NODE_CHUNK_SIZE
sitepair_chunk_size=$SITEPAIR_CHUNK_SIZE
precompute_jobs=$PRECOMPUTE_JOBS
sample_jobs=$SAMPLE_JOBS
corner_jobs=$CORNER_JOBS
transform_jobs=$TRANSFORM_JOBS
precompute_array_limit=$PRECOMPUTE_ARRAY_LIMIT
sample_array_limit=$SAMPLE_ARRAY_LIMIT
corner_array_limit=$CORNER_ARRAY_LIMIT
transform_array_limit=$TRANSFORM_ARRAY_LIMIT
plan_ncpus=$PLAN_NCPUS
precompute_ncpus=$PRECOMPUTE_NCPUS
sample_ncpus=$SAMPLE_NCPUS
corner_ncpus=$CORNER_NCPUS
transform_ncpus=$TRANSFORM_NCPUS
diagonal_ncpus=$DIAGONAL_NCPUS
validate_ncpus=$VALIDATE_NCPUS
mcmcmc_ncpus=$MCMCMC_NCPUS
plan_walltime=$PLAN_WALLTIME
precompute_walltime=$PRECOMPUTE_WALLTIME
sample_walltime=$SAMPLE_WALLTIME
corner_walltime=$CORNER_WALLTIME
transform_walltime=$TRANSFORM_WALLTIME
diagonal_walltime=$DIAGONAL_WALLTIME
validate_walltime=$VALIDATE_WALLTIME
mcmcmc_walltime=$MCMCMC_WALLTIME
precompute_mem_gb=$PRECOMPUTE_MEM_GB
sample_mem_gb=$SAMPLE_MEM_GB
corner_mem_gb=$CORNER_MEM_GB
transform_mem_gb=$TRANSFORM_MEM_GB
diagonal_mem_gb=$DIAGONAL_MEM_GB
mcmcmc_mem_gb=$MCMCMC_MEM_GB
SUMMARY

echo "Prepared chunked workflow in $RUN_DIR"
cat "$RUN_DIR/chunked_resource_summary.txt"
echo
echo "PBS scripts:"
printf "  %s\n" "$RUN_DIR"/0*.pbs

if [[ "$SUBMIT" == true ]]; then
    QSUB_BIN="${QSUB_BIN:-}"
    if [[ -z "$QSUB_BIN" ]]; then
        if command -v qsub >/dev/null 2>&1; then
            QSUB_BIN="$(command -v qsub)"
        elif [[ -x /opt/pbs/bin/qsub ]]; then
            QSUB_BIN="/opt/pbs/bin/qsub"
        fi
    fi
    if [[ -z "$QSUB_BIN" || ! -x "$QSUB_BIN" ]]; then
        echo "qsub not found. Generated scripts but did not submit."
        echo "Set QSUB_BIN=/path/to/qsub or add qsub to PATH."
        exit 1
    fi
    plan_id="$(cd "$RUN_DIR" && "$QSUB_BIN" 01_plan.pbs)"
    precompute_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$plan_id" 02_precompute_calculators.pbs)"
    sample_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$precompute_id" 03_sample_nodes.pbs)"
    corner_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$precompute_id" 04_corners.pbs)"
    transform_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$sample_id" 05_transform.pbs)"
    diagonal_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$transform_id" 06_diagonal.pbs)"
    validate_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$diagonal_id":"$corner_id" 07_validate.pbs)"
    mcmcmc_id="$(cd "$RUN_DIR" && "$QSUB_BIN" -W depend=afterok:"$validate_id" 08_mcmcmc.pbs)"
    echo "Submitted:"
    echo "  plan:       $plan_id"
    echo "  precompute: $precompute_id"
    echo "  sample:     $sample_id"
    echo "  corners:    $corner_id"
    echo "  transform:  $transform_id"
    echo "  diagonal:   $diagonal_id"
    echo "  validate:   $validate_id"
    echo "  mcmcmc:     $mcmcmc_id"
else
    echo "Generated only. After reviewing the scripts, submit the dependency chain explicitly:"
    printf 'cd %q\n' "$RUN_DIR"
    cat <<'COMMANDS'
QSUB_BIN="${QSUB_BIN:-qsub}"
plan_id=$("$QSUB_BIN" 01_plan.pbs)
precompute_id=$("$QSUB_BIN" -W depend=afterok:"$plan_id" 02_precompute_calculators.pbs)
sample_id=$("$QSUB_BIN" -W depend=afterok:"$precompute_id" 03_sample_nodes.pbs)
corner_id=$("$QSUB_BIN" -W depend=afterok:"$precompute_id" 04_corners.pbs)
transform_id=$("$QSUB_BIN" -W depend=afterok:"$sample_id" 05_transform.pbs)
diagonal_id=$("$QSUB_BIN" -W depend=afterok:"$transform_id" 06_diagonal.pbs)
validate_id=$("$QSUB_BIN" -W depend=afterok:"$diagonal_id":"$corner_id" 07_validate.pbs)
"$QSUB_BIN" -W depend=afterok:"$validate_id" 08_mcmcmc.pbs
COMMANDS
fi
