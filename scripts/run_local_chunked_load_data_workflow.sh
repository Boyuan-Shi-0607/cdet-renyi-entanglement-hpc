#!/usr/bin/env bash
set -euo pipefail

# Local driver that mimics the PBS chunked load-data workflow with a bounded
# number of concurrent worker processes. Run it from a directory containing
# params.txt, or pass RUN_DIR.

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-$PWD}"
TARGET="${TARGET:-$SOURCE_DIR/Aphrodite}"
LOCAL_WORKERS="${LOCAL_WORKERS:-8}"
CHUNK_DIR="${CDET_LOAD_DATA_CHUNK_DIR:-}"
USE_LAPACK="${USE_LAPACK:-0}"
LAPACK_LIBS="${LAPACK_LIBS:-}"

cd "$RUN_DIR"
if [[ -e ../params.txt ]]; then
    echo "Refusing to run: ../params.txt would shadow this directory's params.txt." >&2
    exit 2
fi
mkdir -p logs/chunk_plan logs/chunk_precompute logs/chunk_sample logs/chunk_corners logs/chunk_transform logs/chunk_diagonal logs/chunk_validate logs/chunk_mcmcmc

echo "Local chunked workflow"
echo "  source:  $SOURCE_DIR"
echo "  run dir: $RUN_DIR"
echo "  target:  $TARGET"
echo "  workers: $LOCAL_WORKERS"
echo "  lapack:  USE_LAPACK=$USE_LAPACK"

if [[ "$TARGET" == "$SOURCE_DIR/Aphrodite" ]]; then
    # Keep Make targets relative so a checkout path containing spaces works.
    make -C "$SOURCE_DIR" TARGET=Aphrodite BUILD_DIR=build_hpc USE_LAPACK="$USE_LAPACK" LAPACK_LIBS="$LAPACK_LIBS" -j 1
else
    if [[ "$TARGET" == *[[:space:]]* ]]; then
        echo "A custom Make TARGET cannot contain whitespace; use the default target." >&2
        exit 2
    fi
    make -C "$SOURCE_DIR" TARGET="$TARGET" BUILD_DIR=build_hpc USE_LAPACK="$USE_LAPACK" LAPACK_LIBS="$LAPACK_LIBS" -j 1
fi

echo
echo "[1/8] Planning chunked load-data jobs"
CDET_LOAD_DATA_WORKFLOW=plan "$TARGET" | tee logs/chunk_plan/plan.log

node_jobs="$(awk 'BEGIN{n=0} $0 !~ /^#/ && NF {n++} END{print n}' "${CHUNK_DIR:-$(awk -F= '$1=="LOAD_DATA_CHUNK_DIR"{print $2}' params.txt)}/node_jobs.tsv")"
corner_jobs="$(awk 'BEGIN{n=0} $0 !~ /^#/ && NF {n++} END{print n}' "${CHUNK_DIR:-$(awk -F= '$1=="LOAD_DATA_CHUNK_DIR"{print $2}' params.txt)}/corner_jobs.tsv")"
transform_jobs="$(awk 'BEGIN{n=0} $0 !~ /^#/ && NF {n++} END{print n}' "${CHUNK_DIR:-$(awk -F= '$1=="LOAD_DATA_CHUNK_DIR"{print $2}' params.txt)}/transform_jobs.tsv")"

echo
echo "[2/8] Precomputing calculator cache"
CDET_LOAD_DATA_WORKFLOW=precompute_calculators "$TARGET" | tee logs/chunk_precompute/precompute.log

echo
echo "[3/8] Running $node_jobs node-sample jobs with -P $LOCAL_WORKERS"
seq 1 "$node_jobs" | xargs -n 1 -P "$LOCAL_WORKERS" -I {} \
    sh -c 'CDET_LOAD_DATA_WORKFLOW=sample_nodes CDET_LOAD_DATA_JOB_INDEX="$1" "$2" > "logs/chunk_sample/job_${1}.log" 2>&1' sh {} "$TARGET"

echo
echo "[4/8] Running $corner_jobs corner jobs with -P $LOCAL_WORKERS"
seq 1 "$corner_jobs" | xargs -n 1 -P "$LOCAL_WORKERS" -I {} \
    sh -c 'CDET_LOAD_DATA_WORKFLOW=corners CDET_LOAD_DATA_JOB_INDEX="$1" "$2" > "logs/chunk_corners/job_${1}.log" 2>&1' sh {} "$TARGET"

echo
echo "[5/8] Running $transform_jobs sitepair-transform jobs with -P $LOCAL_WORKERS"
seq 1 "$transform_jobs" | xargs -n 1 -P "$LOCAL_WORKERS" -I {} \
    sh -c 'CDET_LOAD_DATA_WORKFLOW=transform_sitepairs CDET_LOAD_DATA_JOB_INDEX="$1" "$2" > "logs/chunk_transform/job_${1}.log" 2>&1' sh {} "$TARGET"

echo
echo "[6/8] Reducing diagonal observable"
CDET_LOAD_DATA_WORKFLOW=diagonal_reduce "$TARGET" | tee logs/chunk_diagonal/diagonal.log

echo
echo "[7/8] Validating chunks"
CDET_LOAD_DATA_WORKFLOW=validate "$TARGET" | tee logs/chunk_validate/validate.log

echo
echo "[8/8] Running MCMCMC from prebuilt chunks"
CDET_LOAD_DATA_WORKFLOW=load_chunks "$TARGET" | tee logs/chunk_mcmcmc/run.log
