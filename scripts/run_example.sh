#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_name="${1:-3x3-resident}"
case "$case_name" in
    3x3-resident|3x3-chunked) ;;
    *) echo "Usage: $0 [3x3-resident|3x3-chunked] [new-run-directory]" >&2; exit 2 ;;
esac
run_dir="${2:-$repo_root/runs/${case_name}_$(date +%Y%m%d_%H%M%S)}"
if [[ -d "$run_dir" && -n "$(ls -A "$run_dir")" ]]; then
    echo "Refusing to overwrite a nonempty run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
if [[ -e "$run_dir/../params.txt" ]]; then
    echo "Parent params.txt would shadow this run; use another run directory." >&2
    exit 2
fi
cp "$repo_root/examples/$case_name/params.txt" "$run_dir/params.txt"
echo "Run directory: $run_dir"
if [[ "$case_name" == "3x3-chunked" ]]; then
    RUN_DIR="$run_dir" SOURCE_DIR="$repo_root" LOCAL_WORKERS="${LOCAL_WORKERS:-2}" \
        bash "$repo_root/scripts/run_local_chunked_load_data_workflow.sh"
else
    make -C "$repo_root" -j 1
    (cd "$run_dir"; mkdir -p logs histograms; "$repo_root/Aphrodite" 2>&1 | tee logs/run.log)
fi
