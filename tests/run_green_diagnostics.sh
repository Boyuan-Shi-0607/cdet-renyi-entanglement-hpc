#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tests/build"
artifact_dir="${repo_root}/tests/artifacts/green_kernel"
cxx="${CXX:-c++}"
source "${repo_root}/tests/build_environment.sh"

mkdir -p "${build_dir}" "${artifact_dir}"

flags=(
  -std=c++20
  -O3
  -Wall
  -Wextra
  -Wpedantic
  -DMP_PRECISION_DIGITS10=100
  -DCDET_USE_LAPACK=1
  -I"${repo_root}/include"
  -I"${repo_root}/tests"
  -I"${boost_include}"
  -pthread
)

libraries=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  libraries=(-framework Accelerate)
else
  libraries=(-llapack -lblas)
fi

"${cxx}" "${flags[@]}" \
  "${repo_root}/tests/green_kernel_diagnostics.cpp" \
  -o "${build_dir}/green_kernel_diagnostics" \
  "${libraries[@]}"

(
  cd "${repo_root}"
  "${build_dir}/green_kernel_diagnostics" "${artifact_dir}" "$@"
)

python_bin="${PYTHON:-python3}"
"${python_bin}" "${repo_root}/tests/plot_green_diagnostics.py" "${artifact_dir}"
echo "Green-kernel diagnostics: ${artifact_dir}"
