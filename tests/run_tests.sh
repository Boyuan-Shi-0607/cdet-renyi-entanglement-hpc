#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tests/build"
cxx="${CXX:-c++}"
source "${repo_root}/tests/build_environment.sh"

mkdir -p "${build_dir}"

common_flags=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -DMP_PRECISION_DIGITS10=100
  -I"${repo_root}/include"
  -I"${repo_root}/tests"
  -I"${boost_include}"
  -pthread
)

tests=(
  test_green_kernel
  test_chebyshev_accuracy
  test_fpm_cdet
  test_chunked_pipeline
  test_integrand_all_subsets
  test_mcmcmc_synthetic
)

for test_name in "${tests[@]}"; do
  echo "[build] ${test_name}"
  "${cxx}" "${common_flags[@]}" \
    "${repo_root}/tests/${test_name}.cpp" \
    -o "${build_dir}/${test_name}"
done

for test_name in "${tests[@]}"; do
  echo "[run] ${test_name}"
  (
    cd "${repo_root}"
    "${build_dir}/${test_name}"
  )
done

echo "All audit tests passed."
