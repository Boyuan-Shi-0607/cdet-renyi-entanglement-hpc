#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tests/build/sanitizers"
cxx="${CXX:-c++}"
source "${repo_root}/tests/build_environment.sh"

mkdir -p "${build_dir}"

flags=(
  -std=c++20
  -O1
  -g
  -Wall
  -Wextra
  -Wpedantic
  -fsanitize=address,undefined
  -fno-omit-frame-pointer
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
  echo "[asan/ubsan build] ${test_name}"
  "${cxx}" "${flags[@]}" \
    "${repo_root}/tests/${test_name}.cpp" \
    -o "${build_dir}/${test_name}"
done

for test_name in "${tests[@]}"; do
  echo "[asan/ubsan run] ${test_name}"
  (
    cd "${repo_root}"
    "${build_dir}/${test_name}"
  )
done

echo "All sanitizer audit tests passed."
