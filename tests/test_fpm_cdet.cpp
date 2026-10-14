#include "audit_test_utils.h"

#include "cdet/integrand_replica_randomly_sampled.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

double partition_cumulant_recursive(std::uint64_t remaining,
                                    const std::vector<double>& moments,
                                    int block_count,
                                    double product) {
    if (remaining == 0U) {
        double factorial = 1.0;
        for (int value = 2; value <= block_count - 1; ++value) {
            factorial *= static_cast<double>(value);
        }
        const double sign = ((block_count - 1) % 2 == 0) ? 1.0 : -1.0;
        return sign * factorial * product;
    }

    const std::uint64_t anchor = remaining & (~remaining + 1U);
    const std::uint64_t rest = remaining ^ anchor;
    double sum = 0.0;
    for (std::uint64_t subset = rest;; subset = (subset - 1U) & rest) {
        const std::uint64_t block = anchor | subset;
        sum += partition_cumulant_recursive(
            remaining ^ block,
            moments,
            block_count + 1,
            product * moments[static_cast<std::size_t>(block)]);
        if (subset == 0U) {
            break;
        }
    }
    return sum;
}

double partition_cumulant(std::uint64_t mask, const std::vector<double>& moments) {
    return partition_cumulant_recursive(mask, moments, 0, 1.0);
}

void compare_fpm_with_reference(const std::vector<double>& matrix,
                                int n,
                                double tolerance,
                                double shift,
                                double comparison_tolerance) {
    const std::vector<double> fast = mat2pm(matrix, n, tolerance, shift);
    const std::vector<double> reference =
        determinant_principal_minors_from_buffer(matrix.data(), n);
    audit_require(fast.size() == reference.size(), "principal-minor output size mismatch");
    for (std::size_t index = 0; index < fast.size(); ++index) {
        audit_require_close(fast[index], reference[index], comparison_tolerance,
                            comparison_tolerance,
                            "fast principal minor disagrees with pivoted determinant");
    }
}

void run_fpm_cdet_tests() {
    std::mt19937_64 rng(0x8ac341eeULL);
    std::uniform_real_distribution<double> distribution(-0.8, 0.8);

    for (int n = 1; n <= 8; ++n) {
        std::vector<double> matrix(static_cast<std::size_t>(n * n), 0.0);
        for (double& value : matrix) {
            value = distribution(rng);
        }
        for (int diagonal = 0; diagonal < n; ++diagonal) {
            matrix[static_cast<std::size_t>(diagonal * n + diagonal)] +=
                1.2 + 0.1 * diagonal;
        }
        compare_fpm_with_reference(matrix, n, 0.0, -1.0, 2.0e-10);
        compare_fpm_with_reference(matrix, n, 0.0, 0.37, 2.0e-10);
        compare_fpm_with_reference(matrix, n, 1.0e-5, -1.0, 2.0e-10);
    }

    const std::vector<double> pivot_pathology = {
        0.0, 1.0, 0.2,
        1.0, 0.3, -0.1,
        0.2, -0.1, 1.4,
    };
    compare_fpm_with_reference(pivot_pathology, 3, 1.0e-5, 0.0, 2.0e-13);

    for (int order = 1; order <= 7; ++order) {
        const std::size_t subset_count = std::size_t{1} << order;
        std::vector<double> up(subset_count - 1U, 0.0);
        std::vector<double> down(subset_count - 1U, 0.0);
        std::vector<double> moments(subset_count, 0.0);
        for (std::size_t mask = 1; mask < subset_count; ++mask) {
            up[mask - 1U] = 0.6 + 0.08 * distribution(rng);
            down[mask - 1U] = 0.7 + 0.08 * distribution(rng);
            moments[mask] = up[mask - 1U] * down[mask - 1U];
        }

        const std::vector<double> connected =
            compute_dp_ztransform_all_subsets(up, down, order, 0.0, 0.0);
        for (std::size_t mask = 1; mask < subset_count; ++mask) {
            const double reference = partition_cumulant(mask, moments);
            audit_require_close(connected[mask], reference, 2.0e-11, 2.0e-11,
                                "CDet subset recursion disagrees with set-partition cumulant");
        }
    }

    for (int order = 2; order <= 9; ++order) {
        const std::size_t subset_count = std::size_t{1} << order;
        std::vector<double> up(subset_count - 1U, 1.0);
        std::vector<double> down(subset_count - 1U, 1.0);
        const std::vector<double> connected =
            compute_dp_ztransform_all_subsets(up, down, order, 2.0e-16, 1.0e4);
        audit_require(std::abs(connected.back()) < 1.0e-14,
                      "roundoff-filtered disconnected constant moment did not cancel");
    }

    const auto upper_corner = prepare_green_lookup_descriptor(
        0, 1, 0.5e-13, 0.25e-13, 0, 0, 0, 0,
        8.0, 1.0e-13, 25, 25, 3, 3, true);
    const auto upper_regular = prepare_green_lookup_descriptor(
        0, 1, 2.0e-13, 1.0e-13, 0, 0, 0, 0,
        8.0, 1.0e-13, 25, 25, 3, 3, true);
    const auto lower_corner = prepare_green_lookup_descriptor(
        0, 1, 0.25e-13, 0.5e-13, 0, 0, 0, 0,
        8.0, 1.0e-13, 25, 25, 3, 3, false);
    audit_require(upper_corner.use_corner && lower_corner.use_corner,
                  "BRANCH_EPSILON did not select the exact corner path");
    audit_require(!upper_regular.use_corner,
                  "BRANCH_EPSILON incorrectly removed a regular triangle point");
}

int main() {
    return audit_test_main("fpm_cdet", run_fpm_cdet_tests);
}
