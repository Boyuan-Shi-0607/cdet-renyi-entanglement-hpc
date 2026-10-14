#include "audit_test_utils.h"

#include "cdet/replica_free_greens.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using Complex = ComplexNumber<double>;

Matrix<Complex> transform_kernel_to_site(const ReplicaGreenCalculator<double>& calculator,
                                         const Matrix<Complex>& kernel) {
    const int n = calculator.n_sites();
    const Matrix<double>& vectors = calculator.eigenvectors();
    Matrix<Complex> temporary(n, n, Complex(0));
    Matrix<Complex> result(n, n, Complex(0));
    for (int site = 0; site < n; ++site) {
        for (int left_mode = 0; left_mode < n; ++left_mode) {
            for (int right_mode = 0; right_mode < n; ++right_mode) {
                temporary(site, right_mode) +=
                    Complex(vectors(site, left_mode)) * kernel(left_mode, right_mode);
            }
        }
    }
    for (int site1 = 0; site1 < n; ++site1) {
        for (int right_mode = 0; right_mode < n; ++right_mode) {
            for (int site2 = 0; site2 < n; ++site2) {
                result(site1, site2) +=
                    temporary(site1, right_mode) * Complex(vectors(site2, right_mode));
            }
        }
    }
    return result;
}

Matrix<Complex> direct_sector_kernel(const ReplicaGreenCalculator<double>& calculator,
                                     int sector,
                                     double tau,
                                     double tau_prime,
                                     TriangleBranch branch) {
    const int n = calculator.n_sites();
    const double beta_value = calculator.beta();
    const double u = tau / beta_value;
    const double v = tau_prime / beta_value;
    const auto& eigenvalues = calculator.eigenvalues();
    const auto& mixing = calculator.sector_mixing_matrix(sector);
    Matrix<Complex> middle(n, n, Complex(0));
    Matrix<Complex> rhs(n, n, Complex(0));
    std::vector<double> left(static_cast<std::size_t>(n), 0.0);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            middle(row, col) = branch == TriangleBranch::Upper
                                   ? mixing(row, col)
                                   : complex_conj(mixing(col, row));
        }
        const double x = beta_value * eigenvalues[static_cast<std::size_t>(row)];
        if (branch == TriangleBranch::Upper) {
            middle(row, row) += Complex(std::exp(x));
            rhs(row, row) = Complex(std::exp(v * x));
            left[static_cast<std::size_t>(row)] = std::exp((1.0 - u) * x);
        } else {
            const double y = -x;
            middle(row, row) += Complex(std::exp(y));
            rhs(row, row) = Complex(std::exp((1.0 - v) * y));
            left[static_cast<std::size_t>(row)] = -std::exp(u * y);
        }
    }

    Matrix<Complex> solved = solve_matrix_system(middle, rhs);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            solved(row, col) = Complex(left[static_cast<std::size_t>(row)]) * solved(row, col);
        }
    }
    return transform_kernel_to_site(calculator, solved);
}

double scaled_middle_max_abs(const ReplicaGreenCalculator<double>& calculator,
                             int sector,
                             double tau,
                             double tau_prime,
                             TriangleBranch branch) {
    const int n = calculator.n_sites();
    const double beta_value = calculator.beta();
    const double u = tau / beta_value;
    const double v = tau_prime / beta_value;
    const bool upper = branch == TriangleBranch::Upper;
    const double theta = upper ? (u + v) / 2.0 : 1.0 - (u + v) / 2.0;
    const auto& eigenvalues = calculator.eigenvalues();
    const auto& mixing = calculator.sector_mixing_matrix(sector);
    double max_abs = 0.0;
    for (int row = 0; row < n; ++row) {
        const double x_row = beta_value * eigenvalues[static_cast<std::size_t>(row)];
        const double signed_row = upper ? x_row : -x_row;
        const double c_row = std::max(0.0, signed_row);
        for (int col = 0; col < n; ++col) {
            const double x_col = beta_value * eigenvalues[static_cast<std::size_t>(col)];
            const double signed_col = upper ? x_col : -x_col;
            const double c_col = std::max(0.0, signed_col);
            const Complex mixing_value =
                upper ? mixing(row, col) : complex_conj(mixing(col, row));
            Complex value = Complex(std::exp(-theta * c_row)) * mixing_value *
                            Complex(std::exp(-(1.0 - theta) * c_col));
            if (row == col) {
                value += Complex(std::exp(signed_row - c_row));
            }
            max_abs = std::max(max_abs, audit_complex_abs_double(value));
        }
    }
    return max_abs;
}

void run_green_kernel_tests() {
    load_data_condition_check_mode = LinearSolveConditionCheckMode::Off;
    ReplicaGreenCalculator<double> calculator(
        3, 2, 2.3, 0.4, true, ReplicaSubregionMask::FirstRow, 0);

    const std::array<std::pair<double, double>, 2> time_pairs = {
        std::pair<double, double>{1.7, 0.4},
        std::pair<double, double>{0.3, 1.8},
    };
    const std::array<TriangleBranch, 2> branches = {
        TriangleBranch::Upper,
        TriangleBranch::Lower,
    };
    for (int sector = 0; sector < 2; ++sector) {
        for (std::size_t index = 0; index < branches.size(); ++index) {
            const auto [tau, tau_prime] = time_pairs[index];
            const Matrix<Complex> stable =
                calculator.green_sector_matrix(sector, tau, tau_prime, branches[index]);
            const Matrix<Complex> direct =
                direct_sector_kernel(calculator, sector, tau, tau_prime, branches[index]);
            const double difference = audit_max_complex_matrix_difference(stable, direct);
            audit_require(difference < 2.0e-11,
                          "stable replica kernel disagrees with direct theory formula");
        }
    }

    const double equal_tau = 0.37 * calculator.beta();
    for (int sector = 0; sector < 2; ++sector) {
        const Matrix<Complex> upper = calculator.green_sector_matrix(
            sector, equal_tau, equal_tau, TriangleBranch::Upper);
        const Matrix<Complex> lower = calculator.green_sector_matrix(
            sector, equal_tau, equal_tau, TriangleBranch::Lower);
        for (int row = 0; row < calculator.n_sites(); ++row) {
            for (int col = 0; col < calculator.n_sites(); ++col) {
                const Complex expected(row == col ? 1.0 : 0.0);
                audit_require(audit_complex_abs_double((upper(row, col) - lower(row, col)) - expected) <
                                  3.0e-11,
                              "Green-function equal-time jump is not the identity");
            }
        }
    }

    const Matrix<Complex> sector0 = calculator.green_sector_matrix(
        0, 1.4, 0.6, TriangleBranch::Upper);
    const Matrix<Complex> sector1 = calculator.green_sector_matrix(
        1, 1.4, 0.6, TriangleBranch::Upper);
    for (int row = 0; row < calculator.n_sites(); ++row) {
        for (int col = 0; col < calculator.n_sites(); ++col) {
            audit_require(audit_complex_abs_double(sector1(row, col) - complex_conj(sector0(row, col))) <
                              2.0e-11,
                          "the two n=2 sectors are not complex-conjugate partners");
        }
    }

    const double tau_prime = 0.41 * calculator.beta();
    const auto& mask = calculator.subregion_mask();
    for (int sector = 0; sector < 2; ++sector) {
        const Matrix<Complex> at_beta = calculator.green_sector_matrix(
            sector, calculator.beta(), tau_prime, TriangleBranch::Upper);
        const Matrix<Complex> at_zero = calculator.green_sector_matrix(
            sector, 0.0, tau_prime, TriangleBranch::Lower);
        for (int row = 0; row < calculator.n_sites(); ++row) {
            const Complex omega = mask[static_cast<std::size_t>(row)] == 0
                                      ? Complex(1.0)
                                      : (sector == 0 ? Complex(0.0, 1.0)
                                                     : Complex(0.0, -1.0));
            for (int col = 0; col < calculator.n_sites(); ++col) {
                audit_require(audit_complex_abs_double(at_beta(row, col) +
                                                       omega * at_zero(row, col)) < 4.0e-11,
                              "graded temporal boundary condition failed");
            }
        }
    }

    ReplicaGreenCalculator<double> large_beta_calculator(
        4, 4, 200.0, 0.1, true, ReplicaSubregionMask::Checkerboard, 0);
    for (int sector = 0; sector < 2; ++sector) {
        const Matrix<Complex> upper = large_beta_calculator.green_sector_matrix(
            sector, 150.0, 50.0, TriangleBranch::Upper);
        const Matrix<Complex> lower = large_beta_calculator.green_sector_matrix(
            sector, 50.0, 150.0, TriangleBranch::Lower);
        audit_require(audit_complex_matrix_is_finite(upper) &&
                          audit_complex_matrix_is_finite(lower),
                      "stable kernel overflowed at large beta");
        audit_require(scaled_middle_max_abs(large_beta_calculator, sector, 150.0, 50.0,
                                            TriangleBranch::Upper) <= 2.0 + 1.0e-12,
                      "upper scaled middle matrix exceeds its theoretical bound");
        audit_require(scaled_middle_max_abs(large_beta_calculator, sector, 50.0, 150.0,
                                            TriangleBranch::Lower) <= 2.0 + 1.0e-12,
                      "lower scaled middle matrix exceeds its theoretical bound");
    }
}

int main() {
    return audit_test_main("green_kernel", run_green_kernel_tests);
}
