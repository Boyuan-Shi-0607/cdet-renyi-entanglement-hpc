#include "audit_test_utils.h"

#include "cdet/replica_free_greens.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Complex = ComplexNumber<double>;

struct MiddleDiagnostics {
    Matrix<Complex> matrix;
    double max_abs = 0.0;
    double min_positive_abs = std::numeric_limits<double>::infinity();
    double rms_abs = 0.0;
    double rcond_one_norm = 0.0;
};

double matrix_one_norm(const Matrix<Complex>& matrix) {
    double norm = 0.0;
    for (int col = 0; col < matrix.cols(); ++col) {
        double sum = 0.0;
        for (int row = 0; row < matrix.rows(); ++row) {
            sum += audit_complex_abs_double(matrix(row, col));
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

MiddleDiagnostics build_scaled_middle(const ReplicaGreenCalculator<double>& calculator,
                                      int sector,
                                      double u,
                                      double v,
                                      TriangleBranch branch) {
    const int n = calculator.n_sites();
    const bool upper = branch == TriangleBranch::Upper;
    const double theta = upper ? (u + v) / 2.0 : 1.0 - (u + v) / 2.0;
    const auto& eigenvalues = calculator.eigenvalues();
    const auto& mixing = calculator.sector_mixing_matrix(sector);
    MiddleDiagnostics diagnostics{Matrix<Complex>(n, n, Complex(0))};
    long double square_sum = 0.0L;

    for (int row = 0; row < n; ++row) {
        const double x_row = calculator.beta() * eigenvalues[static_cast<std::size_t>(row)];
        const double signed_row = upper ? x_row : -x_row;
        const double c_row = std::max(0.0, signed_row);
        for (int col = 0; col < n; ++col) {
            const double x_col = calculator.beta() * eigenvalues[static_cast<std::size_t>(col)];
            const double signed_col = upper ? x_col : -x_col;
            const double c_col = std::max(0.0, signed_col);
            const Complex sector_value =
                upper ? mixing(row, col) : complex_conj(mixing(col, row));
            Complex value = Complex(std::exp(-theta * c_row)) * sector_value *
                            Complex(std::exp(-(1.0 - theta) * c_col));
            if (row == col) {
                value += Complex(std::exp(signed_row - c_row));
            }
            diagnostics.matrix(row, col) = value;
            const double magnitude = audit_complex_abs_double(value);
            diagnostics.max_abs = std::max(diagnostics.max_abs, magnitude);
            if (magnitude > 0.0) {
                diagnostics.min_positive_abs =
                    std::min(diagnostics.min_positive_abs, magnitude);
            }
            square_sum += static_cast<long double>(magnitude) * magnitude;
        }
    }
    diagnostics.rms_abs = std::sqrt(static_cast<double>(
        square_sum / static_cast<long double>(n) / static_cast<long double>(n)));

    Matrix<Complex> identity = identity_matrix<Complex>(n, Complex(1));
    const double norm = matrix_one_norm(diagnostics.matrix);
    const Matrix<Complex> inverse = solve_matrix_system(diagnostics.matrix, identity);
    const double inverse_norm = matrix_one_norm(inverse);
    diagnostics.rcond_one_norm =
        norm > 0.0 && inverse_norm > 0.0 ? 1.0 / (norm * inverse_norm) : 0.0;
    return diagnostics;
}

double sector_unitarity_error(const Matrix<Complex>& mixing) {
    double max_error = 0.0;
    for (int row = 0; row < mixing.rows(); ++row) {
        for (int col = 0; col < mixing.cols(); ++col) {
            Complex value(0);
            for (int inner = 0; inner < mixing.rows(); ++inner) {
                value += complex_conj(mixing(inner, row)) * mixing(inner, col);
            }
            value -= Complex(row == col ? 1.0 : 0.0);
            max_error = std::max(max_error, audit_complex_abs_double(value));
        }
    }
    return max_error;
}

double matrix_max_abs(const Matrix<Complex>& matrix) {
    double result = 0.0;
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            result = std::max(result, audit_complex_abs_double(matrix(row, col)));
        }
    }
    return result;
}

void write_magnitude_csv(const Matrix<Complex>& matrix,
                         const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("could not write " + path.string());
    }
    out << std::setprecision(17);
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col != 0) {
                out << ',';
            }
            out << audit_complex_abs_double(matrix(row, col));
        }
        out << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        load_data_condition_check_mode = LinearSolveConditionCheckMode::Off;
        const std::filesystem::path output_dir =
            argc > 1 ? std::filesystem::path(argv[1])
                     : std::filesystem::path("tests") / "artifacts" / "green_kernel";
        std::filesystem::create_directories(output_dir);
        std::ofstream summary(output_dir / "green_kernel_metrics.csv");
        if (!summary.is_open()) {
            throw std::runtime_error("could not write green-kernel metrics");
        }
        summary << "linear_size,n_sites,branch,sector,u,v,eigen_min,eigen_max,"
                   "middle_max_abs,middle_min_positive_abs,middle_rms_abs,middle_rcond_1,"
                   "sector_unitarity_max_error,kernel_max_abs,kernel_finite\n";
        summary << std::setprecision(17);

        std::vector<int> sizes = {3, 10, 20};
        if (argc > 2) {
            sizes.clear();
            for (int index = 2; index < argc; ++index) {
                sizes.push_back(std::stoi(argv[index]));
            }
        }

        for (int linear_size : sizes) {
            const int radius = std::max(1, linear_size / 5);
            std::cout << "diagnosing " << linear_size << "x" << linear_size
                      << " (origin-square radius " << radius << ")\n";
            ReplicaGreenCalculator<double> calculator(
                linear_size,
                linear_size,
                8.0,
                -0.4,
                true,
                ReplicaSubregionMask::OriginSquare,
                radius);
            const auto [eigen_min_it, eigen_max_it] = std::minmax_element(
                calculator.eigenvalues().begin(), calculator.eigenvalues().end());

            for (TriangleBranch branch : {TriangleBranch::Upper, TriangleBranch::Lower}) {
                const bool upper = branch == TriangleBranch::Upper;
                const double u = upper ? 0.75 : 0.25;
                const double v = upper ? 0.25 : 0.75;
                for (int sector = 0; sector < 2; ++sector) {
                    MiddleDiagnostics diagnostics =
                        build_scaled_middle(calculator, sector, u, v, branch);
                    const Matrix<Complex> kernel = calculator.green_sector_matrix(
                        sector, u * calculator.beta(), v * calculator.beta(), branch);
                    summary << linear_size << ','
                            << calculator.n_sites() << ','
                            << (upper ? "upper" : "lower") << ','
                            << sector << ',' << u << ',' << v << ','
                            << *eigen_min_it << ',' << *eigen_max_it << ','
                            << diagnostics.max_abs << ','
                            << diagnostics.min_positive_abs << ','
                            << diagnostics.rms_abs << ','
                            << diagnostics.rcond_one_norm << ','
                            << sector_unitarity_error(calculator.sector_mixing_matrix(sector)) << ','
                            << matrix_max_abs(kernel) << ','
                            << (audit_complex_matrix_is_finite(kernel) ? 1 : 0) << '\n';
                    audit_require(diagnostics.max_abs <= 2.0 + 1.0e-12,
                                  "scaled middle matrix violated the entry bound");
                    audit_require(audit_complex_matrix_is_finite(kernel),
                                  "Green kernel contains a non-finite value");

                    if (sector == 0) {
                        std::ostringstream name;
                        name << "middle_abs_L" << linear_size << '_'
                             << (upper ? "upper" : "lower") << "_sector0.csv";
                        write_magnitude_csv(diagnostics.matrix, output_dir / name.str());
                    }
                }
            }
        }
        std::cout << "wrote diagnostics to " << output_dir << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "green-kernel diagnostics failed: " << error.what() << '\n';
        return 1;
    }
}
