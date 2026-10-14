#ifndef FERMIONIC_GRADED_COMPLEX_H
#define FERMIONIC_GRADED_COMPLEX_H

#include "cdet/linear_solve_diagnostics.h"
#include "cdet/replica_matrix.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename RealT>
struct ComplexNumber {
    RealT real;
    RealT imag;

    ComplexNumber() : real(0), imag(0) {}
    ComplexNumber(int value) : real(value), imag(0) {}
    ComplexNumber(const RealT& real_value) : real(real_value), imag(0) {}
    ComplexNumber(const RealT& real_value, const RealT& imag_value) : real(real_value), imag(imag_value) {}

    ComplexNumber& operator+=(const ComplexNumber& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    ComplexNumber& operator-=(const ComplexNumber& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    ComplexNumber& operator*=(const ComplexNumber& other) {
        const RealT new_real = real * other.real - imag * other.imag;
        const RealT new_imag = real * other.imag + imag * other.real;
        real = new_real;
        imag = new_imag;
        return *this;
    }

    ComplexNumber& operator/=(const ComplexNumber& other) {
        const RealT denom = other.real * other.real + other.imag * other.imag;
        if (denom == RealT(0)) {
            throw std::runtime_error("Division by zero in ComplexNumber");
        }

        const RealT new_real = (real * other.real + imag * other.imag) / denom;
        const RealT new_imag = (imag * other.real - real * other.imag) / denom;
        real = new_real;
        imag = new_imag;
        return *this;
    }
};

template <typename RealT>
inline ComplexNumber<RealT> operator+(ComplexNumber<RealT> lhs, const ComplexNumber<RealT>& rhs) {
    lhs += rhs;
    return lhs;
}

template <typename RealT>
inline ComplexNumber<RealT> operator-(ComplexNumber<RealT> lhs, const ComplexNumber<RealT>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <typename RealT>
inline ComplexNumber<RealT> operator*(ComplexNumber<RealT> lhs, const ComplexNumber<RealT>& rhs) {
    lhs *= rhs;
    return lhs;
}

template <typename RealT>
inline ComplexNumber<RealT> operator/(ComplexNumber<RealT> lhs, const ComplexNumber<RealT>& rhs) {
    lhs /= rhs;
    return lhs;
}

template <typename RealT>
inline ComplexNumber<RealT> operator-(const ComplexNumber<RealT>& value) {
    return ComplexNumber<RealT>(-value.real, -value.imag);
}

template <typename RealT>
inline bool operator==(const ComplexNumber<RealT>& lhs, const ComplexNumber<RealT>& rhs) {
    return lhs.real == rhs.real && lhs.imag == rhs.imag;
}

template <typename RealT>
inline bool operator!=(const ComplexNumber<RealT>& lhs, const ComplexNumber<RealT>& rhs) {
    return !(lhs == rhs);
}

template <typename RealT>
inline ComplexNumber<RealT> complex_conj(const ComplexNumber<RealT>& value) {
    return ComplexNumber<RealT>(value.real, -value.imag);
}

template <typename RealT>
inline RealT complex_abs(const ComplexNumber<RealT>& value) {
    return real_sqrt_value(RealT(value.real * value.real + value.imag * value.imag));
}

template <typename RealT>
inline ComplexNumber<RealT> complex_i() {
    return ComplexNumber<RealT>(RealT(0), RealT(1));
}

template <typename RealT>
inline std::ostream& operator<<(std::ostream& os, const ComplexNumber<RealT>& value) {
    os << '(' << value.real;
    if (value.imag >= RealT(0)) {
        os << " + " << value.imag << 'i';
    } else {
        os << " - " << (-value.imag) << 'i';
    }
    os << ')';
    return os;
}

template <typename RealT>
inline RealT real_part_if_nearly_real(const ComplexNumber<RealT>& value) {
    const RealT tolerance = real_tolerance_from_digits<RealT>(18);
    const RealT scale = RealT(1) + real_abs_value(value.real);
    if (real_abs_value(value.imag) > tolerance * scale) {
        throw std::runtime_error("Complex value expected to be real within tolerance");
    }
    return value.real;
}

template <typename RealT>
inline double complex_matrix_one_norm_for_condition(
    const Matrix<ComplexNumber<RealT>>& matrix) {
    double norm = 0.0;
    for (int col = 0; col < matrix.cols(); ++col) {
        double column_sum = 0.0;
        for (int row = 0; row < matrix.rows(); ++row) {
            column_sum += static_cast<double>(complex_abs(matrix(row, col)));
        }
        norm = std::max(norm, column_sum);
    }
    return norm;
}

template <typename RealT>
inline std::vector<double> diagonal_rhs_abs_for_condition(
    const Matrix<ComplexNumber<RealT>>& rhs,
    const LinearSolveDiagnosticContext& diagnostic_context) {
    if (!diagnostic_context.rhs_is_diagonal || rhs.rows() != rhs.cols()) {
        return {};
    }

    std::vector<double> diagonal_abs(static_cast<size_t>(rhs.cols()), 0.0);
    for (int col = 0; col < rhs.cols(); ++col) {
        const double value = static_cast<double>(complex_abs(rhs(col, col)));
        if (!(value > 0.0) || !std::isfinite(value)) {
            return {};
        }
        diagonal_abs[static_cast<size_t>(col)] = value;
    }
    return diagonal_abs;
}

template <typename RealT>
inline double condition_rcond_from_diagonal_rhs_solution(
    double matrix_one_norm,
    const std::vector<double>& rhs_diagonal_abs,
    const Matrix<ComplexNumber<RealT>>& solution) {
    if (!(matrix_one_norm > 0.0) || !std::isfinite(matrix_one_norm) ||
        rhs_diagonal_abs.size() != static_cast<size_t>(solution.cols())) {
        return 0.0;
    }

    double inverse_one_norm = 0.0;
    for (int col = 0; col < solution.cols(); ++col) {
        const double rhs_abs = rhs_diagonal_abs[static_cast<size_t>(col)];
        if (!(rhs_abs > 0.0) || !std::isfinite(rhs_abs)) {
            return 0.0;
        }

        double inverse_column_sum = 0.0;
        for (int row = 0; row < solution.rows(); ++row) {
            inverse_column_sum += static_cast<double>(complex_abs(solution(row, col))) / rhs_abs;
        }
        inverse_one_norm = std::max(inverse_one_norm, inverse_column_sum);
    }

    const double condition_number = matrix_one_norm * inverse_one_norm;
    if (!(condition_number > 0.0) || !std::isfinite(condition_number)) {
        return 0.0;
    }
    return 1.0 / condition_number;
}

template <typename RealT>
inline Matrix<ComplexNumber<RealT>> solve_matrix_system(Matrix<ComplexNumber<RealT>> matrix,
                                                        Matrix<ComplexNumber<RealT>> rhs,
                                                        const LinearSolveDiagnosticContext& diagnostic_context = {}) {
    if (matrix.rows() != matrix.cols()) {
        throw std::runtime_error("Matrix solve requires a square coefficient matrix");
    }
    if (matrix.rows() != rhs.rows()) {
        throw std::runtime_error("Matrix solve shape mismatch");
    }

    const int n = matrix.rows();
    const int rhs_cols = rhs.cols();
    const bool condition_check_enabled = linear_solve_condition_check_enabled();
    const double matrix_one_norm =
        condition_check_enabled ? complex_matrix_one_norm_for_condition(matrix) : 0.0;
    const std::vector<double> rhs_diagonal_abs =
        condition_check_enabled
            ? diagonal_rhs_abs_for_condition(rhs, diagnostic_context)
            : std::vector<double>{};

#ifdef CDET_USE_LAPACK
    if constexpr (std::is_same_v<RealT, double>) {
        std::vector<std::complex<double>> a(static_cast<size_t>(n) * static_cast<size_t>(n));
        std::vector<std::complex<double>> b(static_cast<size_t>(n) * static_cast<size_t>(rhs_cols));
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const ComplexNumber<double>& value = matrix(row, col);
                a[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)] =
                    std::complex<double>(value.real, value.imag);
            }
            for (int col = 0; col < rhs_cols; ++col) {
                const ComplexNumber<double>& value = rhs(row, col);
                b[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)] =
                    std::complex<double>(value.real, value.imag);
            }
        }

        std::vector<int> pivots(static_cast<size_t>(n));
        int n_lapack = n;
        int nrhs_lapack = rhs_cols;
        int lda = n;
        int ldb = n;
        int info = 0;
        zgesv_(&n_lapack,
               &nrhs_lapack,
               a.data(),
               &lda,
               pivots.data(),
               b.data(),
               &ldb,
               &info);
        if (info < 0) {
            throw std::runtime_error("LAPACK zgesv received an invalid argument");
        }
        if (info > 0) {
            throw std::runtime_error("Complex matrix solve is singular within LAPACK zgesv");
        }

        Matrix<ComplexNumber<double>> result(n, rhs_cols, ComplexNumber<double>(0));
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < rhs_cols; ++col) {
                const std::complex<double>& value =
                    b[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)];
                result(row, col) = ComplexNumber<double>(value.real(), value.imag());
            }
        }
        if (condition_check_enabled) {
            if (!rhs_diagonal_abs.empty()) {
                const double rcond = condition_rcond_from_diagonal_rhs_solution(
                    matrix_one_norm, rhs_diagonal_abs, result);
                handle_linear_solve_condition_estimate(
                    diagnostic_context, n, rcond,
                    "diagonal-RHS exact one-norm");
            } else if (matrix_one_norm > 0.0) {
                char norm = '1';
                int n_lapack_condition = n;
                int lda_condition = n;
                double anorm = matrix_one_norm;
                double rcond = 0.0;
                int condition_info = 0;
                std::vector<std::complex<double>> work(static_cast<size_t>(2 * n));
                std::vector<double> rwork(static_cast<size_t>(2 * n), 0.0);
                zgecon_(&norm,
                        &n_lapack_condition,
                        a.data(),
                        &lda_condition,
                        &anorm,
                        &rcond,
                        work.data(),
                        rwork.data(),
                        &condition_info);
                if (condition_info < 0) {
                    throw std::runtime_error(
                        "LAPACK zgecon received an invalid argument");
                }
                handle_linear_solve_condition_estimate(
                    diagnostic_context, n, rcond,
                    "LAPACK zgecon one-norm estimate");
            }
        }
        return result;
    }
#endif

    const RealT tolerance = real_tolerance_from_digits<RealT>(8);
    RealT min_pivot_abs = RealT(0);
    bool saw_pivot = false;

    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        RealT pivot_abs = complex_abs(matrix(col, col));
        for (int row = col + 1; row < n; ++row) {
            const RealT candidate_abs = complex_abs(matrix(row, col));
            if (candidate_abs > pivot_abs) {
                pivot_row = row;
                pivot_abs = candidate_abs;
            }
        }
        if (!saw_pivot || pivot_abs < min_pivot_abs) {
            min_pivot_abs = pivot_abs;
            saw_pivot = true;
        }

        if (pivot_abs <= tolerance) {
            throw std::runtime_error("Complex matrix solve is singular within the requested precision");
        }

        if (pivot_row != col) {
            for (int j = 0; j < n; ++j) {
                std::swap(matrix(col, j), matrix(pivot_row, j));
            }
            for (int j = 0; j < rhs_cols; ++j) {
                std::swap(rhs(col, j), rhs(pivot_row, j));
            }
        }

        const ComplexNumber<RealT> pivot = matrix(col, col);
        for (int j = col; j < n; ++j) {
            matrix(col, j) /= pivot;
        }
        for (int j = 0; j < rhs_cols; ++j) {
            rhs(col, j) /= pivot;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }

            const ComplexNumber<RealT> factor = matrix(row, col);
            if (factor == ComplexNumber<RealT>(0)) {
                continue;
            }

            for (int j = col; j < n; ++j) {
                matrix(row, j) -= factor * matrix(col, j);
            }
            for (int j = 0; j < rhs_cols; ++j) {
                rhs(row, j) -= factor * rhs(col, j);
            }
        }
    }

    if (condition_check_enabled) {
        if (!rhs_diagonal_abs.empty()) {
            const double rcond = condition_rcond_from_diagonal_rhs_solution(
                matrix_one_norm, rhs_diagonal_abs, rhs);
            handle_linear_solve_condition_estimate(
                diagnostic_context, n, rcond,
                "diagonal-RHS exact one-norm");
        } else if (saw_pivot && matrix_one_norm > 0.0) {
            const double rcond_proxy =
                static_cast<double>(min_pivot_abs) / matrix_one_norm;
            handle_linear_solve_condition_estimate(
                diagnostic_context, n, rcond_proxy,
                "pivot/anorm proxy");
        }
    }

    return rhs;
}

using ComplexMP = ComplexNumber<MP_Real>;
using ComplexDD = ComplexNumber<DoubleDouble>;

#endif
