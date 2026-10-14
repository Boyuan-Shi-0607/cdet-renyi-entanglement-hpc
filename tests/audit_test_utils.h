#ifndef CDET_AUDIT_TEST_UTILS_H
#define CDET_AUDIT_TEST_UTILS_H

#include "cdet/fermionic_graded_complex.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

inline void audit_require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void audit_require_close(double actual,
                                double expected,
                                double absolute_tolerance,
                                double relative_tolerance,
                                const std::string& message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > absolute_tolerance + relative_tolerance * scale) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

template <typename RealT>
inline double audit_complex_abs_double(const ComplexNumber<RealT>& value) {
    return static_cast<double>(complex_abs(value));
}

template <typename RealT>
inline double audit_max_complex_matrix_difference(
    const Matrix<ComplexNumber<RealT>>& left,
    const Matrix<ComplexNumber<RealT>>& right) {
    audit_require(left.rows() == right.rows() && left.cols() == right.cols(),
                  "matrix shape mismatch");
    double max_difference = 0.0;
    for (int row = 0; row < left.rows(); ++row) {
        for (int col = 0; col < left.cols(); ++col) {
            max_difference = std::max(
                max_difference,
                audit_complex_abs_double(left(row, col) - right(row, col)));
        }
    }
    return max_difference;
}

template <typename RealT>
inline bool audit_complex_matrix_is_finite(const Matrix<ComplexNumber<RealT>>& matrix) {
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            const auto& value = matrix(row, col);
            if (!std::isfinite(static_cast<double>(value.real)) ||
                !std::isfinite(static_cast<double>(value.imag))) {
                return false;
            }
        }
    }
    return true;
}

inline int audit_test_main(const char* name, void (*test_body)()) {
    try {
        test_body();
        std::cout << "PASS " << name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        return 1;
    }
}

#endif
