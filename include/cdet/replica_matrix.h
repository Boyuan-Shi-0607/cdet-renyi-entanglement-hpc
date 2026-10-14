#ifndef REPLICA_MATRIX_H
#define REPLICA_MATRIX_H

#include "cdet/utility/my_funcs.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef CDET_USE_LAPACK
extern "C" {
void dsyev_(char* jobz,
            char* uplo,
            int* n,
            double* a,
            int* lda,
            double* w,
            double* work,
            int* lwork,
            int* info);

void dgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);

void zgesv_(int* n,
            int* nrhs,
            std::complex<double>* a,
            int* lda,
            int* ipiv,
            std::complex<double>* b,
            int* ldb,
            int* info);

void zgecon_(char* norm,
             int* n,
             std::complex<double>* a,
             int* lda,
             double* anorm,
             double* rcond,
             std::complex<double>* work,
             double* rwork,
             int* info);
}
#endif

template <typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(int rows, int cols, const T& value = T())
        : rows_(rows), cols_(cols), data_(static_cast<size_t>(rows) * static_cast<size_t>(cols), value) {}

    int rows() const {
        return rows_;
    }

    int cols() const {
        return cols_;
    }

    T& operator()(int row, int col) {
        return data_[static_cast<size_t>(row) * static_cast<size_t>(cols_) + static_cast<size_t>(col)];
    }

    const T& operator()(int row, int col) const {
        return data_[static_cast<size_t>(row) * static_cast<size_t>(cols_) + static_cast<size_t>(col)];
    }

private:
    int rows_;
    int cols_;
    std::vector<T> data_;
};

template <typename T>
Matrix<T> identity_matrix(int n, const T& one) {
    Matrix<T> result(n, n, T(0));
    for (int i = 0; i < n; ++i) {
        result(i, i) = one;
    }
    return result;
}

template <typename RealT>
using RealVector = std::vector<RealT>;

template <typename RealT>
struct EigenDecomposition {
    RealVector<RealT> eigenvalues;
    Matrix<RealT> eigenvectors;
};

inline MP_Real mp_tolerance_from_digits(int slack_digits) {
    const int digits = std::numeric_limits<MP_Real>::digits10;
    const int exponent = std::max(4, digits - slack_digits);
    return MP_Real("1e-" + std::to_string(exponent));
}

template <typename RealT>
inline RealT real_abs_value(const RealT& value) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return MP_ABS(value);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DD_ABS(value);
    } else {
        return std::abs(value);
    }
}

template <typename RealT>
inline RealT real_sqrt_value(const RealT& value) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return MP_SQRT(value);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DD_SQRT(value);
    } else {
        return std::sqrt(value);
    }
}

template <typename RealT>
inline RealT real_exp_value(const RealT& value) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return MP_EXP(value);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DD_EXP(value);
    } else {
        return std::exp(value);
    }
}

template <typename RealT>
inline RealT real_cos_value(const RealT& value) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return MP_COS(value);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DD_COS(value);
    } else {
        return std::cos(value);
    }
}

template <typename RealT>
inline MP_Real to_mp_real_value(const RealT& value) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return value;
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return value.to_mp_real();
    } else {
        return MP_Real(value);
    }
}

template <typename RealT>
inline RealT real_tolerance_from_digits(int slack_digits) {
    if constexpr (std::is_same_v<RealT, MP_Real>) {
        return mp_tolerance_from_digits(slack_digits);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return RealT(1e-24);
    } else {
        return RealT(1e-12);
    }
}

template <typename RealT>
inline RealT jacobi_eigen_tolerance(const Matrix<RealT>& matrix) {
    if constexpr (std::is_same_v<RealT, double>) {
        double scale = 0.0;
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int col = 0; col < matrix.cols(); ++col) {
                scale = std::max(scale, std::abs(matrix(row, col)));
            }
        }
        scale = std::max(scale, 1.0);
        return RealT(8.0 * std::numeric_limits<double>::epsilon() * scale);
    } else {
        return real_tolerance_from_digits<RealT>(10);
    }
}

template <typename RealT>
inline EigenDecomposition<RealT> jacobi_eigen_symmetric(const Matrix<RealT>& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::runtime_error("Jacobi diagonalization requires a square matrix");
    }

    const int n = matrix.rows();
    Matrix<RealT> a = matrix;
    Matrix<RealT> v = identity_matrix<RealT>(n, RealT(1));
    const RealT tolerance = jacobi_eigen_tolerance(matrix);
    const int max_sweeps = std::max(20, 20 * n * n);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        RealT max_offdiag = RealT(0);
        int p = 0;
        int q = (n > 1) ? 1 : 0;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const RealT value = real_abs_value(a(i, j));
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (n <= 1 || max_offdiag < tolerance) {
            break;
        }

        const RealT app = a(p, p);
        const RealT aqq = a(q, q);
        const RealT apq = a(p, q);
        if (apq == RealT(0)) {
            continue;
        }

        const RealT tau = (aqq - app) / (RealT(2) * apq);
        RealT t = RealT(0);
        if (tau >= RealT(0)) {
            t = RealT(1) / (tau + real_sqrt_value(RealT(RealT(1) + tau * tau)));
        } else {
            t = -RealT(1) / (real_abs_value(tau) + real_sqrt_value(RealT(RealT(1) + tau * tau)));
        }

        const RealT c = RealT(1) / real_sqrt_value(RealT(RealT(1) + t * t));
        const RealT s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }

            const RealT aik = a(k, p);
            const RealT akq = a(k, q);
            a(k, p) = c * aik - s * akq;
            a(p, k) = a(k, p);
            a(k, q) = s * aik + c * akq;
            a(q, k) = a(k, q);
        }

        a(p, p) = c * c * app - RealT(2) * s * c * apq + s * s * aqq;
        a(q, q) = s * s * app + RealT(2) * s * c * apq + c * c * aqq;
        a(p, q) = RealT(0);
        a(q, p) = RealT(0);

        for (int k = 0; k < n; ++k) {
            const RealT vip = v(k, p);
            const RealT viq = v(k, q);
            v(k, p) = c * vip - s * viq;
            v(k, q) = s * vip + c * viq;
        }

        if (sweep + 1 == max_sweeps) {
            throw std::runtime_error("Jacobi diagonalization did not converge");
        }
    }

    std::vector<std::pair<RealT, int>> ordered;
    ordered.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ordered.emplace_back(a(i, i), i);
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    RealVector<RealT> eigenvalues(static_cast<size_t>(n));
    Matrix<RealT> eigenvectors(n, n, RealT(0));
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = ordered[static_cast<size_t>(new_col)].second;
        eigenvalues[static_cast<size_t>(new_col)] = ordered[static_cast<size_t>(new_col)].first;
        for (int row = 0; row < n; ++row) {
            eigenvectors(row, new_col) = v(row, old_col);
        }
    }

    return {eigenvalues, eigenvectors};
}

#ifdef CDET_USE_LAPACK
inline EigenDecomposition<double> lapack_eigen_symmetric(const Matrix<double>& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::runtime_error("LAPACK diagonalization requires a square matrix");
    }

    int n = matrix.rows();
    std::vector<double> a(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            a[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)] =
                matrix(row, col);
        }
    }

    std::vector<double> eigenvalues(static_cast<size_t>(n), 0.0);
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dsyev_(&jobz, &uplo, &n, a.data(), &lda, eigenvalues.data(), &work_query, &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK dsyev workspace query failed with info=" + std::to_string(info));
    }
    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<size_t>(lwork), 0.0);
    dsyev_(&jobz, &uplo, &n, a.data(), &lda, eigenvalues.data(), work.data(), &lwork, &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK dsyev failed with info=" + std::to_string(info));
    }

    Matrix<double> eigenvectors(n, n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            eigenvectors(row, col) =
                a[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)];
        }
    }
    return {std::move(eigenvalues), std::move(eigenvectors)};
}

inline Matrix<double> blas_selected_row_gram(const Matrix<double>& matrix,
                                             const std::vector<int>& selected_rows) {
    const int source_rows = matrix.rows();
    const int source_cols = matrix.cols();
    if (source_rows != source_cols) {
        throw std::runtime_error("BLAS selected-row Gram expects a square eigenvector matrix");
    }
    const int n = source_cols;
    const int k = static_cast<int>(selected_rows.size());
    Matrix<double> gram(n, n, 0.0);
    if (k == 0) {
        return gram;
    }

    std::vector<double> selected(static_cast<size_t>(k) * static_cast<size_t>(n), 0.0);
    for (int row = 0; row < k; ++row) {
        const int source_row = selected_rows[static_cast<size_t>(row)];
        if (source_row < 0 || source_row >= source_rows) {
            throw std::runtime_error("Selected row is out of range for BLAS Gram matrix");
        }
        for (int col = 0; col < n; ++col) {
            selected[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] =
                matrix(source_row, col);
        }
    }

    std::vector<double> gram_column_major(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    char transa = 'T';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int m = n;
    int n_cols = n;
    int k_rows = k;
    int lda = k;
    int ldb = k;
    int ldc = n;
    dgemm_(&transa,
           &transb,
           &m,
           &n_cols,
           &k_rows,
           &alpha,
           selected.data(),
           &lda,
           selected.data(),
           &ldb,
           &beta,
           gram_column_major.data(),
           &ldc);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            gram(row, col) =
                gram_column_major[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)];
        }
    }
    return gram;
}
#endif

template <typename RealT>
inline EigenDecomposition<RealT> symmetric_eigen_decomposition(const Matrix<RealT>& matrix) {
#ifdef CDET_USE_LAPACK
    if constexpr (std::is_same_v<RealT, double>) {
        return lapack_eigen_symmetric(matrix);
    } else
#endif
    {
        return jacobi_eigen_symmetric(matrix);
    }
}

#endif
