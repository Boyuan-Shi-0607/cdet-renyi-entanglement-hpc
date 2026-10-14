#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_FAST_PRINCIPLE_MINORS_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_FAST_PRINCIPLE_MINORS_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <type_traits>
#include "cdet/utility/my_funcs.h"
#include "cdet/runtime_counters.h"

// Check if T is double or DoubleDouble
template<typename T>
static constexpr bool IS_DOUBLE = std::is_same_v<T, double>;

template<typename T>
static constexpr bool IS_DOUBLEDOUBLE = std::is_same_v<T, DoubleDouble>;

// Helper for absolute value
template<typename T>
inline T abs_helper(const T& x) {
    if constexpr (IS_DOUBLEDOUBLE<T>) {
        return DD_ABS(x);
    } else {
        return std::abs(x);
    }
}

template<typename T>
inline T subtract_product(const T& left, const T& right, const T& base) {
    if constexpr (IS_DOUBLE<T>) {
        return std::fma(-left, right, base);
    } else {
        return base - left * right;
    }
}

template<typename T>
T determinant_principal_minor_from_buffer(const T* A, int n, size_t mask) {
    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if ((mask & (size_t(1) << static_cast<size_t>(i))) != 0U) {
            indices.push_back(i);
        }
    }

    const int k = static_cast<int>(indices.size());
    if (k == 0) {
        return T(1);
    }

    std::vector<T> matrix(static_cast<size_t>(k * k), T(0));
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < k; ++col) {
            matrix[static_cast<size_t>(row * k + col)] =
                A[static_cast<size_t>(indices[row] * n + indices[col])];
        }
    }

    T determinant = T(1);
    int sign = 1;
    for (int col = 0; col < k; ++col) {
        int pivot_row = col;
        T pivot_abs = abs_helper(matrix[static_cast<size_t>(col * k + col)]);
        for (int row = col + 1; row < k; ++row) {
            const T candidate_abs = abs_helper(matrix[static_cast<size_t>(row * k + col)]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if (pivot_abs == T(0)) {
            return T(0);
        }

        if (pivot_row != col) {
            for (int swap_col = col; swap_col < k; ++swap_col) {
                std::swap(matrix[static_cast<size_t>(col * k + swap_col)],
                          matrix[static_cast<size_t>(pivot_row * k + swap_col)]);
            }
            sign = -sign;
        }

        const T pivot = matrix[static_cast<size_t>(col * k + col)];
        determinant *= pivot;
        for (int row = col + 1; row < k; ++row) {
            const T factor = matrix[static_cast<size_t>(row * k + col)] / pivot;
            if (factor == T(0)) {
                continue;
            }
            matrix[static_cast<size_t>(row * k + col)] = T(0);
            for (int update_col = col + 1; update_col < k; ++update_col) {
                matrix[static_cast<size_t>(row * k + update_col)] =
                    subtract_product(factor,
                                     matrix[static_cast<size_t>(col * k + update_col)],
                                     matrix[static_cast<size_t>(row * k + update_col)]);
            }
        }
    }

    return sign < 0 ? -determinant : determinant;
}

template<typename T>
std::vector<T> determinant_principal_minors_from_buffer(const T* A, int n) {
    const size_t subset_count = size_t(1) << static_cast<size_t>(n);
    std::vector<T> pm(subset_count - 1, T(0));
    for (size_t mask = 1; mask < subset_count; ++mask) {
        pm[mask - 1] = determinant_principal_minor_from_buffer(A, n, mask);
    }
    return pm;
}

// FastFeynman published-style principal minors:
// compute all principal minors of A + p I, then remove the uniform shift p
// by multilinearity over all diagonal entries.
template<typename T>
std::vector<T> mat2pm_from_buffer(const T* A, int n, T tol, T thresh = T(-1)) {
    if (n <= 0 || n > 48) {
        std::cerr << "mat2pm: n out of range [1..48]\n";
        return {};
    }

    const size_t subset_count = size_t(1) << static_cast<size_t>(n);
    const size_t pm_size = subset_count - 1;
    std::vector<T> pm(pm_size, T(0));

    T max_abs = T(0);
    const int matrix_size = n * n;
    for (int idx = 0; idx < matrix_size; ++idx) {
        const T entry_abs = abs_helper(A[idx]);
        if (entry_abs > max_abs) {
            max_abs = entry_abs;
        }
    }

    const bool has_explicit_shift = !(thresh < T(0));
    const T pivot_shift = has_explicit_shift ? thresh : max_abs / T(matrix_size);
    if (max_abs == T(0) && pivot_shift == T(0)) {
        return pm;
    }
    const T fallback_epsilon = (tol > T(0) && max_abs > T(0)) ? tol * max_abs : T(0);

    size_t max_matrix_list_size = static_cast<size_t>(matrix_size);
    for (int level = 0; level < n - 1; ++level) {
        const size_t new_size = static_cast<size_t>(n - level - 1);
        const size_t candidate = (size_t(1) << static_cast<size_t>(level + 1)) * new_size * new_size;
        max_matrix_list_size = std::max(max_matrix_list_size, candidate);
    }

    thread_local std::vector<T> old_matrix_list;
    thread_local std::vector<T> new_matrix_list;
    old_matrix_list.resize(max_matrix_list_size);
    new_matrix_list.resize(max_matrix_list_size);
    std::copy_n(A, static_cast<size_t>(matrix_size), old_matrix_list.data());

    thread_local std::vector<T> shifted_minors;
    shifted_minors.assign(subset_count, pivot_shift);
    shifted_minors[0] = T(1);
    shifted_minors[1] += old_matrix_list[0];

    size_t minor_index = 2;
    for (int level = 0; level < n - 1; ++level) {
        const size_t old_size = static_cast<size_t>(n - level);
        const size_t old_matrix_size = old_size * old_size;
        const size_t new_size = old_size - 1;
        const size_t new_matrix_size = new_size * new_size;
        const size_t sets_at_level = size_t(1) << static_cast<size_t>(level);

        for (size_t set = 0; set < sets_at_level; ++set) {
            size_t dst_without = set * new_matrix_size;
            const size_t src = set * old_matrix_size;
            const size_t first_column = src + old_size;
            const size_t trailing_block = first_column + 1;
            size_t dst_with = dst_without + sets_at_level * new_matrix_size;

            const T pivot_denominator = old_matrix_list[src] + pivot_shift;
            const T pivot_denominator_abs = abs_helper(pivot_denominator);
            if (pivot_denominator_abs == T(0) ||
                (fallback_epsilon > T(0) && pivot_denominator_abs <= fallback_epsilon)) {
                record_cdet_fpm_fallback();
                return determinant_principal_minors_from_buffer(A, n);
            }
            const T inverse_pivot = T(1) / pivot_denominator;

            size_t row_start = trailing_block;
            size_t column_start = first_column;
            for (size_t row = 0; row < new_size; ++row) {
                size_t top_row = src + 1;
                const T first_column_factor = old_matrix_list[column_start] * inverse_pivot;
                for (size_t col = 0; col < new_size; ++col) {
                    const T base = old_matrix_list[row_start];
                    new_matrix_list[dst_without] = base;
                    new_matrix_list[dst_with] =
                        subtract_product(old_matrix_list[top_row], first_column_factor, base);

                    ++top_row;
                    ++row_start;
                    ++dst_without;
                    ++dst_with;
                }
                ++row_start;
                column_start += old_size;
            }
        }

        const size_t generated_matrix_count = 2 * sets_at_level;
        for (size_t matrix = 0; matrix < generated_matrix_count; ++matrix) {
            shifted_minors[minor_index++] += new_matrix_list[matrix * new_matrix_size];
        }
        old_matrix_list.swap(new_matrix_list);
    }

    size_t set_step = 1;
    for (int level = 0; level < n; ++level) {
        for (size_t set = set_step + 1; set < 2 * set_step; ++set) {
            shifted_minors[set] *= shifted_minors[set - set_step];
        }
        set_step <<= 1;
    }

    set_step = subset_count;
    for (int level = n; level > 0; --level) {
        const size_t half_step = set_step >> 1;
        for (size_t set = set_step - 1;; --set) {
            for (size_t shifted_set = set; shifted_set < subset_count; shifted_set += set_step) {
                shifted_minors[shifted_set] =
                    subtract_product(pivot_shift, shifted_minors[shifted_set - half_step],
                                     shifted_minors[shifted_set]);
            }
            if (set == half_step) {
                break;
            }
        }
        set_step = half_step;
    }

    for (size_t mask = 1; mask < subset_count; ++mask) {
        pm[mask - 1] = shifted_minors[mask];
    }

    return pm;
}

template<typename T>
std::vector<T> mat2pm(const std::vector<T>& A, int n, T tol, T thresh = T(-1)) {
    if (static_cast<int>(A.size()) != n * n) {
        std::cerr << "mat2pm: matrix size mismatch\n";
        return {};
    }
    return mat2pm_from_buffer(A.data(), n, tol, thresh);
}

// Helper function for popcnt
inline unsigned popcnt(unsigned x) {
    return __builtin_popcount(x);
}

#endif
