#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_G_FUNCS_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_G_FUNCS_H

#include "cdet/replica_chebyshev_tables.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

constexpr int CHEBYSHEV_FAST_BUFFER_LIMIT = 32;

#ifndef CDET_USE_CHEB_SIMD
#define CDET_USE_CHEB_SIMD 0
#endif

#if CDET_USE_CHEB_SIMD > 0 && (defined(__GNUC__) || defined(__clang__))
using ChebyshevDoublePairLane = double __attribute__((vector_size(16)));
#if CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
using ChebyshevDoubleQuadLane = double __attribute__((vector_size(32)));
#endif
#if CDET_USE_CHEB_SIMD == 8
using ChebyshevDoubleOctLane = double __attribute__((vector_size(64)));
#endif

inline ChebyshevDoublePairLane make_chebyshev_double_pair_lane(double up, double down) {
    return ChebyshevDoublePairLane{up, down};
}

inline std::pair<double, double> unpack_chebyshev_double_pair_lane(ChebyshevDoublePairLane value) {
    alignas(16) double lanes[2];
    std::memcpy(lanes, &value, sizeof(lanes));
    return {lanes[0], lanes[1]};
}

inline ChebyshevDoublePairLane load_chebyshev_double_lane(const double* values) {
    ChebyshevDoublePairLane lane;
    std::memcpy(&lane, values, sizeof(lane));
    return lane;
}

#if CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
inline ChebyshevDoubleQuadLane load_chebyshev_double_quad_lane(const double* values) {
    ChebyshevDoubleQuadLane lane;
    std::memcpy(&lane, values, sizeof(lane));
    return lane;
}
#endif

#if CDET_USE_CHEB_SIMD == 8
inline ChebyshevDoubleOctLane load_chebyshev_double_oct_lane(const double* values) {
    ChebyshevDoubleOctLane lane;
    std::memcpy(&lane, values, sizeof(lane));
    return lane;
}
#endif

inline double horizontal_sum_chebyshev_double_lane(ChebyshevDoublePairLane value) {
    alignas(16) double lanes[2];
    std::memcpy(lanes, &value, sizeof(lanes));
    return lanes[0] + lanes[1];
}

#if CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
inline double horizontal_sum_chebyshev_double_quad_lane(ChebyshevDoubleQuadLane value) {
    alignas(32) double lanes[4];
    std::memcpy(lanes, &value, sizeof(lanes));
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif

#if CDET_USE_CHEB_SIMD == 8
inline double horizontal_sum_chebyshev_double_oct_lane(ChebyshevDoubleOctLane value) {
    alignas(64) double lanes[8];
    std::memcpy(lanes, &value, sizeof(lanes));
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
}
#endif

template <std::size_t Count>
inline void fill_chebyshev_basis(double x, std::array<double, Count>& basis) {
    if constexpr (Count > 0) {
        basis[0] = 1.0;
    }
    if constexpr (Count > 1) {
        basis[1] = x;
    }
    if constexpr (Count > 2) {
        const double two_x = 2.0 * x;
        for (std::size_t degree = 2; degree < Count; ++degree) {
            basis[static_cast<size_t>(degree)] =
                two_x * basis[static_cast<size_t>(degree - 1)] -
                basis[static_cast<size_t>(degree - 2)];
        }
    }
}

template <std::size_t Count>
inline double dot_chebyshev_coefficients_lane2(const double* coeffs,
                                               const double* basis) {
    ChebyshevDoublePairLane acc = make_chebyshev_double_pair_lane(0.0, 0.0);
    std::size_t index = 0;
    for (; index + 1 < Count; index += 2) {
        acc += load_chebyshev_double_lane(coeffs + index) *
               load_chebyshev_double_lane(basis + index);
    }

    double result = horizontal_sum_chebyshev_double_lane(acc);
    if constexpr ((Count % 2) != 0) {
        result += coeffs[Count - 1] * basis[Count - 1];
    }
    return result;
}

template <std::size_t Count>
inline double dot_chebyshev_coefficients_lane4(const double* coeffs,
                                               const double* basis) {
#if CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
    ChebyshevDoubleQuadLane acc{};
    std::size_t index = 0;
    for (; index + 3 < Count; index += 4) {
        acc += load_chebyshev_double_quad_lane(coeffs + index) *
               load_chebyshev_double_quad_lane(basis + index);
    }

    double result = horizontal_sum_chebyshev_double_quad_lane(acc);
    for (; index < Count; ++index) {
        result += coeffs[index] * basis[index];
    }
    return result;
#else
    return dot_chebyshev_coefficients_lane2<Count>(coeffs, basis);
#endif
}

template <std::size_t Count>
inline double dot_chebyshev_coefficients_lane8(const double* coeffs,
                                               const double* basis) {
#if CDET_USE_CHEB_SIMD == 8
    ChebyshevDoubleOctLane acc{};
    std::size_t index = 0;
    for (; index + 7 < Count; index += 8) {
        acc += load_chebyshev_double_oct_lane(coeffs + index) *
               load_chebyshev_double_oct_lane(basis + index);
    }

    double result = horizontal_sum_chebyshev_double_oct_lane(acc);
    for (; index < Count; ++index) {
        result += coeffs[index] * basis[index];
    }
    return result;
#else
    return dot_chebyshev_coefficients_lane4<Count>(coeffs, basis);
#endif
}

template <int Degree>
inline ChebyshevDoublePairLane clenshaw_fixed_degree_pair_lane(const double* coeffs_up,
                                                               const double* coeffs_down,
                                                               double x) {
    if constexpr (Degree < 0) {
        return make_chebyshev_double_pair_lane(0.0, 0.0);
    } else if constexpr (Degree == 0) {
        return make_chebyshev_double_pair_lane(coeffs_up[0], coeffs_down[0]);
    } else if constexpr (Degree == 1) {
        const ChebyshevDoublePairLane coeff0 =
            make_chebyshev_double_pair_lane(coeffs_up[0], coeffs_down[0]);
        const ChebyshevDoublePairLane coeff1 =
            make_chebyshev_double_pair_lane(coeffs_up[1], coeffs_down[1]);
        const ChebyshevDoublePairLane x_lane = make_chebyshev_double_pair_lane(x, x);
        return coeff0 + coeff1 * x_lane;
    } else {
        const ChebyshevDoublePairLane two_x = make_chebyshev_double_pair_lane(2.0 * x, 2.0 * x);
        ChebyshevDoublePairLane b_kplus1 = make_chebyshev_double_pair_lane(0.0, 0.0);
        ChebyshevDoublePairLane b_kplus2 = make_chebyshev_double_pair_lane(0.0, 0.0);

        for (int k = Degree; k >= 1; --k) {
            const ChebyshevDoublePairLane coeff =
                make_chebyshev_double_pair_lane(coeffs_up[k], coeffs_down[k]);
            const ChebyshevDoublePairLane b_k = coeff + two_x * b_kplus1 - b_kplus2;
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }

        const ChebyshevDoublePairLane coeff0 =
            make_chebyshev_double_pair_lane(coeffs_up[0], coeffs_down[0]);
        const ChebyshevDoublePairLane x_lane = make_chebyshev_double_pair_lane(x, x);
        return coeff0 + b_kplus1 * x_lane - b_kplus2;
    }
}

template <int Degree>
inline ChebyshevDoublePairLane clenshaw_fixed_degree_lane_contiguous(
    const ChebyshevDoublePairLane* coeffs,
    double x) {
    if constexpr (Degree < 0) {
        return make_chebyshev_double_pair_lane(0.0, 0.0);
    } else if constexpr (Degree == 0) {
        return coeffs[0];
    } else if constexpr (Degree == 1) {
        const ChebyshevDoublePairLane x_lane = make_chebyshev_double_pair_lane(x, x);
        return coeffs[0] + coeffs[1] * x_lane;
    } else {
        const ChebyshevDoublePairLane two_x = make_chebyshev_double_pair_lane(2.0 * x, 2.0 * x);
        ChebyshevDoublePairLane b_kplus1 = make_chebyshev_double_pair_lane(0.0, 0.0);
        ChebyshevDoublePairLane b_kplus2 = make_chebyshev_double_pair_lane(0.0, 0.0);

        for (int k = Degree; k >= 1; --k) {
            const ChebyshevDoublePairLane b_k = coeffs[k] + two_x * b_kplus1 - b_kplus2;
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }

        const ChebyshevDoublePairLane x_lane = make_chebyshev_double_pair_lane(x, x);
        return coeffs[0] + b_kplus1 * x_lane - b_kplus2;
    }
}

template <int ChebNu, int ChebNv>
inline std::pair<double, double> chebyshev_sum_2d_pair_fixed_double_simd(const double* coeffs_up_ptr,
                                                                         const double* coeffs_down_ptr,
                                                                         double u,
                                                                         double v) {
    std::array<ChebyshevDoublePairLane, ChebNu> row_values{};

    for (int u_index = 0; u_index < ChebNu; ++u_index) {
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(ChebNv);
        row_values[static_cast<size_t>(u_index)] =
            clenshaw_fixed_degree_pair_lane<ChebNv - 1>(
                coeffs_up_ptr + row_offset,
                coeffs_down_ptr + row_offset,
                v);
    }

    const ChebyshevDoublePairLane result =
        clenshaw_fixed_degree_lane_contiguous<ChebNu - 1>(row_values.data(), u);
    return unpack_chebyshev_double_pair_lane(result);
}

template <int ChebNu, int ChebNv>
inline std::pair<double, double> chebyshev_sum_2d_pair_fixed_double_dot_simd(const double* coeffs_up_ptr,
                                                                             const double* coeffs_down_ptr,
                                                                             double u,
                                                                             double v) {
    std::array<double, ChebNu> basis_u{};
    std::array<double, ChebNv> basis_v{};
    fill_chebyshev_basis(u, basis_u);
    fill_chebyshev_basis(v, basis_v);

    double up_total = 0.0;
    double down_total = 0.0;
    for (int u_index = 0; u_index < ChebNu; ++u_index) {
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(ChebNv);
        const double u_weight = basis_u[static_cast<size_t>(u_index)];
        up_total += u_weight *
#if CDET_USE_CHEB_SIMD == 8
                    dot_chebyshev_coefficients_lane8<ChebNv>(
                        coeffs_up_ptr + row_offset, basis_v.data());
#elif CDET_USE_CHEB_SIMD == 4
                    dot_chebyshev_coefficients_lane4<ChebNv>(
                        coeffs_up_ptr + row_offset, basis_v.data());
#else
                    dot_chebyshev_coefficients_lane2<ChebNv>(
                        coeffs_up_ptr + row_offset, basis_v.data());
#endif
        down_total += u_weight *
#if CDET_USE_CHEB_SIMD == 8
                      dot_chebyshev_coefficients_lane8<ChebNv>(
                          coeffs_down_ptr + row_offset, basis_v.data());
#elif CDET_USE_CHEB_SIMD == 4
                      dot_chebyshev_coefficients_lane4<ChebNv>(
                          coeffs_down_ptr + row_offset, basis_v.data());
#else
                      dot_chebyshev_coefficients_lane2<ChebNv>(
                          coeffs_down_ptr + row_offset, basis_v.data());
#endif
    }
    return {up_total, down_total};
}
#endif

template <typename RealT>
struct GreenLookupDescriptor {
    bool use_upper_branch = false;
    bool use_corner = false;
    size_t corner_index = 0;
    size_t coeff_base_index = 0;
    RealT u = RealT(0);
    RealT v = RealT(0);
};

using GreenLookupDescriptorDD = GreenLookupDescriptor<DoubleDouble>;

template <typename RealT>
inline RealT clamp_to_unit_interval(const RealT& value) {
    if (value < RealT(-1)) {
        return RealT(-1);
    }
    if (value > RealT(1)) {
        return RealT(1);
    }
    return value;
}

inline DoubleDouble clamp_to_unit_interval_dd(const DoubleDouble& value) {
    return clamp_to_unit_interval(value);
}

template <typename RealT>
inline int floor_to_int_for_real(const RealT& value) {
    if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DD_FLOOR_TO_INT(value);
    } else if constexpr (std::is_same_v<RealT, MP_Real>) {
        return static_cast<int>(MP_FLOOR(value));
    } else {
        const long double floored = std::floor(static_cast<long double>(value));
        constexpr auto int_min = static_cast<long double>(std::numeric_limits<int>::min());
        constexpr auto int_max = static_cast<long double>(std::numeric_limits<int>::max());
        if (floored < int_min || floored > int_max) {
            throw std::overflow_error("floor_to_int_for_real: result out of int range");
        }
        return static_cast<int>(floored);
    }
}

template <typename RealT>
inline int sampled_coordinate_to_index(const RealT& coordinate, int extent) {
    int index = floor_to_int_for_real(coordinate);
    if (index < 0) {
        index = 0;
    }
    if (index >= extent) {
        index = extent - 1;
    }
    return index;
}

inline size_t coeff_replica_sitepair_base_index(int replica_alpha,
                                                int replica_beta,
                                                int R1_b1,
                                                int R1_b2,
                                                int R2_b1,
                                                int R2_b2,
                                                int cheb_nu,
                                                int cheb_nv,
                                                int N1,
                                                int N2) {
    return full_coeff_index(
        0, 0, replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2, cheb_nu, cheb_nv, N1, N2);
}

template <typename RealT>
inline RealT clenshaw_from_strided_coeffs(const std::vector<RealT>& coeffs,
                                          int degree,
                                          const RealT& x,
                                          size_t base_index,
                                          size_t stride) {
    if (degree < 0) {
        return RealT(0);
    }
    if (degree == 0) {
        return coeffs[base_index];
    }
    if (degree == 1) {
        return coeffs[base_index] + coeffs[base_index + stride] * x;
    }

    const RealT two_x = RealT(2) * x;
    RealT b_kplus1(0);
    RealT b_kplus2(0);

    for (int k = degree; k >= 1; --k) {
        const RealT b_k = coeffs[base_index + static_cast<size_t>(k) * stride] + two_x * b_kplus1 - b_kplus2;
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    return coeffs[base_index] + b_kplus1 * x - b_kplus2;
}

inline DoubleDouble clenshaw_from_strided_coeffs_dd(const DD1DVector& coeffs,
                                                    int degree,
                                                    const DoubleDouble& x,
                                                    size_t base_index,
                                                    size_t stride) {
    return clenshaw_from_strided_coeffs(coeffs, degree, x, base_index, stride);
}

template <typename RealT>
inline RealT clenshaw_from_contiguous_coeffs(const RealT* coeffs, int degree, const RealT& x) {
    if (degree < 0) {
        return RealT(0);
    }
    if (degree == 0) {
        return coeffs[0];
    }
    if (degree == 1) {
        return coeffs[0] + coeffs[1] * x;
    }

    const RealT two_x = RealT(2) * x;
    RealT b_kplus1(0);
    RealT b_kplus2(0);

    for (int k = degree; k >= 1; --k) {
        const RealT b_k = coeffs[k] + two_x * b_kplus1 - b_kplus2;
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    return coeffs[0] + b_kplus1 * x - b_kplus2;
}

inline DoubleDouble clenshaw_from_contiguous_coeffs_dd(const DoubleDouble* coeffs,
                                                       int degree,
                                                       const DoubleDouble& x) {
    return clenshaw_from_contiguous_coeffs(coeffs, degree, x);
}

template <typename RealT>
inline std::pair<RealT, RealT> clenshaw_pair_from_contiguous_coeffs(const RealT* coeffs_up,
                                                                    const RealT* coeffs_down,
                                                                    int degree,
                                                                    const RealT& x) {
    if (degree < 0) {
        return {RealT(0), RealT(0)};
    }
    if (degree == 0) {
        return {coeffs_up[0], coeffs_down[0]};
    }
    if (degree == 1) {
        return {
            coeffs_up[0] + coeffs_up[1] * x,
            coeffs_down[0] + coeffs_down[1] * x,
        };
    }

    const RealT two_x = RealT(2) * x;
    RealT up_b_kplus1(0);
    RealT up_b_kplus2(0);
    RealT down_b_kplus1(0);
    RealT down_b_kplus2(0);

    for (int k = degree; k >= 1; --k) {
        const RealT up_b_k = coeffs_up[k] + two_x * up_b_kplus1 - up_b_kplus2;
        const RealT down_b_k = coeffs_down[k] + two_x * down_b_kplus1 - down_b_kplus2;
        up_b_kplus2 = up_b_kplus1;
        up_b_kplus1 = up_b_k;
        down_b_kplus2 = down_b_kplus1;
        down_b_kplus1 = down_b_k;
    }

    return {
        coeffs_up[0] + up_b_kplus1 * x - up_b_kplus2,
        coeffs_down[0] + down_b_kplus1 * x - down_b_kplus2,
    };
}

inline std::pair<DoubleDouble, DoubleDouble> clenshaw_pair_from_contiguous_coeffs_dd(
    const DoubleDouble* coeffs_up,
    const DoubleDouble* coeffs_down,
    int degree,
    const DoubleDouble& x) {
    return clenshaw_pair_from_contiguous_coeffs(coeffs_up, coeffs_down, degree, x);
}

template <int Degree, typename RealT>
inline RealT clenshaw_fixed_degree_contiguous(const RealT* coeffs, const RealT& x) {
    if constexpr (Degree < 0) {
        return RealT(0);
    } else if constexpr (Degree == 0) {
        return coeffs[0];
    } else if constexpr (Degree == 1) {
        return coeffs[0] + coeffs[1] * x;
    } else {
        const RealT two_x = RealT(2) * x;
        RealT b_kplus1(0);
        RealT b_kplus2(0);

        for (int k = Degree; k >= 1; --k) {
            const RealT b_k = coeffs[k] + two_x * b_kplus1 - b_kplus2;
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }

        return coeffs[0] + b_kplus1 * x - b_kplus2;
    }
}

template <int Degree>
inline DoubleDouble clenshaw_fixed_degree_contiguous_dd(const DoubleDouble* coeffs,
                                                        const DoubleDouble& x) {
    return clenshaw_fixed_degree_contiguous<Degree>(coeffs, x);
}

template <int Degree, typename RealT>
inline std::pair<RealT, RealT> clenshaw_fixed_degree_pair_contiguous(const RealT* coeffs_up,
                                                                     const RealT* coeffs_down,
                                                                     const RealT& x) {
    if constexpr (Degree < 0) {
        return {RealT(0), RealT(0)};
    } else if constexpr (Degree == 0) {
        return {coeffs_up[0], coeffs_down[0]};
    } else if constexpr (Degree == 1) {
        return {
            coeffs_up[0] + coeffs_up[1] * x,
            coeffs_down[0] + coeffs_down[1] * x,
        };
    } else {
        const RealT two_x = RealT(2) * x;
        RealT up_b_kplus1(0);
        RealT up_b_kplus2(0);
        RealT down_b_kplus1(0);
        RealT down_b_kplus2(0);

        for (int k = Degree; k >= 1; --k) {
            const RealT up_b_k = coeffs_up[k] + two_x * up_b_kplus1 - up_b_kplus2;
            const RealT down_b_k = coeffs_down[k] + two_x * down_b_kplus1 - down_b_kplus2;
            up_b_kplus2 = up_b_kplus1;
            up_b_kplus1 = up_b_k;
            down_b_kplus2 = down_b_kplus1;
            down_b_kplus1 = down_b_k;
        }

        return {
            coeffs_up[0] + up_b_kplus1 * x - up_b_kplus2,
            coeffs_down[0] + down_b_kplus1 * x - down_b_kplus2,
        };
    }
}

template <int Degree>
inline std::pair<DoubleDouble, DoubleDouble> clenshaw_fixed_degree_pair_contiguous_dd(
    const DoubleDouble* coeffs_up,
    const DoubleDouble* coeffs_down,
    const DoubleDouble& x) {
    return clenshaw_fixed_degree_pair_contiguous<Degree>(coeffs_up, coeffs_down, x);
}

inline size_t coeff_replica_sitepair_offset(int replica_alpha,
                                            int replica_beta,
                                            int R1_b1,
                                            int R1_b2,
                                            int R2_b1,
                                            int R2_b2,
                                            int cheb_nu,
                                            int cheb_nv,
                                            int N1,
                                            int N2) {
    const size_t coeff_count_per_replica_sitepair = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    return replica_sitepair_index(replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2, N1, N2) *
           coeff_count_per_replica_sitepair;
}

template <int ChebNu, int ChebNv, typename RealT>
inline std::pair<RealT, RealT> chebyshev_sum_2d_pair_fixed(const RealT* coeffs_up_ptr,
                                                           const RealT* coeffs_down_ptr,
                                                           const RealT& u,
                                                           const RealT& v) {
    std::array<RealT, ChebNu> row_values_up{};
    std::array<RealT, ChebNu> row_values_down{};

    for (int u_index = 0; u_index < ChebNu; ++u_index) {
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(ChebNv);
        const auto row_pair = clenshaw_fixed_degree_pair_contiguous<ChebNv - 1>(
            coeffs_up_ptr + row_offset, coeffs_down_ptr + row_offset, v);
        row_values_up[static_cast<size_t>(u_index)] = row_pair.first;
        row_values_down[static_cast<size_t>(u_index)] = row_pair.second;
    }

    return clenshaw_fixed_degree_pair_contiguous<ChebNu - 1>(row_values_up.data(), row_values_down.data(), u);
}

template <int ChebNu, int ChebNv>
inline std::pair<DoubleDouble, DoubleDouble> chebyshev_sum_2d_pair_fixed_dd(
    const DoubleDouble* coeffs_up_ptr,
    const DoubleDouble* coeffs_down_ptr,
    const DoubleDouble& u,
    const DoubleDouble& v) {
    return chebyshev_sum_2d_pair_fixed<ChebNu, ChebNv>(coeffs_up_ptr, coeffs_down_ptr, u, v);
}

template <int ChebNu, int ChebNv, typename RealT>
inline RealT chebyshev_sum_2d_fixed(const RealT* coeffs_ptr, const RealT& u, const RealT& v) {
    std::array<RealT, ChebNu> row_values{};

    for (int u_index = 0; u_index < ChebNu; ++u_index) {
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(ChebNv);
        row_values[static_cast<size_t>(u_index)] =
            clenshaw_fixed_degree_contiguous<ChebNv - 1>(coeffs_ptr + row_offset, v);
    }

    return clenshaw_fixed_degree_contiguous<ChebNu - 1>(row_values.data(), u);
}

template <int ChebNu, int ChebNv>
inline DoubleDouble chebyshev_sum_2d_fixed_dd(const DoubleDouble* coeffs_ptr,
                                              const DoubleDouble& u,
                                              const DoubleDouble& v) {
    return chebyshev_sum_2d_fixed<ChebNu, ChebNv>(coeffs_ptr, u, v);
}

template <typename RealT>
inline GreenLookupDescriptor<RealT> prepare_green_lookup_descriptor(int replica_alpha,
                                                                    int replica_beta,
                                                                    const RealT& tau_1,
                                                                    const RealT& tau_2,
                                                                    int x1,
                                                                    int y1,
                                                                    int x2,
                                                                    int y2,
                                                                    const RealT& beta_s,
                                                                    const RealT& branch_eps,
                                                                    int cheb_nu,
                                                                    int cheb_nv,
                                                                    int N1,
                                                                    int N2,
                                                                    bool use_upper_branch) {
    GreenLookupDescriptor<RealT> descriptor;
    descriptor.use_upper_branch = use_upper_branch;

    if (use_upper_branch) {
        if (tau_1 < branch_eps) {
            descriptor.use_corner = true;
            descriptor.corner_index =
                corner_replica_sitepair_index(replica_alpha, replica_beta, x1, y1, x2, y2, N1, N2);
            return descriptor;
        }

        descriptor.u = clamp_to_unit_interval(RealT(2) * tau_1 / beta_s - RealT(1));
        descriptor.v = clamp_to_unit_interval(RealT(2) * tau_2 / tau_1 - RealT(1));
    } else {
        if (tau_2 < branch_eps) {
            descriptor.use_corner = true;
            descriptor.corner_index =
                corner_replica_sitepair_index(replica_alpha, replica_beta, x1, y1, x2, y2, N1, N2);
            return descriptor;
        }

        descriptor.u = clamp_to_unit_interval(RealT(2) * tau_2 / beta_s - RealT(1));
        descriptor.v = clamp_to_unit_interval(RealT(2) * tau_1 / tau_2 - RealT(1));
    }

    descriptor.coeff_base_index =
        coeff_replica_sitepair_offset(replica_alpha, replica_beta, x1, y1, x2, y2, cheb_nu, cheb_nv, N1, N2);
    return descriptor;
}

inline GreenLookupDescriptorDD prepare_green_lookup_descriptor_dd(int replica_alpha,
                                                                 int replica_beta,
                                                                 const DoubleDouble& tau_1,
                                                                 const DoubleDouble& tau_2,
                                                                 int x1,
                                                                 int y1,
                                                                 int x2,
                                                                 int y2,
                                                                 const DoubleDouble& beta_s,
                                                                 const DoubleDouble& branch_eps,
                                                                 int cheb_nu,
                                                                 int cheb_nv,
                                                                 int N1,
                                                                 int N2,
                                                                 bool use_upper_branch) {
    return prepare_green_lookup_descriptor(replica_alpha,
                                           replica_beta,
                                           tau_1,
                                           tau_2,
                                           x1,
                                           y1,
                                           x2,
                                           y2,
                                           beta_s,
                                           branch_eps,
                                           cheb_nu,
                                           cheb_nv,
                                           N1,
                                           N2,
                                           use_upper_branch);
}

template <typename RealT>
inline std::pair<RealT, RealT> chebyshev_sum_2d_pair_from_ptrs(const RealT* coeffs_up_ptr,
                                                               const RealT* coeffs_down_ptr,
                                                               const RealT& u,
                                                               const RealT& v,
                                                               int cheb_nu,
                                                               int cheb_nv) {
#if CDET_USE_CHEB_SIMD > 0 && (defined(__GNUC__) || defined(__clang__))
    if constexpr (std::is_same_v<RealT, double>) {
        if (cheb_nu == 25 && cheb_nv == 25) {
#if CDET_USE_CHEB_SIMD == 2 || CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
            return chebyshev_sum_2d_pair_fixed_double_dot_simd<25, 25>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#else
            return chebyshev_sum_2d_pair_fixed_double_simd<25, 25>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#endif
        }
        if (cheb_nu == 30 && cheb_nv == 30) {
#if CDET_USE_CHEB_SIMD == 2 || CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
            return chebyshev_sum_2d_pair_fixed_double_dot_simd<30, 30>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#else
            return chebyshev_sum_2d_pair_fixed_double_simd<30, 30>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#endif
        }
        if (cheb_nu == 32 && cheb_nv == 32) {
#if CDET_USE_CHEB_SIMD == 2 || CDET_USE_CHEB_SIMD == 4 || CDET_USE_CHEB_SIMD == 8
            return chebyshev_sum_2d_pair_fixed_double_dot_simd<32, 32>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#else
            return chebyshev_sum_2d_pair_fixed_double_simd<32, 32>(
                coeffs_up_ptr, coeffs_down_ptr, u, v);
#endif
        }
    }
#endif

    if (cheb_nu == 25 && cheb_nv == 25) {
        return chebyshev_sum_2d_pair_fixed<25, 25>(coeffs_up_ptr, coeffs_down_ptr, u, v);
    }
    if (cheb_nu == 30 && cheb_nv == 30) {
        return chebyshev_sum_2d_pair_fixed<30, 30>(coeffs_up_ptr, coeffs_down_ptr, u, v);
    }
    if (cheb_nu == 32 && cheb_nv == 32) {
        return chebyshev_sum_2d_pair_fixed<32, 32>(coeffs_up_ptr, coeffs_down_ptr, u, v);
    }

    if (cheb_nu <= CHEBYSHEV_FAST_BUFFER_LIMIT && cheb_nv <= CHEBYSHEV_FAST_BUFFER_LIMIT) {
        std::array<RealT, CHEBYSHEV_FAST_BUFFER_LIMIT> row_values_up{};
        std::array<RealT, CHEBYSHEV_FAST_BUFFER_LIMIT> row_values_down{};

        for (int u_index = 0; u_index < cheb_nu; ++u_index) {
            const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
            const auto row_pair = clenshaw_pair_from_contiguous_coeffs(
                coeffs_up_ptr + row_offset, coeffs_down_ptr + row_offset, cheb_nv - 1, v);
            row_values_up[static_cast<size_t>(u_index)] = row_pair.first;
            row_values_down[static_cast<size_t>(u_index)] = row_pair.second;
        }

        return clenshaw_pair_from_contiguous_coeffs(row_values_up.data(), row_values_down.data(), cheb_nu - 1, u);
    }

    std::vector<RealT> row_values_up(static_cast<size_t>(cheb_nu), RealT(0));
    std::vector<RealT> row_values_down(static_cast<size_t>(cheb_nu), RealT(0));

    for (int u_index = 0; u_index < cheb_nu; ++u_index) {
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
        const auto row_pair = clenshaw_pair_from_contiguous_coeffs(
            coeffs_up_ptr + row_offset, coeffs_down_ptr + row_offset, cheb_nv - 1, v);
        row_values_up[static_cast<size_t>(u_index)] = row_pair.first;
        row_values_down[static_cast<size_t>(u_index)] = row_pair.second;
    }

    return clenshaw_pair_from_contiguous_coeffs(row_values_up.data(), row_values_down.data(), cheb_nu - 1, u);
}

template <typename RealT>
inline std::pair<RealT, RealT> chebyshev_sum_2d_pair_from_base(const std::vector<RealT>& coeffs_up,
                                                               const std::vector<RealT>& coeffs_down,
                                                               size_t coeff_base_index,
                                                               const RealT& u,
                                                               const RealT& v,
                                                               int cheb_nu,
                                                               int cheb_nv) {
    return chebyshev_sum_2d_pair_from_ptrs(
        coeffs_up.data() + coeff_base_index,
        coeffs_down.data() + coeff_base_index,
        u,
        v,
        cheb_nu,
        cheb_nv);
}

inline std::pair<DoubleDouble, DoubleDouble> chebyshev_sum_2d_pair_from_base_dd(
    const DD1DVector& coeffs_up,
    const DD1DVector& coeffs_down,
    size_t coeff_base_index,
    const DoubleDouble& u,
    const DoubleDouble& v,
    int cheb_nu,
    int cheb_nv) {
    return chebyshev_sum_2d_pair_from_base(coeffs_up, coeffs_down, coeff_base_index, u, v, cheb_nu, cheb_nv);
}

template <typename RealT>
inline std::pair<RealT, RealT> evaluate_green_pair_descriptor(const GreenLookupDescriptor<RealT>& descriptor,
                                                              const PiecewiseChebyshevTable<RealT>& table_up,
                                                              const PiecewiseChebyshevTable<RealT>& table_down,
                                                              int cheb_nu,
                                                              int cheb_nv) {
    if (descriptor.use_corner) {
        if (descriptor.use_upper_branch) {
            return {
                table_up.upper_corners[descriptor.corner_index],
                table_down.upper_corners[descriptor.corner_index],
            };
        }
        return {
            table_up.lower_corners[descriptor.corner_index],
            table_down.lower_corners[descriptor.corner_index],
        };
    }

    if (descriptor.use_upper_branch) {
        return chebyshev_sum_2d_pair_from_base(
            table_up.upper_coeffs, table_down.upper_coeffs, descriptor.coeff_base_index, descriptor.u, descriptor.v, cheb_nu, cheb_nv);
    }
    return chebyshev_sum_2d_pair_from_base(
        table_up.lower_coeffs, table_down.lower_coeffs, descriptor.coeff_base_index, descriptor.u, descriptor.v, cheb_nu, cheb_nv);
}

template <typename RealT>
inline std::pair<RealT, RealT> evaluate_green_pair_descriptor(
    const GreenLookupDescriptor<RealT>& descriptor,
    const DiskBackedPiecewiseChebyshevTable<RealT>& table_up,
    const DiskBackedPiecewiseChebyshevTable<RealT>& table_down,
    int cheb_nu,
    int cheb_nv) {
    if (descriptor.use_corner) {
        if (descriptor.use_upper_branch) {
            return {
                table_up.upper_corners[descriptor.corner_index],
                table_down.upper_corners[descriptor.corner_index],
            };
        }
        return {
            table_up.lower_corners[descriptor.corner_index],
            table_down.lower_corners[descriptor.corner_index],
        };
    }

    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t sitepair_index = descriptor.coeff_base_index / node_count;
    const TriangleBranch branch =
        descriptor.use_upper_branch ? TriangleBranch::Upper : TriangleBranch::Lower;
    if (!table_up.coeff_cache || !table_down.coeff_cache) {
        throw std::runtime_error("Disk-backed Green table cache is not initialized");
    }
    const DiskBackedCoeffChunk& up_chunk = table_up.find_chunk(branch, sitepair_index);
    const DiskBackedCoeffChunk& down_chunk = table_down.find_chunk(branch, sitepair_index);
    const auto up_block = table_up.coeff_cache->get_block(
        table_up.table_kind, branch, up_chunk, node_count);
    const auto down_block = table_down.coeff_cache->get_block(
        table_down.table_kind, branch, down_chunk, node_count);
    const size_t up_local_base = (sitepair_index - up_chunk.sitepair_begin) * node_count;
    const size_t down_local_base = (sitepair_index - down_chunk.sitepair_begin) * node_count;
    return chebyshev_sum_2d_pair_from_ptrs(up_block->data() + up_local_base,
                                           down_block->data() + down_local_base,
                                           descriptor.u,
                                           descriptor.v,
                                           cheb_nu,
                                           cheb_nv);
}

inline std::pair<DoubleDouble, DoubleDouble> evaluate_green_pair_descriptor_dd(
    const GreenLookupDescriptorDD& descriptor,
    const GreenFunctionTableDD& table_up,
    const GreenFunctionTableDD& table_down,
    int cheb_nu,
    int cheb_nv) {
    return evaluate_green_pair_descriptor(descriptor, table_up, table_down, cheb_nu, cheb_nv);
}

template <typename RealT>
inline RealT chebyshev_sum_2d(const std::vector<RealT>& coeffs,
                              int replica_alpha,
                              int replica_beta,
                              int R1_b1,
                              int R1_b2,
                              int R2_b1,
                              int R2_b2,
                              const RealT& u,
                              const RealT& v,
                              int cheb_nu,
                              int cheb_nv,
                              int N1,
                              int N2) {
    const size_t replica_sitepair_base =
        coeff_replica_sitepair_offset(replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2, cheb_nu, cheb_nv, N1, N2);
    const RealT* coeffs_ptr = coeffs.data() + replica_sitepair_base;

    RealT result(0);
    if (cheb_nu == 25 && cheb_nv == 25) {
        result = chebyshev_sum_2d_fixed<25, 25>(coeffs_ptr, u, v);
    } else if (cheb_nu == 30 && cheb_nv == 30) {
        result = chebyshev_sum_2d_fixed<30, 30>(coeffs_ptr, u, v);
    } else if (cheb_nu == 32 && cheb_nv == 32) {
        result = chebyshev_sum_2d_fixed<32, 32>(coeffs_ptr, u, v);
    } else if (cheb_nu <= CHEBYSHEV_FAST_BUFFER_LIMIT && cheb_nv <= CHEBYSHEV_FAST_BUFFER_LIMIT) {
        std::array<RealT, CHEBYSHEV_FAST_BUFFER_LIMIT> row_values{};
        for (int u_index = 0; u_index < cheb_nu; ++u_index) {
            const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
            row_values[static_cast<size_t>(u_index)] =
                clenshaw_from_contiguous_coeffs(coeffs_ptr + row_offset, cheb_nv - 1, v);
        }
        result = clenshaw_from_contiguous_coeffs(row_values.data(), cheb_nu - 1, u);
    } else {
        const size_t stride_v = 1;
        const size_t stride_u = static_cast<size_t>(cheb_nv);
        std::vector<RealT> row_values(static_cast<size_t>(cheb_nu), RealT(0));
        for (int u_index = 0; u_index < cheb_nu; ++u_index) {
            const size_t base_index = static_cast<size_t>(u_index) * stride_u + replica_sitepair_base;
            row_values[static_cast<size_t>(u_index)] =
                clenshaw_from_strided_coeffs(coeffs, cheb_nv - 1, v, base_index, stride_v);
        }
        result = clenshaw_from_strided_coeffs(row_values, cheb_nu - 1, u, 0, 1);
    }
    return result;
}

inline DoubleDouble chebyshev_sum_2d_dd(const DD1DVector& coeffs,
                                        int replica_alpha,
                                        int replica_beta,
                                        int R1_b1,
                                        int R1_b2,
                                        int R2_b1,
                                        int R2_b2,
                                        const DoubleDouble& u,
                                        const DoubleDouble& v,
                                        int cheb_nu,
                                        int cheb_nv,
                                        int N1,
                                        int N2) {
    return chebyshev_sum_2d(
        coeffs, replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2, u, v, cheb_nu, cheb_nv, N1, N2);
}

template <typename RealT>
inline RealT evaluate_green_descriptor(const GreenLookupDescriptor<RealT>& descriptor,
                                       const PiecewiseChebyshevTable<RealT>& table,
                                       int replica_alpha,
                                       int replica_beta,
                                       int x1,
                                       int y1,
                                       int x2,
                                       int y2,
                                       int cheb_nu,
                                       int cheb_nv,
                                       int N1,
                                       int N2) {
    if (descriptor.use_corner) {
        if (descriptor.use_upper_branch) {
            return table.upper_corners[descriptor.corner_index];
        }
        return table.lower_corners[descriptor.corner_index];
    }

    if (descriptor.use_upper_branch) {
        return chebyshev_sum_2d(
            table.upper_coeffs, replica_alpha, replica_beta, x1, y1, x2, y2, descriptor.u, descriptor.v, cheb_nu, cheb_nv, N1, N2);
    }
    return chebyshev_sum_2d(
        table.lower_coeffs, replica_alpha, replica_beta, x1, y1, x2, y2, descriptor.u, descriptor.v, cheb_nu, cheb_nv, N1, N2);
}

template <typename RealT>
inline RealT g_func_flattened(int replica_alpha,
                              int replica_beta,
                              const RealT& tau_1,
                              const RealT& tau_2,
                              const RealT& R1_b1,
                              const RealT& R1_b2,
                              const RealT& R2_b1,
                              const RealT& R2_b2,
                              const PiecewiseChebyshevTable<RealT>& table,
                              const RealT& beta_s,
                              int N1,
                              int N2,
                              const RealT& branch_eps,
                              int cheb_nu,
                              int cheb_nv,
                              bool use_upper_branch) {
    const int x1 = sampled_coordinate_to_index(R1_b1, N1);
    const int y1 = sampled_coordinate_to_index(R1_b2, N2);
    const int x2 = sampled_coordinate_to_index(R2_b1, N1);
    const int y2 = sampled_coordinate_to_index(R2_b2, N2);

    const GreenLookupDescriptor<RealT> descriptor =
        prepare_green_lookup_descriptor(replica_alpha,
                                        replica_beta,
                                        tau_1,
                                        tau_2,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        beta_s,
                                        branch_eps,
                                        cheb_nu,
                                        cheb_nv,
                                        N1,
                                        N2,
                                        use_upper_branch);
    const RealT result = evaluate_green_descriptor(
        descriptor, table, replica_alpha, replica_beta, x1, y1, x2, y2, cheb_nu, cheb_nv, N1, N2);
    return result;
}

inline DoubleDouble g_func_flattened_dd(int replica_alpha,
                                        int replica_beta,
                                        const DoubleDouble& tau_1,
                                        const DoubleDouble& tau_2,
                                        const DoubleDouble& R1_b1,
                                        const DoubleDouble& R1_b2,
                                        const DoubleDouble& R2_b1,
                                        const DoubleDouble& R2_b2,
                                        const GreenFunctionTableDD& table,
                                        const DoubleDouble& beta_s,
                                        int N1,
                                        int N2,
                                        const DoubleDouble& branch_eps,
                                        int cheb_nu,
                                        int cheb_nv,
                                        bool use_upper_branch) {
    return g_func_flattened(replica_alpha,
                            replica_beta,
                            tau_1,
                            tau_2,
                            R1_b1,
                            R1_b2,
                            R2_b1,
                            R2_b2,
                            table,
                            beta_s,
                            N1,
                            N2,
                            branch_eps,
                            cheb_nu,
                            cheb_nv,
                            use_upper_branch);
}

inline DoubleDouble factorial_dd(int n) {
    if (n < 0) {
        throw std::runtime_error("Factorial is not defined for negative numbers");
    }
    DoubleDouble result(1);
    for (int i = 1; i <= n; ++i) {
        result *= DoubleDouble(i);
    }
    return result;
}

#endif
