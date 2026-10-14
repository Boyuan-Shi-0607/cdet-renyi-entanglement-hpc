#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_INTEGRAND_REPLICA_RANDOMLY_SAMPLED_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_INTEGRAND_REPLICA_RANDOMLY_SAMPLED_H

#include "cdet/utility/my_funcs.h"
#include "cdet/fast_principle_minors.h"
#include "cdet/g_funcs.h"
#include "cdet/integrand_parallel.h"
#include "cdet/precision_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename RealT>
struct SampledVertex {
    RealT tau = RealT(0);
    RealT tau_unit = RealT(0);
    int x = 0;
    int y = 0;
    int replica = 0;
};

template <typename RealT>
inline bool sampled_vertices_equal(const SampledVertex<RealT>& left,
                                   const SampledVertex<RealT>& right) {
    return left.tau == right.tau &&
           left.tau_unit == right.tau_unit &&
           left.x == right.x &&
           left.y == right.y &&
           left.replica == right.replica;
}

template <typename RealT>
struct IntegrandRuntimeScalars {
    RealT beta = RealT(0);
    RealT u = RealT(0);
    RealT branch_eps = RealT(0);
    RealT scale = RealT(1);
    RealT diagonal_up_shift = RealT(0);
    RealT diagonal_down_shift = RealT(0);
    RealT tol = RealT(0);
    RealT fpm_global_shift = RealT(-1);
};

constexpr bool ENABLE_HOT_PATH_SANITY_CHECKS = false;
constexpr int MIN_PAIR_TASKS_PER_THREAD = 2;
constexpr std::size_t GREEN_PAIR_CACHE_ENTRY_LIMIT = 8;

template <typename VectorRealT, typename IntegrandRealT>
struct GreenPairMatrixCacheEntry {
    const void* up_table_key = nullptr;
    const void* down_table_key = nullptr;
    int order_num = 0;
    std::vector<SampledVertex<VectorRealT>> vertices;
    std::vector<VectorRealT> up_matrix;
    std::vector<VectorRealT> down_matrix;
    std::vector<IntegrandRealT> diagonal_up;
    std::vector<IntegrandRealT> diagonal_down;
};

template <typename VectorRealT, typename IntegrandRealT>
inline GreenPairMatrixCacheEntry<VectorRealT, IntegrandRealT>& green_pair_matrix_cache_entry_for(
    const void* up_table_key,
    const void* down_table_key) {
    thread_local std::vector<GreenPairMatrixCacheEntry<VectorRealT, IntegrandRealT>> entries;
    for (auto& entry : entries) {
        if (entry.up_table_key == up_table_key && entry.down_table_key == down_table_key) {
            return entry;
        }
    }
    if (entries.size() >= GREEN_PAIR_CACHE_ENTRY_LIMIT) {
        entries.erase(entries.begin());
    }
    entries.push_back({});
    entries.back().up_table_key = up_table_key;
    entries.back().down_table_key = down_table_key;
    return entries.back();
}

template <typename T>
inline T integrand_real_from_dd(const DoubleDouble& value) {
    return real_precision_cast<T>(value);
}

template <typename T>
inline DoubleDouble double_double_from_integrand_real(const T& value) {
    return real_precision_cast<DoubleDouble>(value);
}

template <typename RealT>
inline IntegrandRuntimeScalars<RealT> make_integrand_runtime_scalars(const DoubleDouble& beta_s,
                                                                     const DoubleDouble& U_s,
                                                                     int N1,
                                                                     int N2,
                                                                     const DoubleDouble& branch_eps,
                                                                     const DoubleDouble& alpha_shift_up,
                                                                     const DoubleDouble& alpha_shift_down,
                                                                     const DoubleDouble& tol,
                                                                     const DoubleDouble& fpm_global_shift) {
    IntegrandRuntimeScalars<RealT> scalars;
    scalars.beta = integrand_real_from_dd<RealT>(beta_s);
    scalars.u = integrand_real_from_dd<RealT>(U_s);
    scalars.branch_eps = integrand_real_from_dd<RealT>(branch_eps);
    scalars.scale =
        integrand_real_from_dd<RealT>(DD_SQRT(U_s * DoubleDouble(N1) * DoubleDouble(N2) * beta_s));
    scalars.diagonal_up_shift = integrand_real_from_dd<RealT>(alpha_shift_down / U_s);
    scalars.diagonal_down_shift = integrand_real_from_dd<RealT>(alpha_shift_up / U_s);
    scalars.tol = integrand_real_from_dd<RealT>(tol);
    scalars.fpm_global_shift = integrand_real_from_dd<RealT>(fpm_global_shift);
    return scalars;
}

template <typename OutputRealT, typename IntegrandRealT>
inline OutputRealT cdet_output_from_integrand_real(const IntegrandRealT& value) {
    if constexpr (std::is_same_v<OutputRealT, DoubleDouble>) {
        return double_double_from_integrand_real(value);
    } else {
        return real_precision_cast<OutputRealT>(value);
    }
}

inline int sampled_replica_from_unit_coordinate(const DoubleDouble& coordinate, int replica_count) {
    if (replica_count <= 1) {
        return 0;
    }
    if (replica_count > 2) {
        throw std::runtime_error("Only replica_count = 1 or 2 is supported by the resident Green tables");
    }
    const int sampled_replica = sampled_coordinate_to_index(
        DoubleDouble(replica_count) * coordinate, replica_count);
    return std::clamp(sampled_replica, 0, replica_count - 1);
}

template <typename RealT>
inline int sampled_replica_from_unit_coordinate_typed(const RealT& coordinate, int replica_count) {
    if (replica_count <= 1) {
        return 0;
    }
    if (replica_count > 2) {
        throw std::runtime_error("Only replica_count = 1 or 2 is supported by the resident Green tables");
    }
    const int sampled_replica = sampled_coordinate_to_index(
        RealT(replica_count) * coordinate, replica_count);
    return std::clamp(sampled_replica, 0, replica_count - 1);
}

template <typename OutputRealT>
inline OutputRealT cdet_order_prefactor(int order_num, int N1, int N2, int replica_count = 2) {
    if constexpr (std::is_same_v<OutputRealT, DoubleDouble>) {
        return DD_POW(DoubleDouble(-replica_count), DoubleDouble(order_num)) /
               (factorial_dd(order_num) * DoubleDouble(N1) * DoubleDouble(N2));
    } else {
        long double numerator = 1.0L;
        for (int i = 0; i < order_num; ++i) {
            numerator *= -static_cast<long double>(replica_count);
        }
        long double denominator = static_cast<long double>(N1) * static_cast<long double>(N2);
        for (int i = 2; i <= order_num; ++i) {
            denominator *= static_cast<long double>(i);
        }
        return static_cast<OutputRealT>(numerator / denominator);
    }
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline IntegrandRealT order1_density_integrand_from_sampled_vertex(
    const SampledVertex<VectorRealT>& vertex,
    const UpTableT& g_table_1,
    const DownTableT& g_table_2,
    int N1,
    int N2,
    int cheb_nu,
    int cheb_nv,
    const IntegrandRuntimeScalars<VectorRealT>& vector_scalars,
    const IntegrandRuntimeScalars<IntegrandRealT>& integrand_scalars,
    int replica_count = 2) {
    const GreenLookupDescriptor<VectorRealT> descriptor =
        prepare_green_lookup_descriptor(vertex.replica,
                                        vertex.replica,
                                        vertex.tau,
                                        vertex.tau,
                                        vertex.x,
                                        vertex.y,
                                        vertex.x,
                                        vertex.y,
                                        vector_scalars.beta,
                                        vector_scalars.branch_eps,
                                        cheb_nu,
                                        cheb_nv,
                                        N1,
                                        N2,
                                        false);

    auto [up_value, down_value] =
        evaluate_green_pair_descriptor(descriptor, g_table_1, g_table_2, cheb_nu, cheb_nv);
    const IntegrandRealT shifted_up =
        real_precision_cast<IntegrandRealT>(up_value) + integrand_scalars.diagonal_up_shift;
    const IntegrandRealT shifted_down =
        real_precision_cast<IntegrandRealT>(down_value) + integrand_scalars.diagonal_down_shift;
    const IntegrandRealT counterterm =
        integrand_scalars.diagonal_up_shift * integrand_scalars.diagonal_down_shift;

    return IntegrandRealT(-replica_count) * integrand_scalars.u * integrand_scalars.beta *
           (shifted_up * shifted_down - counterterm);
}

template <typename T>
inline void check_vector_nan(const std::vector<T>& vec, const std::string& vector_name) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (is_nan(vec[i])) {
            std::stringstream ss;
            ss << "NaN detected in " << vector_name << "[" << i << "]";
            throw std::runtime_error(ss.str());
        }
    }
}

template <typename T>
T compute_dp_ztransform(const std::vector<T>& up,
                        const std::vector<T>& down,
                        int order_num,
                        const T& roundoff_epsilon0 = T(0),
                        const T& roundoff_safety_factor = T(0)) {
    const size_t subset_count = size_t{1} << order_num;
    thread_local std::vector<T> a_workspace;
    thread_local std::vector<T> dp_workspace;
    thread_local std::vector<std::vector<size_t>> by_popcount_workspace;
    thread_local int cached_order_num = -1;

    a_workspace.resize(subset_count);
    dp_workspace.resize(subset_count);

    if (cached_order_num != order_num) {
        by_popcount_workspace.assign(static_cast<size_t>(order_num + 1), {});
        for (size_t mask = 0; mask < subset_count; ++mask) {
            by_popcount_workspace[static_cast<size_t>(popcnt(static_cast<uint32_t>(mask)))].push_back(mask);
        }
        cached_order_num = order_num;
    }

    T* a = a_workspace.data();
    T* dp = dp_workspace.data();
    const T roundoff_threshold = roundoff_epsilon0 * roundoff_safety_factor;
    const bool enable_roundoff_filter = roundoff_threshold > T(0);

    a[0] = T(0);
    for (size_t mask = 1; mask < subset_count; ++mask) {
        a[mask] = up[mask - 1] * down[mask - 1];
    }

    for (size_t mask = 0; mask < subset_count; ++mask) {
        dp[mask] = T(0);
    }

    for (int i = 0; i < order_num; ++i) {
        dp[size_t{1} << i] = a[size_t{1} << i];
    }

    for (int bits = 2; bits <= order_num; ++bits) {
        for (size_t mask : by_popcount_workspace[static_cast<size_t>(bits)]) {
            T acc = a[mask];
            T rhs_abs_sum = abs_helper(a[mask]);
            if (mask & 1U) {
                for (size_t sub = (mask - 1) & mask; sub; sub = (sub - 1) & mask) {
                    if ((sub & 1U) == 0U) {
                        continue;
                    }
                    const T disconnected_term = dp[sub] * a[mask ^ sub];
                    acc = acc - disconnected_term;
                    rhs_abs_sum += abs_helper(disconnected_term);
                }
                if (enable_roundoff_filter && abs_helper(acc) < roundoff_threshold * rhs_abs_sum) {
                    acc = T(0);
                }
            }
            dp[mask] = acc;
        }
    }

    const T result = dp[subset_count - 1];
    return result;
}

template <typename T>
std::vector<T> compute_dp_ztransform_all_subsets(const std::vector<T>& up,
                                                 const std::vector<T>& down,
                                                 int order_num,
                                                 const T& roundoff_epsilon0 = T(0),
                                                 const T& roundoff_safety_factor = T(0)) {
    const size_t subset_count = size_t{1} << order_num;
    std::vector<T> a(subset_count, T(0));
    std::vector<T> dp(subset_count, T(0));
    struct PopcountMaskCache {
        int order = 0;
        std::vector<std::vector<size_t>> masks_by_popcount;
    };
    thread_local PopcountMaskCache popcount_cache;
    if (popcount_cache.order != order_num) {
        popcount_cache.order = order_num;
        popcount_cache.masks_by_popcount.assign(static_cast<size_t>(order_num + 1), {});
        for (size_t mask = 0; mask < subset_count; ++mask) {
            popcount_cache.masks_by_popcount[static_cast<size_t>(popcnt(static_cast<uint32_t>(mask)))].push_back(mask);
        }
    }
    const std::vector<std::vector<size_t>>& by_popcount = popcount_cache.masks_by_popcount;

    const T roundoff_threshold = roundoff_epsilon0 * roundoff_safety_factor;
    const bool enable_roundoff_filter = roundoff_threshold > T(0);

    for (size_t mask = 1; mask < subset_count; ++mask) {
        a[mask] = up[mask - 1] * down[mask - 1];
    }

    for (int i = 0; i < order_num; ++i) {
        dp[size_t{1} << i] = a[size_t{1} << i];
    }

    for (int bits = 2; bits <= order_num; ++bits) {
        for (size_t mask : by_popcount[static_cast<size_t>(bits)]) {
            const size_t anchor = mask & (~mask + 1);
            T acc = a[mask];
            T rhs_abs_sum = abs_helper(a[mask]);
            for (size_t sub = (mask - 1) & mask; sub; sub = (sub - 1) & mask) {
                if ((sub & anchor) == 0U) {
                    continue;
                }
                const T disconnected_term = dp[sub] * a[mask ^ sub];
                acc = acc - disconnected_term;
                rhs_abs_sum += abs_helper(disconnected_term);
            }
            if (enable_roundoff_filter && abs_helper(acc) < roundoff_threshold * rhs_abs_sum) {
                acc = T(0);
            }
            dp[mask] = acc;
        }
    }

    return dp;
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline DoubleDouble integrand_temp_typed(const DoubleDouble& beta_s,
                                         const DoubleDouble& U_s,
                                         const DoubleDouble* args,
                                         const UpTableT& g_table_1,
                                         const DownTableT& g_table_2,
                                         int order_num,
                                         int N1,
                                         int N2,
                                         const DoubleDouble& branch_eps,
                                         int cheb_nu,
                                         int cheb_nv,
                                         const DoubleDouble& alpha_shift_up,
                                         const DoubleDouble& alpha_shift_down,
                                         const DoubleDouble& tol,
                                         const DoubleDouble& fpm_global_shift,
                                         const IntegrandRealT& ztransform_epsilon0_t,
                                         const IntegrandRealT& ztransform_safety_factor_t,
                                         int replica_count = 2) {
    const IntegrandRealT tol_t = integrand_real_from_dd<IntegrandRealT>(tol);
    const IntegrandRealT fpm_global_shift_t = integrand_real_from_dd<IntegrandRealT>(fpm_global_shift);
    const VectorRealT beta_t = integrand_real_from_dd<VectorRealT>(beta_s);
    const VectorRealT branch_eps_t = integrand_real_from_dd<VectorRealT>(branch_eps);

    if (order_num <= 1) {
        throw std::runtime_error("order_num must be greater than 1 in the replica-Chebyshev integrand");
    }

    constexpr int vertex_stride = 4;
    constexpr int FAST_ORDER_STORAGE_LIMIT = 16;
    const int order_sq = order_num * order_num;

    if constexpr (ENABLE_HOT_PATH_SANITY_CHECKS) {
        for (int i = 0, arg_count = vertex_stride * order_num; i < arg_count; ++i) {
            if (args[i] < DoubleDouble(0) || args[i] > DoubleDouble(1)) {
                return DoubleDouble(0);
            }
        }
    }

    const VectorRealT scale =
        integrand_real_from_dd<VectorRealT>(DD_SQRT(U_s * DoubleDouble(N1) * DoubleDouble(N2) * beta_s));
    const VectorRealT diagonal_up_shift =
        integrand_real_from_dd<VectorRealT>(alpha_shift_down / U_s);
    const VectorRealT diagonal_down_shift =
        integrand_real_from_dd<VectorRealT>(alpha_shift_up / U_s);
    const int requested_thread_count = get_integrand_internal_thread_count();

    std::array<SampledVertex<VectorRealT>, FAST_ORDER_STORAGE_LIMIT> vertex_fast{};
    std::array<VectorRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec1_fast{};
    std::array<VectorRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec2_fast{};

    thread_local std::vector<SampledVertex<VectorRealT>> vertex_slow;
    thread_local std::vector<VectorRealT> vec1_slow;
    thread_local std::vector<VectorRealT> vec2_slow;

    SampledVertex<VectorRealT>* vertices = nullptr;
    VectorRealT* vec1_data = nullptr;
    VectorRealT* vec2_data = nullptr;

    if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
        vertices = vertex_fast.data();
        vec1_data = vec1_fast.data();
        vec2_data = vec2_fast.data();
    } else {
        vertex_slow.resize(static_cast<size_t>(order_num));
        vec1_slow.resize(static_cast<size_t>(order_sq));
        vec2_slow.resize(static_cast<size_t>(order_sq));
        vertices = vertex_slow.data();
        vec1_data = vec1_slow.data();
        vec2_data = vec2_slow.data();
    }

    for (int i = 0; i < order_num; ++i) {
        const int vertex_offset = vertex_stride * i;
        vertices[i].tau_unit = integrand_real_from_dd<VectorRealT>(args[vertex_offset]);
        vertices[i].tau = beta_t * vertices[i].tau_unit;
        vertices[i].x = sampled_coordinate_to_index(
            VectorRealT(N1) * integrand_real_from_dd<VectorRealT>(args[vertex_offset + 1]), N1);
        vertices[i].y = sampled_coordinate_to_index(
            VectorRealT(N2) * integrand_real_from_dd<VectorRealT>(args[vertex_offset + 2]), N2);
        vertices[i].replica = sampled_replica_from_unit_coordinate_typed(
            integrand_real_from_dd<VectorRealT>(args[vertex_offset + 3]), replica_count);
    }

    const int task_count = order_sq;
    const int max_parallel_threads =
        std::min<int>(requested_thread_count, task_count / MIN_PAIR_TASKS_PER_THREAD);

    auto fill_pair_range = [&](size_t /*worker_index*/, size_t begin, size_t end) {
        for (size_t linear_idx = begin; linear_idx < end; ++linear_idx) {
            const int idx = static_cast<int>(linear_idx);
            const int i = idx / order_num;
            const int j = idx - i * order_num;
            const SampledVertex<VectorRealT>& vertex_i = vertices[i];
            const SampledVertex<VectorRealT>& vertex_j = vertices[j];
            const bool use_upper_branch = (i != j) && (vertex_i.tau_unit > vertex_j.tau_unit);

            const GreenLookupDescriptor<VectorRealT> descriptor =
                prepare_green_lookup_descriptor(vertex_i.replica,
                                                vertex_j.replica,
                                                vertex_i.tau,
                                                vertex_j.tau,
                                                vertex_i.x,
                                                vertex_i.y,
                                                vertex_j.x,
                                                vertex_j.y,
                                                beta_t,
                                                branch_eps_t,
                                                cheb_nu,
                                                cheb_nv,
                                                N1,
                                                N2,
                                                use_upper_branch);

            auto [up_value, down_value] =
                evaluate_green_pair_descriptor(descriptor, g_table_1, g_table_2, cheb_nu, cheb_nv);

            if (i == j) {
                up_value += diagonal_up_shift;
                down_value += diagonal_down_shift;
            }

            vec1_data[idx] = up_value * scale;
            vec2_data[idx] = down_value * scale;
        }
    };

    if (max_parallel_threads >= 2) {
        IntegrandParallelExecutor::instance().parallel_for(
            static_cast<size_t>(max_parallel_threads),
            static_cast<size_t>(task_count),
            fill_pair_range);
    } else {
        fill_pair_range(0, 0, static_cast<size_t>(task_count));
    }

    if constexpr (ENABLE_HOT_PATH_SANITY_CHECKS) {
        std::vector<VectorRealT> vec1(static_cast<size_t>(order_sq));
        std::vector<VectorRealT> vec2(static_cast<size_t>(order_sq));
        std::copy_n(vec1_data, order_sq, vec1.data());
        std::copy_n(vec2_data, order_sq, vec2.data());
        check_vector_nan(vec1, "vec1");
        check_vector_nan(vec2, "vec2");
    }

    std::vector<IntegrandRealT> up_mat_minors;
    std::vector<IntegrandRealT> down_mat_minors;
    if constexpr (std::is_same_v<VectorRealT, IntegrandRealT>) {
        up_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec1_data, order_num, tol_t, fpm_global_shift_t);
        down_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec2_data, order_num, tol_t, fpm_global_shift_t);
    } else {
        std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec1_integrand_fast{};
        std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec2_integrand_fast{};
        thread_local std::vector<IntegrandRealT> vec1_integrand_slow;
        thread_local std::vector<IntegrandRealT> vec2_integrand_slow;
        IntegrandRealT* vec1_integrand_data = nullptr;
        IntegrandRealT* vec2_integrand_data = nullptr;
        if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
            vec1_integrand_data = vec1_integrand_fast.data();
            vec2_integrand_data = vec2_integrand_fast.data();
        } else {
            vec1_integrand_slow.resize(static_cast<size_t>(order_sq));
            vec2_integrand_slow.resize(static_cast<size_t>(order_sq));
            vec1_integrand_data = vec1_integrand_slow.data();
            vec2_integrand_data = vec2_integrand_slow.data();
        }
        for (int idx = 0; idx < order_sq; ++idx) {
            vec1_integrand_data[idx] = real_precision_cast<IntegrandRealT>(vec1_data[idx]);
            vec2_integrand_data[idx] = real_precision_cast<IntegrandRealT>(vec2_data[idx]);
        }
        up_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec1_integrand_data, order_num, tol_t, fpm_global_shift_t);
        down_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec2_integrand_data, order_num, tol_t, fpm_global_shift_t);
    }

    if constexpr (ENABLE_HOT_PATH_SANITY_CHECKS) {
        check_vector_nan(up_mat_minors, "up_mat_minors");
        check_vector_nan(down_mat_minors, "down_mat_minors");
    }

    const IntegrandRealT c_result = compute_dp_ztransform(
        up_mat_minors,
        down_mat_minors,
        order_num,
        ztransform_epsilon0_t,
        ztransform_safety_factor_t);
    const DoubleDouble value =
        cdet_order_prefactor<DoubleDouble>(order_num, N1, N2, replica_count) *
        double_double_from_integrand_real(c_result);

    if (!is_finite(value)) {
        throw std::runtime_error("Error (integrand): Value is not finite");
    }
    return value;
}

template <typename VectorRealT, typename IntegrandRealT, typename OutputRealT, typename UpTableT, typename DownTableT>
inline void integrand_all_subset_values_typed_into(
    const DoubleDouble& beta_s,
    const DoubleDouble& U_s,
    const DoubleDouble* args,
    const UpTableT& g_table_1,
    const DownTableT& g_table_2,
    int order_num,
    int N1,
    int N2,
    const DoubleDouble& branch_eps,
    int cheb_nu,
    int cheb_nv,
    const DoubleDouble& alpha_shift_up,
    const DoubleDouble& alpha_shift_down,
    const DoubleDouble& tol,
    const DoubleDouble& fpm_global_shift,
    const IntegrandRealT& ztransform_epsilon0_t,
    const IntegrandRealT& ztransform_safety_factor_t,
    std::vector<OutputRealT>& values,
    const IntegrandRuntimeScalars<VectorRealT>* cached_vector_scalars = nullptr,
    const IntegrandRuntimeScalars<IntegrandRealT>* cached_integrand_scalars = nullptr,
    int replica_count = 2) {
    if (order_num <= 1) {
        throw std::runtime_error("order_num must be greater than 1 in the all-subset CDet integrand");
    }

    constexpr int vertex_stride = 4;
    constexpr int FAST_ORDER_STORAGE_LIMIT = 16;
    const int order_sq = order_num * order_num;
    const size_t subset_count = size_t{1} << order_num;

    if constexpr (ENABLE_HOT_PATH_SANITY_CHECKS) {
        for (int i = 0, arg_count = vertex_stride * order_num; i < arg_count; ++i) {
            if (args[i] < DoubleDouble(0) || args[i] > DoubleDouble(1)) {
                values.assign(subset_count, OutputRealT(0));
                return;
            }
        }
    }

    const IntegrandRuntimeScalars<VectorRealT> local_vector_scalars =
        cached_vector_scalars == nullptr
            ? make_integrand_runtime_scalars<VectorRealT>(
                  beta_s, U_s, N1, N2, branch_eps, alpha_shift_up, alpha_shift_down, tol, fpm_global_shift)
            : IntegrandRuntimeScalars<VectorRealT>{};
    const IntegrandRuntimeScalars<VectorRealT>& vector_scalars =
        cached_vector_scalars == nullptr ? local_vector_scalars : *cached_vector_scalars;

    const IntegrandRuntimeScalars<IntegrandRealT> local_integrand_scalars =
        cached_integrand_scalars == nullptr
            ? make_integrand_runtime_scalars<IntegrandRealT>(
                  beta_s, U_s, N1, N2, branch_eps, alpha_shift_up, alpha_shift_down, tol, fpm_global_shift)
            : IntegrandRuntimeScalars<IntegrandRealT>{};
    const IntegrandRuntimeScalars<IntegrandRealT>& integrand_scalars =
        cached_integrand_scalars == nullptr ? local_integrand_scalars : *cached_integrand_scalars;

    const IntegrandRealT tol_t = integrand_scalars.tol;
    const VectorRealT beta_t = vector_scalars.beta;
    const VectorRealT branch_eps_t = vector_scalars.branch_eps;
    const VectorRealT scale = vector_scalars.scale;
    const VectorRealT diagonal_up_shift = vector_scalars.diagonal_up_shift;
    const VectorRealT diagonal_down_shift = vector_scalars.diagonal_down_shift;
    const IntegrandRealT fpm_global_shift_t = integrand_scalars.fpm_global_shift;
    const int requested_thread_count = get_integrand_internal_thread_count();

    std::array<SampledVertex<VectorRealT>, FAST_ORDER_STORAGE_LIMIT> vertex_fast{};
    std::array<VectorRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec1_fast{};
    std::array<VectorRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec2_fast{};
    std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT> diagonal_up_fast{};
    std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT> diagonal_down_fast{};

    thread_local std::vector<SampledVertex<VectorRealT>> vertex_slow;
    thread_local std::vector<VectorRealT> vec1_slow;
    thread_local std::vector<VectorRealT> vec2_slow;
    thread_local std::vector<IntegrandRealT> diagonal_up_slow;
    thread_local std::vector<IntegrandRealT> diagonal_down_slow;

    SampledVertex<VectorRealT>* vertices = nullptr;
    VectorRealT* vec1_data = nullptr;
    VectorRealT* vec2_data = nullptr;
    IntegrandRealT* diagonal_up_data = nullptr;
    IntegrandRealT* diagonal_down_data = nullptr;

    if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
        vertices = vertex_fast.data();
        vec1_data = vec1_fast.data();
        vec2_data = vec2_fast.data();
        diagonal_up_data = diagonal_up_fast.data();
        diagonal_down_data = diagonal_down_fast.data();
    } else {
        vertex_slow.resize(static_cast<size_t>(order_num));
        vec1_slow.resize(static_cast<size_t>(order_sq));
        vec2_slow.resize(static_cast<size_t>(order_sq));
        diagonal_up_slow.resize(static_cast<size_t>(order_num));
        diagonal_down_slow.resize(static_cast<size_t>(order_num));
        vertices = vertex_slow.data();
        vec1_data = vec1_slow.data();
        vec2_data = vec2_slow.data();
        diagonal_up_data = diagonal_up_slow.data();
        diagonal_down_data = diagonal_down_slow.data();
    }

    for (int i = 0; i < order_num; ++i) {
        const int vertex_offset = vertex_stride * i;
        vertices[i].tau_unit = integrand_real_from_dd<VectorRealT>(args[vertex_offset]);
        vertices[i].tau = beta_t * vertices[i].tau_unit;
        vertices[i].x = sampled_coordinate_to_index(
            VectorRealT(N1) * integrand_real_from_dd<VectorRealT>(args[vertex_offset + 1]), N1);
        vertices[i].y = sampled_coordinate_to_index(
            VectorRealT(N2) * integrand_real_from_dd<VectorRealT>(args[vertex_offset + 2]), N2);
        vertices[i].replica = sampled_replica_from_unit_coordinate_typed(
            integrand_real_from_dd<VectorRealT>(args[vertex_offset + 3]), replica_count);
    }

    auto& green_pair_cache =
        green_pair_matrix_cache_entry_for<VectorRealT, IntegrandRealT>(
            static_cast<const void*>(&g_table_1),
            static_cast<const void*>(&g_table_2));
    const bool can_reuse_green_pairs =
        green_pair_cache.order_num == order_num &&
        green_pair_cache.vertices.size() == static_cast<std::size_t>(order_num) &&
        green_pair_cache.up_matrix.size() == static_cast<std::size_t>(order_sq) &&
        green_pair_cache.down_matrix.size() == static_cast<std::size_t>(order_sq) &&
        green_pair_cache.diagonal_up.size() == static_cast<std::size_t>(order_num) &&
        green_pair_cache.diagonal_down.size() == static_cast<std::size_t>(order_num);

    std::array<int, FAST_ORDER_STORAGE_LIMIT> previous_index_fast{};
    std::vector<int> previous_index_slow;
    int* previous_indices = nullptr;
    if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
        previous_indices = previous_index_fast.data();
        std::fill(previous_index_fast.begin(), previous_index_fast.begin() + order_num, -1);
    } else {
        previous_index_slow.assign(static_cast<std::size_t>(order_num), -1);
        previous_indices = previous_index_slow.data();
    }
    if (can_reuse_green_pairs) {
        for (int i = 0; i < order_num; ++i) {
            for (int previous = 0; previous < order_num; ++previous) {
                if (sampled_vertices_equal(vertices[i],
                                           green_pair_cache.vertices[static_cast<std::size_t>(previous)])) {
                    previous_indices[i] = previous;
                    break;
                }
            }
        }
    }

    const int task_count = order_sq;
    const int max_parallel_threads =
        std::min<int>(requested_thread_count, task_count / MIN_PAIR_TASKS_PER_THREAD);
    std::array<std::uint64_t, MAX_INTERNAL_PARALLEL_THREADS> pair_cache_hits_by_worker{};
    std::array<std::uint64_t, MAX_INTERNAL_PARALLEL_THREADS> pair_cache_misses_by_worker{};

    auto fill_pair_range = [&](size_t worker_index, size_t begin, size_t end) {
        std::uint64_t pair_cache_hits = 0;
        std::uint64_t pair_cache_misses = 0;
        for (size_t linear_idx = begin; linear_idx < end; ++linear_idx) {
            const int idx = static_cast<int>(linear_idx);
            const int i = idx / order_num;
            const int j = idx - i * order_num;
            const int previous_i = previous_indices[i];
            const int previous_j = previous_indices[j];
            if (previous_i >= 0 && previous_j >= 0) {
                const std::size_t previous_idx =
                    static_cast<std::size_t>(previous_i * order_num + previous_j);
                vec1_data[idx] = green_pair_cache.up_matrix[previous_idx];
                vec2_data[idx] = green_pair_cache.down_matrix[previous_idx];
                if (i == j) {
                    diagonal_up_data[i] = green_pair_cache.diagonal_up[static_cast<std::size_t>(previous_i)];
                    diagonal_down_data[i] = green_pair_cache.diagonal_down[static_cast<std::size_t>(previous_i)];
                }
                ++pair_cache_hits;
                continue;
            }

            ++pair_cache_misses;
            const SampledVertex<VectorRealT>& vertex_i = vertices[i];
            const SampledVertex<VectorRealT>& vertex_j = vertices[j];
            const bool use_upper_branch = (i != j) && (vertex_i.tau_unit > vertex_j.tau_unit);

            const GreenLookupDescriptor<VectorRealT> descriptor = prepare_green_lookup_descriptor(
                vertex_i.replica,
                vertex_j.replica,
                vertex_i.tau,
                vertex_j.tau,
                vertex_i.x,
                vertex_i.y,
                vertex_j.x,
                vertex_j.y,
                beta_t,
                branch_eps_t,
                cheb_nu,
                cheb_nv,
                N1,
                N2,
                use_upper_branch);

            auto [up_value, down_value] =
                evaluate_green_pair_descriptor(descriptor, g_table_1, g_table_2, cheb_nu, cheb_nv);

            if (i == j) {
                diagonal_up_data[i] =
                    real_precision_cast<IntegrandRealT>(up_value) + integrand_scalars.diagonal_up_shift;
                diagonal_down_data[i] =
                    real_precision_cast<IntegrandRealT>(down_value) + integrand_scalars.diagonal_down_shift;
                up_value += diagonal_up_shift;
                down_value += diagonal_down_shift;
            }

            vec1_data[idx] = up_value * scale;
            vec2_data[idx] = down_value * scale;
        }
        if (worker_index < MAX_INTERNAL_PARALLEL_THREADS) {
            pair_cache_hits_by_worker[worker_index] += pair_cache_hits;
            pair_cache_misses_by_worker[worker_index] += pair_cache_misses;
        }
    };

    if (max_parallel_threads >= 2) {
        IntegrandParallelExecutor::instance().parallel_for(
            static_cast<size_t>(max_parallel_threads),
            static_cast<size_t>(task_count),
            fill_pair_range);
    } else {
        fill_pair_range(0, 0, static_cast<size_t>(task_count));
    }

    green_pair_cache.order_num = order_num;
    green_pair_cache.vertices.assign(vertices, vertices + order_num);
    green_pair_cache.up_matrix.assign(vec1_data, vec1_data + order_sq);
    green_pair_cache.down_matrix.assign(vec2_data, vec2_data + order_sq);
    green_pair_cache.diagonal_up.assign(diagonal_up_data, diagonal_up_data + order_num);
    green_pair_cache.diagonal_down.assign(diagonal_down_data, diagonal_down_data + order_num);

    std::uint64_t pair_cache_hits = 0;
    std::uint64_t pair_cache_misses = 0;
    for (std::size_t worker = 0; worker < pair_cache_hits_by_worker.size(); ++worker) {
        pair_cache_hits += pair_cache_hits_by_worker[worker];
        pair_cache_misses += pair_cache_misses_by_worker[worker];
    }
    record_cdet_green_pair_cache_counts(pair_cache_hits, pair_cache_misses);

    std::vector<IntegrandRealT> up_mat_minors;
    std::vector<IntegrandRealT> down_mat_minors;
    if constexpr (std::is_same_v<VectorRealT, IntegrandRealT>) {
        up_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec1_data, order_num, tol_t, fpm_global_shift_t);
        down_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec2_data, order_num, tol_t, fpm_global_shift_t);
    } else {
        std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec1_integrand_fast{};
        std::array<IntegrandRealT, FAST_ORDER_STORAGE_LIMIT * FAST_ORDER_STORAGE_LIMIT> vec2_integrand_fast{};
        thread_local std::vector<IntegrandRealT> vec1_integrand_slow;
        thread_local std::vector<IntegrandRealT> vec2_integrand_slow;
        IntegrandRealT* vec1_integrand_data = nullptr;
        IntegrandRealT* vec2_integrand_data = nullptr;
        if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
            vec1_integrand_data = vec1_integrand_fast.data();
            vec2_integrand_data = vec2_integrand_fast.data();
        } else {
            vec1_integrand_slow.resize(static_cast<size_t>(order_sq));
            vec2_integrand_slow.resize(static_cast<size_t>(order_sq));
            vec1_integrand_data = vec1_integrand_slow.data();
            vec2_integrand_data = vec2_integrand_slow.data();
        }
        for (int idx = 0; idx < order_sq; ++idx) {
            vec1_integrand_data[idx] = real_precision_cast<IntegrandRealT>(vec1_data[idx]);
            vec2_integrand_data[idx] = real_precision_cast<IntegrandRealT>(vec2_data[idx]);
        }
        up_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec1_integrand_data, order_num, tol_t, fpm_global_shift_t);
        down_mat_minors =
            mat2pm_from_buffer<IntegrandRealT>(vec2_integrand_data, order_num, tol_t, fpm_global_shift_t);
    }

    const std::vector<IntegrandRealT> connected_by_mask =
        compute_dp_ztransform_all_subsets(up_mat_minors,
                                          down_mat_minors,
                                          order_num,
                                          ztransform_epsilon0_t,
                                          ztransform_safety_factor_t);

    values.resize(subset_count);
    values[0] = OutputRealT(0);
    const IntegrandRealT counterterm =
        integrand_scalars.diagonal_up_shift * integrand_scalars.diagonal_down_shift;
    for (int vertex_index = 0; vertex_index < order_num; ++vertex_index) {
        const IntegrandRealT raw_order1 =
            IntegrandRealT(-replica_count) * integrand_scalars.u * integrand_scalars.beta *
            (diagonal_up_data[vertex_index] * diagonal_down_data[vertex_index] - counterterm);
        const OutputRealT value = cdet_output_from_integrand_real<OutputRealT>(raw_order1);
        if (!is_finite(value)) {
            throw std::runtime_error("Error (order-1 singleton integrand): Value is not finite");
        }
        values[size_t{1} << static_cast<size_t>(vertex_index)] = value;
    }

    using PrefactorRealT =
        std::conditional_t<std::is_same_v<OutputRealT, DoubleDouble>, DoubleDouble, IntegrandRealT>;
    std::array<PrefactorRealT, FAST_ORDER_STORAGE_LIMIT + 1> prefactors_fast{};
    std::vector<PrefactorRealT> prefactors_slow;
    PrefactorRealT* prefactors = nullptr;
    if (order_num <= FAST_ORDER_STORAGE_LIMIT) {
        prefactors = prefactors_fast.data();
    } else {
        prefactors_slow.assign(static_cast<size_t>(order_num + 1), PrefactorRealT(0));
        prefactors = prefactors_slow.data();
    }
    for (int subset_order = 2; subset_order <= order_num; ++subset_order) {
        prefactors[subset_order] =
            cdet_order_prefactor<PrefactorRealT>(subset_order, N1, N2, replica_count);
    }
    for (size_t mask = 1; mask < subset_count; ++mask) {
        const int subset_order = popcnt(static_cast<uint32_t>(mask));
        if (subset_order <= 1) {
            continue;
        }
        const PrefactorRealT raw_value =
            prefactors[subset_order] *
            cdet_output_from_integrand_real<PrefactorRealT>(connected_by_mask[mask]);
        const OutputRealT value =
            cdet_output_from_integrand_real<OutputRealT>(raw_value);
        if (!is_finite(value)) {
            throw std::runtime_error("Error (all-subset integrand): Value is not finite");
        }
        values[mask] = value;
    }
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline std::vector<DoubleDouble> integrand_all_subset_values_typed(
    const DoubleDouble& beta_s,
    const DoubleDouble& U_s,
    const DoubleDouble* args,
    const UpTableT& g_table_1,
    const DownTableT& g_table_2,
    int order_num,
    int N1,
    int N2,
    const DoubleDouble& branch_eps,
    int cheb_nu,
    int cheb_nv,
    const DoubleDouble& alpha_shift_up,
    const DoubleDouble& alpha_shift_down,
    const DoubleDouble& tol,
    const DoubleDouble& fpm_global_shift,
    const IntegrandRealT& ztransform_epsilon0_t,
    const IntegrandRealT& ztransform_safety_factor_t,
    int replica_count = 2) {
    std::vector<DoubleDouble> values;
    integrand_all_subset_values_typed_into<VectorRealT, IntegrandRealT, DoubleDouble>(
        beta_s,
        U_s,
        args,
        g_table_1,
        g_table_2,
        order_num,
        N1,
        N2,
        branch_eps,
        cheb_nu,
        cheb_nv,
        alpha_shift_up,
        alpha_shift_down,
        tol,
        fpm_global_shift,
        ztransform_epsilon0_t,
        ztransform_safety_factor_t,
        values,
        nullptr,
        nullptr,
        replica_count);
    return values;
}

inline DoubleDouble integrand_temp_dd(const DoubleDouble& beta_s,
                                      const DoubleDouble& U_s,
                                      const DoubleDouble* args,
                                      const GreenFunctionTableDD& g_table_1,
                                      const GreenFunctionTableDD& g_table_2,
                                      int order_num,
                                      int N1,
                                      int N2,
                                      const DoubleDouble& branch_eps,
                                      int cheb_nu,
                                      int cheb_nv,
                                      const DoubleDouble& alpha_shift_up,
                                      const DoubleDouble& alpha_shift_down,
                                      const DoubleDouble& tol) {
    return integrand_temp_typed<DoubleDouble, DoubleDouble>(
        beta_s,
        U_s,
        args,
        g_table_1,
        g_table_2,
        order_num,
        N1,
        N2,
        branch_eps,
        cheb_nu,
        cheb_nv,
        alpha_shift_up,
        alpha_shift_down,
        tol,
        DoubleDouble(-1),
        default_ztransform_epsilon0_for_precision<DoubleDouble>(),
        DoubleDouble(0));
}

#endif
