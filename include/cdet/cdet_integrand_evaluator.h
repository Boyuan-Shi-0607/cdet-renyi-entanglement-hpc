#ifndef CDET_INTEGRAND_EVALUATOR_H
#define CDET_INTEGRAND_EVALUATOR_H

#include "cdet/integrand_replica_randomly_sampled.h"
#include "cdet/parameters.h"
#include "cdet/runtime_counters.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

struct UnitVertex {
    double tau = 0.0;
    double x = 0.0;
    double y = 0.0;
    double replica = 0.0;
};

inline IntegrandRealType selected_integrand_real_type() {
    thread_local std::string cached_name;
    thread_local IntegrandRealType cached_type = IntegrandRealType::DoubleDouble;
    if (cached_name != integrand_real_name) {
        cached_type = parse_integrand_real_type(integrand_real_name);
        cached_name = integrand_real_name;
    }
    return cached_type;
}

inline IntegrandRealType selected_vector_real_type() {
    thread_local std::string cached_name;
    thread_local IntegrandRealType cached_type = IntegrandRealType::DoubleDouble;
    if (cached_name != vector_real_name) {
        cached_type = parse_vector_real_type(vector_real_name);
        cached_name = vector_real_name;
    }
    return cached_type;
}

template <typename RealT>
struct CachedZTransformRoundoff {
    RealT epsilon0;
    RealT safety_factor;
};

template <typename RealT>
inline CachedZTransformRoundoff<RealT> cached_ztransform_roundoff_for_real() {
    const auto& config = ztransform_roundoff_config();
    thread_local bool has_cache = false;
    thread_local bool cached_auto = false;
    thread_local std::string cached_epsilon0_literal;
    thread_local std::string cached_safety_factor_literal;
    thread_local CachedZTransformRoundoff<RealT> cached_values{};

    if (!has_cache ||
        cached_auto != config.epsilon0_is_auto() ||
        cached_epsilon0_literal != config.epsilon0_literal() ||
        cached_safety_factor_literal != config.safety_factor_literal()) {
        cached_values.epsilon0 = config.epsilon0<RealT>();
        cached_values.safety_factor = config.safety_factor<RealT>();
        cached_auto = config.epsilon0_is_auto();
        cached_epsilon0_literal = config.epsilon0_literal();
        cached_safety_factor_literal = config.safety_factor_literal();
        has_cache = true;
    }

    return cached_values;
}

template <typename RealT>
inline const IntegrandRuntimeScalars<RealT>& cached_integrand_runtime_scalars_for_real() {
    thread_local bool has_cache = false;
    thread_local int cached_N1 = 0;
    thread_local int cached_N2 = 0;
    thread_local IntegrandRuntimeScalars<RealT> cached_scalars;

    if (!has_cache || cached_N1 != N1 || cached_N2 != N2) {
        cached_scalars = make_integrand_runtime_scalars<RealT>(DoubleDouble(beta),
                                                               DoubleDouble(U),
                                                               N1,
                                                               N2,
                                                               DoubleDouble(branch_epsilon),
                                                               DoubleDouble(alpha_shift_up),
                                                               DoubleDouble(alpha_shift_down),
                                                               DoubleDouble(tol_fpm),
                                                               DoubleDouble(fpm_global_shift));
        cached_N1 = N1;
        cached_N2 = N2;
        has_cache = true;
    }

    return cached_scalars;
}

template <typename VectorRealT>
inline const PiecewiseChebyshevTable<VectorRealT>& require_resident_green_table(
    const ResidentGreenFunctionTable& table,
    const char* table_name) {
    const auto* typed_table = std::get_if<PiecewiseChebyshevTable<VectorRealT>>(&table);
    if (typed_table == nullptr) {
        throw std::runtime_error(
            std::string("Resident Green-function table ") + table_name +
            " type does not match VECTOR_REAL=" + vector_real_name);
    }
    return *typed_table;
}

template <typename VectorRealT>
inline const DiskBackedPiecewiseChebyshevTable<VectorRealT>& require_disk_green_table(
    const DiskBackedGreenFunctionTable& table,
    const char* table_name) {
    const auto* typed_table = std::get_if<DiskBackedPiecewiseChebyshevTable<VectorRealT>>(&table);
    if (typed_table == nullptr) {
        throw std::runtime_error(
            std::string("Disk-backed Green-function table ") + table_name +
            " type does not match VECTOR_REAL=" + vector_real_name);
    }
    if (!g_disk_tables_initialized) {
        throw std::runtime_error("Disk-backed Green-function tables have not been initialized");
    }
    return *typed_table;
}

template <typename VectorRealT>
inline SampledVertex<VectorRealT> sampled_vertex_from_unit_args(
    const DoubleDouble* args,
    const IntegrandRuntimeScalars<VectorRealT>& scalars,
    int replica_count) {
    SampledVertex<VectorRealT> vertex;
    vertex.tau_unit = integrand_real_from_dd<VectorRealT>(args[0]);
    vertex.tau = scalars.beta * vertex.tau_unit;
    vertex.x = sampled_coordinate_to_index(
        VectorRealT(N1) * integrand_real_from_dd<VectorRealT>(args[1]), N1);
    vertex.y = sampled_coordinate_to_index(
        VectorRealT(N2) * integrand_real_from_dd<VectorRealT>(args[2]), N2);
    vertex.replica = sampled_replica_from_unit_coordinate_typed(
        integrand_real_from_dd<VectorRealT>(args[3]), replica_count);
    return vertex;
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline DoubleDouble evaluate_cdet_density_order1_integrand_for_real_with_tables(
    const DoubleDouble* args,
    const UpTableT& up_table,
    const DownTableT& down_table,
    int replica_count) {
    const auto& vector_scalars = cached_integrand_runtime_scalars_for_real<VectorRealT>();
    const auto& integrand_scalars = cached_integrand_runtime_scalars_for_real<IntegrandRealT>();
    const SampledVertex<VectorRealT> vertex =
        sampled_vertex_from_unit_args(args, vector_scalars, replica_count);

    const IntegrandRealT value =
        order1_density_integrand_from_sampled_vertex<VectorRealT, IntegrandRealT>(
            vertex,
            up_table,
            down_table,
            N1,
            N2,
            cheb_nu,
            cheb_nv,
            vector_scalars,
            integrand_scalars,
            replica_count);
    return double_double_from_integrand_real(value);
}

template <typename VectorRealT, typename IntegrandRealT>
inline DoubleDouble evaluate_cdet_density_order1_integrand_for_real(const DoubleDouble* args) {
    if (mcmcmc_table_backend == "disk_lru") {
        const auto& twisted_up = require_disk_green_table<VectorRealT>(g_disk_table_up, "g_disk_table_up");
        const auto& twisted_down = require_disk_green_table<VectorRealT>(g_disk_table_down, "g_disk_table_down");
        const auto& ordinary_up =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_up, "g_disk_table_ordinary_up");
        const auto& ordinary_down =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_down, "g_disk_table_ordinary_down");
        const DoubleDouble twisted =
            evaluate_cdet_density_order1_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, twisted_up, twisted_down, 2);
        const DoubleDouble ordinary =
            evaluate_cdet_density_order1_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, ordinary_up, ordinary_down, 1);
        return twisted - DoubleDouble(2) * ordinary;
    }

    const auto& twisted_up = require_resident_green_table<VectorRealT>(g_table_up, "g_table_up");
    const auto& twisted_down = require_resident_green_table<VectorRealT>(g_table_down, "g_table_down");
    const auto& ordinary_up =
        require_resident_green_table<VectorRealT>(g_table_ordinary_up, "g_table_ordinary_up");
    const auto& ordinary_down =
        require_resident_green_table<VectorRealT>(g_table_ordinary_down, "g_table_ordinary_down");
    const DoubleDouble twisted =
        evaluate_cdet_density_order1_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, twisted_up, twisted_down, 2);
    const DoubleDouble ordinary =
        evaluate_cdet_density_order1_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, ordinary_up, ordinary_down, 1);
    return twisted - DoubleDouble(2) * ordinary;
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline DoubleDouble evaluate_cdet_density_integrand_for_real_with_tables(
    const DoubleDouble* args,
    int order,
    const UpTableT& up_table,
    const DownTableT& down_table,
    int replica_count) {
    const auto roundoff = cached_ztransform_roundoff_for_real<IntegrandRealT>();

    return integrand_temp_typed<VectorRealT, IntegrandRealT>(DoubleDouble(beta),
                                                            DoubleDouble(U),
                                                            args,
                                                            up_table,
                                                            down_table,
                                                            order,
                                                            N1,
                                                            N2,
                                                            DoubleDouble(branch_epsilon),
                                                            cheb_nu,
                                                            cheb_nv,
                                                            DoubleDouble(alpha_shift_up),
                                                            DoubleDouble(alpha_shift_down),
                                                            DoubleDouble(tol_fpm),
                                                            DoubleDouble(fpm_global_shift),
                                                            roundoff.epsilon0,
                                                            roundoff.safety_factor,
                                                            replica_count);
}

template <typename VectorRealT, typename IntegrandRealT>
inline DoubleDouble evaluate_cdet_density_integrand_for_real(const DoubleDouble* args, int order) {
    if (mcmcmc_table_backend == "disk_lru") {
        const auto& twisted_up = require_disk_green_table<VectorRealT>(g_disk_table_up, "g_disk_table_up");
        const auto& twisted_down = require_disk_green_table<VectorRealT>(g_disk_table_down, "g_disk_table_down");
        const auto& ordinary_up =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_up, "g_disk_table_ordinary_up");
        const auto& ordinary_down =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_down, "g_disk_table_ordinary_down");
        const DoubleDouble twisted =
            evaluate_cdet_density_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, order, twisted_up, twisted_down, 2);
        const DoubleDouble ordinary =
            evaluate_cdet_density_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, order, ordinary_up, ordinary_down, 1);
        return twisted - DoubleDouble(2) * ordinary;
    }

    const auto& twisted_up = require_resident_green_table<VectorRealT>(g_table_up, "g_table_up");
    const auto& twisted_down = require_resident_green_table<VectorRealT>(g_table_down, "g_table_down");
    const auto& ordinary_up =
        require_resident_green_table<VectorRealT>(g_table_ordinary_up, "g_table_ordinary_up");
    const auto& ordinary_down =
        require_resident_green_table<VectorRealT>(g_table_ordinary_down, "g_table_ordinary_down");
    const DoubleDouble twisted =
        evaluate_cdet_density_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, order, twisted_up, twisted_down, 2);
    const DoubleDouble ordinary =
        evaluate_cdet_density_integrand_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, order, ordinary_up, ordinary_down, 1);
    return twisted - DoubleDouble(2) * ordinary;
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline void evaluate_cdet_density_all_subset_integrands_into_for_real_with_tables(
    const DoubleDouble* args,
    int envelope_order,
    const UpTableT& up_table,
    const DownTableT& down_table,
    int replica_count,
    std::vector<double>& values) {
    const auto roundoff = cached_ztransform_roundoff_for_real<IntegrandRealT>();
    const auto& vector_scalars = cached_integrand_runtime_scalars_for_real<VectorRealT>();
    const auto& integrand_scalars = cached_integrand_runtime_scalars_for_real<IntegrandRealT>();
    static const DoubleDouble unused_zero(0);
    static const DoubleDouble unused_one(1);

    integrand_all_subset_values_typed_into<VectorRealT, IntegrandRealT, double>(
        unused_zero,
        unused_one,
        args,
        up_table,
        down_table,
        envelope_order,
        N1,
        N2,
        unused_zero,
        cheb_nu,
        cheb_nv,
        unused_zero,
        unused_zero,
        unused_zero,
        DoubleDouble(fpm_global_shift),
        roundoff.epsilon0,
        roundoff.safety_factor,
        values,
        &vector_scalars,
        &integrand_scalars,
        replica_count);
}

template <typename VectorRealT, typename IntegrandRealT>
inline void evaluate_cdet_density_all_subset_integrands_into_for_real(const DoubleDouble* args,
                                                                      int envelope_order,
                                                                      std::vector<double>& values) {
    if (mcmcmc_table_backend == "disk_lru") {
        const auto& twisted_up = require_disk_green_table<VectorRealT>(g_disk_table_up, "g_disk_table_up");
        const auto& twisted_down = require_disk_green_table<VectorRealT>(g_disk_table_down, "g_disk_table_down");
        const auto& ordinary_up =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_up, "g_disk_table_ordinary_up");
        const auto& ordinary_down =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_down, "g_disk_table_ordinary_down");

        evaluate_cdet_density_all_subset_integrands_into_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, envelope_order, twisted_up, twisted_down, 2, values);

        thread_local std::vector<double> ordinary_values;
        evaluate_cdet_density_all_subset_integrands_into_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, envelope_order, ordinary_up, ordinary_down, 1, ordinary_values);

        if (ordinary_values.size() != values.size()) {
            throw std::runtime_error("Internal error: ordinary and twisted subset arrays have different sizes");
        }
        for (std::size_t mask = 0; mask < values.size(); ++mask) {
            values[mask] -= 2.0 * ordinary_values[mask];
        }
        return;
    }

    const auto& twisted_up = require_resident_green_table<VectorRealT>(g_table_up, "g_table_up");
    const auto& twisted_down = require_resident_green_table<VectorRealT>(g_table_down, "g_table_down");
    const auto& ordinary_up =
        require_resident_green_table<VectorRealT>(g_table_ordinary_up, "g_table_ordinary_up");
    const auto& ordinary_down =
        require_resident_green_table<VectorRealT>(g_table_ordinary_down, "g_table_ordinary_down");

    evaluate_cdet_density_all_subset_integrands_into_for_real_with_tables<VectorRealT, IntegrandRealT>(
        args, envelope_order, twisted_up, twisted_down, 2, values);

    thread_local std::vector<double> ordinary_values;
    evaluate_cdet_density_all_subset_integrands_into_for_real_with_tables<VectorRealT, IntegrandRealT>(
        args, envelope_order, ordinary_up, ordinary_down, 1, ordinary_values);

    if (ordinary_values.size() != values.size()) {
        throw std::runtime_error("Internal error: ordinary and twisted subset arrays have different sizes");
    }
    for (std::size_t mask = 0; mask < values.size(); ++mask) {
        values[mask] -= 2.0 * ordinary_values[mask];
    }
}

template <typename VectorRealT, typename IntegrandRealT, typename UpTableT, typename DownTableT>
inline std::vector<DoubleDouble> evaluate_cdet_density_all_subset_integrands_for_real_with_tables(
    const DoubleDouble* args,
    int envelope_order,
    const UpTableT& up_table,
    const DownTableT& down_table,
    int replica_count) {
    const auto roundoff = cached_ztransform_roundoff_for_real<IntegrandRealT>();

    return integrand_all_subset_values_typed<VectorRealT, IntegrandRealT>(
        DoubleDouble(beta),
        DoubleDouble(U),
        args,
        up_table,
        down_table,
        envelope_order,
        N1,
        N2,
        DoubleDouble(branch_epsilon),
        cheb_nu,
        cheb_nv,
        DoubleDouble(alpha_shift_up),
        DoubleDouble(alpha_shift_down),
        DoubleDouble(tol_fpm),
        DoubleDouble(fpm_global_shift),
        roundoff.epsilon0,
        roundoff.safety_factor,
        replica_count);
}

template <typename VectorRealT, typename IntegrandRealT>
inline std::vector<DoubleDouble> evaluate_cdet_density_all_subset_integrands_for_real(const DoubleDouble* args,
                                                                                     int envelope_order) {
    if (mcmcmc_table_backend == "disk_lru") {
        const auto& twisted_up = require_disk_green_table<VectorRealT>(g_disk_table_up, "g_disk_table_up");
        const auto& twisted_down = require_disk_green_table<VectorRealT>(g_disk_table_down, "g_disk_table_down");
        const auto& ordinary_up =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_up, "g_disk_table_ordinary_up");
        const auto& ordinary_down =
            require_disk_green_table<VectorRealT>(g_disk_table_ordinary_down, "g_disk_table_ordinary_down");

        std::vector<DoubleDouble> values =
            evaluate_cdet_density_all_subset_integrands_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, envelope_order, twisted_up, twisted_down, 2);
        const std::vector<DoubleDouble> ordinary_values =
            evaluate_cdet_density_all_subset_integrands_for_real_with_tables<VectorRealT, IntegrandRealT>(
                args, envelope_order, ordinary_up, ordinary_down, 1);
        if (ordinary_values.size() != values.size()) {
            throw std::runtime_error("Internal error: ordinary and twisted subset arrays have different sizes");
        }
        for (std::size_t mask = 0; mask < values.size(); ++mask) {
            values[mask] -= DoubleDouble(2) * ordinary_values[mask];
        }
        return values;
    }

    const auto& twisted_up = require_resident_green_table<VectorRealT>(g_table_up, "g_table_up");
    const auto& twisted_down = require_resident_green_table<VectorRealT>(g_table_down, "g_table_down");
    const auto& ordinary_up =
        require_resident_green_table<VectorRealT>(g_table_ordinary_up, "g_table_ordinary_up");
    const auto& ordinary_down =
        require_resident_green_table<VectorRealT>(g_table_ordinary_down, "g_table_ordinary_down");

    std::vector<DoubleDouble> values =
        evaluate_cdet_density_all_subset_integrands_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, envelope_order, twisted_up, twisted_down, 2);
    const std::vector<DoubleDouble> ordinary_values =
        evaluate_cdet_density_all_subset_integrands_for_real_with_tables<VectorRealT, IntegrandRealT>(
            args, envelope_order, ordinary_up, ordinary_down, 1);
    if (ordinary_values.size() != values.size()) {
        throw std::runtime_error("Internal error: ordinary and twisted subset arrays have different sizes");
    }
    for (std::size_t mask = 0; mask < values.size(); ++mask) {
        values[mask] -= DoubleDouble(2) * ordinary_values[mask];
    }
    return values;
}

template <typename VectorRealT>
inline DoubleDouble evaluate_cdet_density_integrand_with_vector_real(const DoubleDouble* args, int order) {
    const IntegrandRealType integrand_real_type = selected_integrand_real_type();
    if (order == 1) {
        switch (integrand_real_type) {
            case IntegrandRealType::Double:
                return evaluate_cdet_density_order1_integrand_for_real<VectorRealT, double>(args);
            case IntegrandRealType::LongDouble:
                return evaluate_cdet_density_order1_integrand_for_real<VectorRealT, long double>(args);
            case IntegrandRealType::DoubleDouble:
                return evaluate_cdet_density_order1_integrand_for_real<VectorRealT, DoubleDouble>(args);
        }
    }
    switch (integrand_real_type) {
        case IntegrandRealType::Double:
            return evaluate_cdet_density_integrand_for_real<VectorRealT, double>(args, order);
        case IntegrandRealType::LongDouble:
            return evaluate_cdet_density_integrand_for_real<VectorRealT, long double>(args, order);
        case IntegrandRealType::DoubleDouble:
            return evaluate_cdet_density_integrand_for_real<VectorRealT, DoubleDouble>(args, order);
    }
    throw std::runtime_error("Unhandled integrand real type");
}

inline DoubleDouble evaluate_cdet_density_integrand_with_selected_real(const DoubleDouble* args, int order) {
    switch (selected_vector_real_type()) {
        case IntegrandRealType::Double:
            return evaluate_cdet_density_integrand_with_vector_real<double>(args, order);
        case IntegrandRealType::LongDouble:
            return evaluate_cdet_density_integrand_with_vector_real<long double>(args, order);
        case IntegrandRealType::DoubleDouble:
            return evaluate_cdet_density_integrand_with_vector_real<DoubleDouble>(args, order);
    }
    throw std::runtime_error("Unhandled vector real type");
}

template <typename VectorRealT>
inline std::vector<DoubleDouble> evaluate_cdet_density_all_subset_integrands_with_vector_real(
    const DoubleDouble* args,
    int envelope_order) {
    const IntegrandRealType integrand_real_type = selected_integrand_real_type();
    switch (integrand_real_type) {
        case IntegrandRealType::Double:
            return evaluate_cdet_density_all_subset_integrands_for_real<VectorRealT, double>(args, envelope_order);
        case IntegrandRealType::LongDouble:
            return evaluate_cdet_density_all_subset_integrands_for_real<VectorRealT, long double>(args, envelope_order);
        case IntegrandRealType::DoubleDouble:
            return evaluate_cdet_density_all_subset_integrands_for_real<VectorRealT, DoubleDouble>(
                args, envelope_order);
    }
    throw std::runtime_error("Unhandled integrand real type");
}

inline std::vector<DoubleDouble> evaluate_cdet_density_all_subset_integrands_with_selected_real(
    const DoubleDouble* args,
    int envelope_order) {
    switch (selected_vector_real_type()) {
        case IntegrandRealType::Double:
            return evaluate_cdet_density_all_subset_integrands_with_vector_real<double>(args, envelope_order);
        case IntegrandRealType::LongDouble:
            return evaluate_cdet_density_all_subset_integrands_with_vector_real<long double>(args, envelope_order);
        case IntegrandRealType::DoubleDouble:
            return evaluate_cdet_density_all_subset_integrands_with_vector_real<DoubleDouble>(args, envelope_order);
    }
    throw std::runtime_error("Unhandled vector real type");
}

template <typename VectorRealT>
inline void evaluate_cdet_density_all_subset_integrands_into_with_vector_real(
    const DoubleDouble* args,
    int envelope_order,
    std::vector<double>& values) {
    const IntegrandRealType integrand_real_type = selected_integrand_real_type();
    switch (integrand_real_type) {
        case IntegrandRealType::Double:
            evaluate_cdet_density_all_subset_integrands_into_for_real<VectorRealT, double>(
                args, envelope_order, values);
            return;
        case IntegrandRealType::LongDouble:
            evaluate_cdet_density_all_subset_integrands_into_for_real<VectorRealT, long double>(
                args, envelope_order, values);
            return;
        case IntegrandRealType::DoubleDouble:
            evaluate_cdet_density_all_subset_integrands_into_for_real<VectorRealT, DoubleDouble>(
                args, envelope_order, values);
            return;
    }
    throw std::runtime_error("Unhandled integrand real type");
}

inline void evaluate_cdet_density_all_subset_integrands_into_with_selected_real(
    const DoubleDouble* args,
    int envelope_order,
    std::vector<double>& values) {
    switch (selected_vector_real_type()) {
        case IntegrandRealType::Double:
            evaluate_cdet_density_all_subset_integrands_into_with_vector_real<double>(
                args, envelope_order, values);
            return;
        case IntegrandRealType::LongDouble:
            evaluate_cdet_density_all_subset_integrands_into_with_vector_real<long double>(
                args, envelope_order, values);
            return;
        case IntegrandRealType::DoubleDouble:
            evaluate_cdet_density_all_subset_integrands_into_with_vector_real<DoubleDouble>(
                args, envelope_order, values);
            return;
    }
    throw std::runtime_error("Unhandled vector real type");
}

inline double evaluate_cdet_density_integrand_double(const std::vector<UnitVertex>& vertices) {
    const int order = static_cast<int>(vertices.size());
    if (order <= 0) {
        throw std::runtime_error("The replica integrand is defined only for order >= 1");
    }
    record_cdet_integrand_call();

    thread_local std::vector<DoubleDouble> args_buffer;
    args_buffer.resize(static_cast<std::size_t>(4 * order));
    for (int i = 0; i < order; ++i) {
        const std::size_t offset = static_cast<std::size_t>(4 * i);
        args_buffer[offset] = DoubleDouble(vertices[static_cast<std::size_t>(i)].tau);
        args_buffer[offset + 1] = DoubleDouble(vertices[static_cast<std::size_t>(i)].x);
        args_buffer[offset + 2] = DoubleDouble(vertices[static_cast<std::size_t>(i)].y);
        args_buffer[offset + 3] = DoubleDouble(vertices[static_cast<std::size_t>(i)].replica);
    }

    return evaluate_cdet_density_integrand_with_selected_real(args_buffer.data(), order).to_double();
}

inline void evaluate_cdet_density_all_subset_integrands_double_into(
    const std::vector<UnitVertex>& vertices,
    std::vector<double>& values) {
    const int envelope_order = static_cast<int>(vertices.size());
    if (envelope_order <= 1) {
        throw std::runtime_error("The all-subset replica integrand is defined only for envelope order >= 2");
    }
    record_cdet_integrand_call();

    thread_local std::vector<DoubleDouble> args_buffer;
    args_buffer.resize(static_cast<std::size_t>(4 * envelope_order));
    for (int i = 0; i < envelope_order; ++i) {
        const std::size_t offset = static_cast<std::size_t>(4 * i);
        args_buffer[offset] = DoubleDouble(vertices[static_cast<std::size_t>(i)].tau);
        args_buffer[offset + 1] = DoubleDouble(vertices[static_cast<std::size_t>(i)].x);
        args_buffer[offset + 2] = DoubleDouble(vertices[static_cast<std::size_t>(i)].y);
        args_buffer[offset + 3] = DoubleDouble(vertices[static_cast<std::size_t>(i)].replica);
    }

    evaluate_cdet_density_all_subset_integrands_into_with_selected_real(
        args_buffer.data(), envelope_order, values);
}

inline std::vector<double> evaluate_cdet_density_all_subset_integrands_double(
    const std::vector<UnitVertex>& vertices) {
    std::vector<double> values;
    evaluate_cdet_density_all_subset_integrands_double_into(vertices, values);
    return values;
}

template <typename FullVertexContainer>
inline std::vector<UnitVertex> subset_vertices(const FullVertexContainer& full_vertices, std::uint64_t mask) {
    std::vector<UnitVertex> result;
    result.reserve(static_cast<std::size_t>(__builtin_popcountll(mask)));
    for (std::size_t i = 0; i < full_vertices.size(); ++i) {
        if ((mask & (std::uint64_t{1} << i)) != 0U) {
            result.push_back(full_vertices[i]);
        }
    }
    return result;
}

#endif
