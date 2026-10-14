#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include "cdet/chunked_load_data.h"
#include "cdet/parameters.h"
#include "cdet/precision_config.h"

#include <chrono>
#include <iostream>

template <typename BuildRealT>
inline ReplicaGreenCalculator<BuildRealT> make_resident_calculator_with_optional_cache(
    ChunkedGreenTableKind kind) {
    if (load_data_calculator_cache == "off") {
        return make_chunked_calculator<BuildRealT>(kind);
    }

    const std::filesystem::path cache_path = chunked_calculator_cache_path(kind);
    if (std::filesystem::exists(cache_path)) {
        std::cout << "  - Loading resident calculator cache: " << cache_path << '\n';
        return read_chunked_calculator_cache<BuildRealT>(kind);
    }
    if (load_data_calculator_cache == "require") {
        throw std::runtime_error("Missing required resident calculator cache: " + cache_path.string());
    }

    std::cout << "  - Building resident calculator cache for "
              << chunked_table_kind_name(kind) << '\n';
    ReplicaGreenCalculator<BuildRealT> calculator = make_chunked_calculator<BuildRealT>(kind);
    write_chunked_calculator_cache(kind, calculator);
    std::cout << "  - Wrote resident calculator cache: " << cache_path << '\n';
    return calculator;
}

template <typename BuildRealT, typename TargetRealT>
inline PiecewiseChebyshevTable<TargetRealT> build_spin_green_table_resident(ChunkedGreenTableKind kind) {
    const ReplicaGreenCalculator<BuildRealT> calculator =
        make_resident_calculator_with_optional_cache<BuildRealT>(kind);
    if (kind == ChunkedGreenTableKind::OrdinaryUp || kind == ChunkedGreenTableKind::OrdinaryDown) {
        std::cout << "  - Ordinary resident table stores only the replica (0,0) block used by Z1\n";
        return convert_green_function_table<TargetRealT>(
            build_ordinary_replica00_green_function_table(
                calculator, cheb_nu, cheb_nv, table_build_threads));
    }
    return build_green_function_table_as<BuildRealT, TargetRealT>(
        calculator, cheb_nu, cheb_nv, table_build_threads);
}

template <typename BuildRealT, typename TargetRealT>
inline void load_data_into_resident_tables() {
    std::cout << "  - Load-data construction real type = " << real_type_name<BuildRealT>() << '\n';
    std::cout << "  - Resident Green-table real type = " << real_type_name<TargetRealT>() << '\n';
    if (green_table_drop_bits > 0) {
        std::cout << "  - Resident Green-table mantissa drop bits = " << green_table_drop_bits
                  << " (native bits = " << resident_green_table_mantissa_digits<TargetRealT>()
                  << ", kept bits = " << resident_green_table_keep_bits<TargetRealT>(green_table_drop_bits)
                  << ")\n";
    }

    GreenTableTruncationSummary truncation_summary;
    auto build_and_truncate = [&](ChunkedGreenTableKind kind) {
        PiecewiseChebyshevTable<TargetRealT> table =
            build_spin_green_table_resident<BuildRealT, TargetRealT>(kind);
        truncation_summary.merge(
            truncate_green_function_table_mantissa(table, green_table_drop_bits));
        return table;
    };

    std::cout << "  - Building twisted n=2 replica Green tables\n";
    if (alpha_shift_up == alpha_shift_down) {
        std::cout << "  - ALPHA_UP == ALPHA_DOWN; reusing one twisted resident table for up/down\n";
        PiecewiseChebyshevTable<TargetRealT> twisted =
            build_and_truncate(ChunkedGreenTableKind::TwistedUp);
        g_table_up = twisted;
        g_table_down = std::move(twisted);
    } else {
        g_table_up = build_and_truncate(ChunkedGreenTableKind::TwistedUp);
        g_table_down = build_and_truncate(ChunkedGreenTableKind::TwistedDown);
    }

    std::cout << "  - Building ordinary n=1 thermal Green tables for Z1 subtraction\n";
    if (alpha_shift_up == alpha_shift_down) {
        std::cout << "  - ALPHA_UP == ALPHA_DOWN; reusing one ordinary resident table for up/down\n";
        PiecewiseChebyshevTable<TargetRealT> ordinary =
            build_and_truncate(ChunkedGreenTableKind::OrdinaryUp);
        g_table_ordinary_up = ordinary;
        g_table_ordinary_down = std::move(ordinary);
    } else {
        g_table_ordinary_up = build_and_truncate(ChunkedGreenTableKind::OrdinaryUp);
        g_table_ordinary_down = build_and_truncate(ChunkedGreenTableKind::OrdinaryDown);
    }

    if (green_table_drop_bits > 0) {
        std::cout << "  - Green-table truncation values = " << truncation_summary.value_count
                  << ", max_abs_delta ~= " << static_cast<double>(truncation_summary.max_abs_delta)
                  << ", max_rel_delta ~= " << static_cast<double>(truncation_summary.max_rel_delta)
                  << '\n';
    }
}

template <typename TargetRealT>
inline void dispatch_load_data_build_type(LoadDataRealType build_type) {
    switch (build_type) {
        case LoadDataRealType::MP:
            load_data_into_resident_tables<MP_Real, TargetRealT>();
            break;
        case LoadDataRealType::Double:
            load_data_into_resident_tables<double, TargetRealT>();
            break;
        case LoadDataRealType::LongDouble:
            load_data_into_resident_tables<long double, TargetRealT>();
            break;
        case LoadDataRealType::DoubleDouble:
            load_data_into_resident_tables<DoubleDouble, TargetRealT>();
            break;
    }
}

inline void load_data() {
    const auto load_data_start = std::chrono::steady_clock::now();

    if (load_data_workflow == "load_chunks") {
        std::cout << "Loading prebuilt chunked replica Green-function tables from "
                  << load_data_chunk_dir << '\n';
        if (mcmcmc_table_backend == "disk_lru") {
            load_chunked_data_into_disk_lru_tables();
        } else {
            load_chunked_data_into_resident_tables();
        }
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - load_data_start;
        std::cout << "load_data() elapsed time = " << elapsed.count() << " s\n";
        return;
    }
    if (mcmcmc_table_backend == "disk_lru") {
        throw std::runtime_error(
            "MCMCMC_TABLE_BACKEND=disk_lru requires LOAD_DATA_WORKFLOW=load_chunks "
            "so coefficient chunks can be read on demand");
    }
    if (load_data_workflow != "resident") {
        throw std::runtime_error(
            "load_data() can only run with LOAD_DATA_WORKFLOW=resident or load_chunks; "
            "standalone chunk stages must exit before MCMCMC starts");
    }

    const ReplicaSubregionMask mask_type = parse_subregion_mask(subregion_mask_name);
    const int origin_square_radius =
        mask_type == ReplicaSubregionMask::OriginSquare ? subregion_square_radius : 0;
    const LoadDataRealType build_type = parse_load_data_real_type(load_data_real_name);
    const IntegrandRealType resident_type = parse_vector_real_type(vector_real_name);

    std::cout << "Loading replica Green-function data for one parameter set:\n";
    std::cout << "  - N1 = " << N1 << '\n';
    std::cout << "  - N2 = " << N2 << '\n';
    std::cout << "  - beta = " << beta << '\n';
    std::cout << "  - Subregion mask = " << subregion_mask_to_string(mask_type) << '\n';
    if (mask_type == ReplicaSubregionMask::OriginSquare) {
        std::cout << "  - Origin-square radius = " << origin_square_radius << '\n';
    }
    std::cout << "  - Spatial storage: full absolute-coordinate table [R1][R2]\n";
    std::cout << "  - Table build threads = " << table_build_threads << '\n';

    switch (resident_type) {
        case IntegrandRealType::Double:
            dispatch_load_data_build_type<double>(build_type);
            break;
        case IntegrandRealType::LongDouble:
            dispatch_load_data_build_type<long double>(build_type);
            break;
        case IntegrandRealType::DoubleDouble:
            dispatch_load_data_build_type<DoubleDouble>(build_type);
            break;
    }

    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - load_data_start;
    std::cout << "load_data() elapsed time = " << elapsed.count() << " s\n";
}

#endif
