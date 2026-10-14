#ifndef CDET_CHUNKED_LOAD_DATA_H
#define CDET_CHUNKED_LOAD_DATA_H

#include "cdet/diagonal_green_observable.h"
#include "cdet/parameters.h"
#include "cdet/precision_config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

enum class ChunkedGreenTableKind {
    TwistedUp,
    TwistedDown,
    OrdinaryUp,
    OrdinaryDown,
};

struct ChunkedNodeJob {
    int index = 0;
    ChunkedGreenTableKind kind = ChunkedGreenTableKind::TwistedUp;
    TriangleBranch branch = TriangleBranch::Upper;
    int node_begin = 0;
    int node_end = 0;
};

struct ChunkedSitepairJob {
    int index = 0;
    ChunkedGreenTableKind kind = ChunkedGreenTableKind::TwistedUp;
    TriangleBranch branch = TriangleBranch::Upper;
    size_t sitepair_begin = 0;
    size_t sitepair_end = 0;
};

struct ChunkedCornerJob {
    int index = 0;
    ChunkedGreenTableKind kind = ChunkedGreenTableKind::TwistedUp;
    TriangleBranch branch = TriangleBranch::Upper;
};

inline std::array<ChunkedGreenTableKind, 4> chunked_table_kinds() {
    return {
        ChunkedGreenTableKind::TwistedUp,
        ChunkedGreenTableKind::TwistedDown,
        ChunkedGreenTableKind::OrdinaryUp,
        ChunkedGreenTableKind::OrdinaryDown,
    };
}

inline std::array<TriangleBranch, 2> chunked_branches() {
    return {TriangleBranch::Upper, TriangleBranch::Lower};
}

inline const char* chunked_table_kind_name(ChunkedGreenTableKind kind) {
    switch (kind) {
        case ChunkedGreenTableKind::TwistedUp:
            return "twisted_up";
        case ChunkedGreenTableKind::TwistedDown:
            return "twisted_down";
        case ChunkedGreenTableKind::OrdinaryUp:
            return "ordinary_up";
        case ChunkedGreenTableKind::OrdinaryDown:
            return "ordinary_down";
    }
    throw std::runtime_error("Unhandled chunked table kind");
}

inline ChunkedGreenTableKind parse_chunked_table_kind(const std::string& raw_name) {
    const std::string name = normalize_real_type_name(raw_name);
    if (name == "twisted_up") {
        return ChunkedGreenTableKind::TwistedUp;
    }
    if (name == "twisted_down") {
        return ChunkedGreenTableKind::TwistedDown;
    }
    if (name == "ordinary_up") {
        return ChunkedGreenTableKind::OrdinaryUp;
    }
    if (name == "ordinary_down") {
        return ChunkedGreenTableKind::OrdinaryDown;
    }
    throw std::runtime_error("Unknown chunked table kind: " + raw_name);
}

inline const char* chunked_branch_name(TriangleBranch branch) {
    switch (branch) {
        case TriangleBranch::Upper:
            return "upper";
        case TriangleBranch::Lower:
            return "lower";
    }
    throw std::runtime_error("Unhandled triangle branch");
}

inline TriangleBranch parse_chunked_branch(const std::string& raw_name) {
    const std::string name = normalize_real_type_name(raw_name);
    if (name == "upper") {
        return TriangleBranch::Upper;
    }
    if (name == "lower") {
        return TriangleBranch::Lower;
    }
    throw std::runtime_error("Unknown chunked branch: " + raw_name);
}

inline std::filesystem::path chunked_root_dir() {
    return std::filesystem::path(load_data_chunk_dir);
}

inline std::string chunked_range_token(size_t begin, size_t end) {
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << begin
        << "_" << std::setw(10) << std::setfill('0') << end;
    return out.str();
}

inline std::filesystem::path chunked_sample_path(ChunkedGreenTableKind kind,
                                                 TriangleBranch branch,
                                                 int node_begin,
                                                 int node_end) {
    return chunked_root_dir() / "samples" / chunked_table_kind_name(kind) /
           chunked_branch_name(branch) /
           ("nodes_" + chunked_range_token(static_cast<size_t>(node_begin), static_cast<size_t>(node_end)) + ".bin");
}

inline std::filesystem::path chunked_coeff_path(ChunkedGreenTableKind kind,
                                                TriangleBranch branch,
                                                size_t sitepair_begin,
                                                size_t sitepair_end) {
    return chunked_root_dir() / "coeffs" / chunked_table_kind_name(kind) /
           chunked_branch_name(branch) /
           ("sitepairs_" + chunked_range_token(sitepair_begin, sitepair_end) + ".bin");
}

inline std::filesystem::path chunked_corner_path(ChunkedGreenTableKind kind,
                                                 TriangleBranch branch) {
    return chunked_root_dir() / "corners" / chunked_table_kind_name(kind) /
           (std::string(chunked_branch_name(branch)) + ".bin");
}

inline std::filesystem::path chunked_manifest_path() {
    return chunked_root_dir() / "manifest.txt";
}

inline std::filesystem::path chunked_node_jobs_path() {
    return chunked_root_dir() / "node_jobs.tsv";
}

inline std::filesystem::path chunked_transform_jobs_path() {
    return chunked_root_dir() / "transform_jobs.tsv";
}

inline std::filesystem::path chunked_corner_jobs_path() {
    return chunked_root_dir() / "corner_jobs.tsv";
}

inline std::filesystem::path chunked_ready_path() {
    return chunked_root_dir() / "READY";
}

inline std::filesystem::path chunked_resource_plan_path() {
    return chunked_root_dir() / "resource_plan.txt";
}

inline std::filesystem::path chunked_diagonal_line_path(ChunkedGreenTableKind kind,
                                                        size_t sitepair_begin,
                                                        size_t sitepair_end) {
    return chunked_root_dir() / "diagonal_lines" / chunked_table_kind_name(kind) /
           ("sitepairs_" + chunked_range_token(sitepair_begin, sitepair_end) + ".bin");
}

inline std::filesystem::path chunked_diagonal_observable_path() {
    return chunked_root_dir() / "diagonal_observable.txt";
}

inline std::filesystem::path chunked_calculator_cache_dir() {
    return chunked_root_dir() / "calculator_cache";
}

inline std::filesystem::path chunked_calculator_cache_path(ChunkedGreenTableKind kind) {
    return chunked_calculator_cache_dir() / (std::string(chunked_table_kind_name(kind)) + ".bin");
}

inline std::string chunked_mp_string(const MP_Real& value) {
    return format_mp_real_compact(value);
}

inline std::string chunked_dd_string(const DoubleDouble& value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value.to_double();
    return out.str();
}

template <typename T>
inline void chunked_write_binary_value(std::ofstream& out, const T& value, const char* label) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error(std::string("Failed while writing ") + label);
    }
}

template <typename T>
inline T chunked_read_binary_value(std::ifstream& in, const char* label) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error(std::string("Failed while reading ") + label);
    }
    return value;
}

template <typename RealT>
inline void chunked_write_real_vector(std::ofstream& out,
                                      const std::vector<RealT>& values,
                                      const char* label) {
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    chunked_write_binary_value(out, size, label);
    for (const RealT& value : values) {
        chunked_write_binary_value(out, value, label);
    }
}

template <typename RealT>
inline std::vector<RealT> chunked_read_real_vector(std::ifstream& in, const char* label) {
    const std::uint64_t size = chunked_read_binary_value<std::uint64_t>(in, label);
    std::vector<RealT> values(static_cast<size_t>(size));
    for (RealT& value : values) {
        value = chunked_read_binary_value<RealT>(in, label);
    }
    return values;
}

template <typename RealT>
inline void chunked_write_real_matrix(std::ofstream& out,
                                      const Matrix<RealT>& matrix,
                                      const char* label) {
    const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
    const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
    chunked_write_binary_value(out, rows, label);
    chunked_write_binary_value(out, cols, label);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            chunked_write_binary_value(out, matrix(i, j), label);
        }
    }
}

template <typename RealT>
inline Matrix<RealT> chunked_read_real_matrix(std::ifstream& in, const char* label) {
    const std::int32_t rows = chunked_read_binary_value<std::int32_t>(in, label);
    const std::int32_t cols = chunked_read_binary_value<std::int32_t>(in, label);
    if (rows < 0 || cols < 0) {
        throw std::runtime_error(std::string("Invalid cached matrix shape for ") + label);
    }
    Matrix<RealT> matrix(rows, cols, RealT(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = chunked_read_binary_value<RealT>(in, label);
        }
    }
    return matrix;
}

template <typename RealT>
inline void chunked_write_complex_matrix(std::ofstream& out,
                                         const Matrix<ComplexNumber<RealT>>& matrix,
                                         const char* label) {
    const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
    const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
    chunked_write_binary_value(out, rows, label);
    chunked_write_binary_value(out, cols, label);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            const ComplexNumber<RealT>& value = matrix(i, j);
            chunked_write_binary_value(out, value.real, label);
            chunked_write_binary_value(out, value.imag, label);
        }
    }
}

template <typename RealT>
inline Matrix<ComplexNumber<RealT>> chunked_read_complex_matrix(std::ifstream& in, const char* label) {
    const std::int32_t rows = chunked_read_binary_value<std::int32_t>(in, label);
    const std::int32_t cols = chunked_read_binary_value<std::int32_t>(in, label);
    if (rows < 0 || cols < 0) {
        throw std::runtime_error(std::string("Invalid cached complex matrix shape for ") + label);
    }
    Matrix<ComplexNumber<RealT>> matrix(rows, cols, ComplexNumber<RealT>(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const RealT real = chunked_read_binary_value<RealT>(in, label);
            const RealT imag = chunked_read_binary_value<RealT>(in, label);
            matrix(i, j) = ComplexNumber<RealT>(real, imag);
        }
    }
    return matrix;
}

inline std::map<std::string, std::string> current_chunked_manifest_values(size_t node_chunk_size,
                                                                          size_t sitepair_chunk_size) {
    const size_t n_sites = static_cast<size_t>(N1) * static_cast<size_t>(N2);
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);

    return {
        {"layout_version", "1"},
        {"N1", std::to_string(N1)},
        {"N2", std::to_string(N2)},
        {"n_sites", std::to_string(n_sites)},
        {"replica_sitepair_count", std::to_string(sitepair_count)},
        {"CHEB_NU", std::to_string(cheb_nu)},
        {"CHEB_NV", std::to_string(cheb_nv)},
        {"node_count", std::to_string(node_count)},
        {"node_chunk_size", std::to_string(node_chunk_size)},
        {"sitepair_chunk_size", std::to_string(sitepair_chunk_size)},
        {"BETA", chunked_mp_string(beta)},
        {"MU", chunked_mp_string(mu)},
        {"ALPHA_UP", chunked_mp_string(alpha_shift_up)},
        {"ALPHA_DOWN", chunked_mp_string(alpha_shift_down)},
        {"PBC", pbc ? "true" : "false"},
        {"SUBREGION_MASK", subregion_mask_name},
        {"SUBREGION_SQUARE_RADIUS", std::to_string(subregion_square_radius)},
        {"LOAD_DATA_REAL", normalize_real_type_name(load_data_real_name)},
        {"VECTOR_REAL", normalize_real_type_name(vector_real_name)},
    };
}

inline std::map<std::string, std::string> read_key_value_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + path.string());
    }

    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        values[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return values;
}

inline void write_key_value_file(const std::filesystem::path& path,
                                 const std::map<std::string, std::string>& values) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path tmp_path = path.string() + ".tmp";
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + tmp_path.string());
        }
        for (const auto& [key, value] : values) {
            out << key << '=' << value << '\n';
        }
    }
    std::filesystem::rename(tmp_path, path);
}

inline std::vector<std::string> split_tsv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, '\t')) {
        tokens.push_back(token);
    }
    return tokens;
}

template <typename RealT>
inline constexpr bool chunked_binary_real_supported_v =
    std::is_same_v<RealT, double> || std::is_same_v<RealT, long double> ||
    std::is_same_v<RealT, DoubleDouble>;

template <typename RealT>
inline void chunked_write_values(std::ostream& out, const RealT* values, size_t count) {
    static_assert(chunked_binary_real_supported_v<RealT>, "Unsupported chunked binary real type");
    out.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count * sizeof(RealT)));
    if (!out) {
        throw std::runtime_error("Failed while writing chunked binary values");
    }
}

template <typename RealT>
inline void chunked_read_values(std::istream& in, RealT* values, size_t count) {
    static_assert(chunked_binary_real_supported_v<RealT>, "Unsupported chunked binary real type");
    in.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(count * sizeof(RealT)));
    if (!in) {
        throw std::runtime_error("Failed while reading chunked binary values");
    }
}

template <typename RealT>
inline void chunked_apply_green_table_drop_bits(std::vector<RealT>& values) {
    if (green_table_drop_bits <= 0) {
        return;
    }
    for (RealT& value : values) {
        value = truncate_real_mantissa_bits(value, green_table_drop_bits);
    }
}

inline size_t file_size_or_throw(const std::filesystem::path& path) {
    std::error_code ec;
    const size_t size = static_cast<size_t>(std::filesystem::file_size(path, ec));
    if (ec) {
        throw std::runtime_error("Could not stat " + path.string() + ": " + ec.message());
    }
    return size;
}

inline void require_file_size(const std::filesystem::path& path, size_t expected_size) {
    const size_t actual_size = file_size_or_throw(path);
    if (actual_size != expected_size) {
        std::ostringstream error;
        error << "Unexpected file size for " << path << ": expected "
              << expected_size << " bytes, got " << actual_size;
        throw std::runtime_error(error.str());
    }
}

inline void ensure_chunk_workflow_real_types_match() {
    if (normalize_real_type_name(load_data_real_name) != normalize_real_type_name(vector_real_name)) {
        throw std::runtime_error(
            "The first chunked load-data implementation requires LOAD_DATA_REAL == VECTOR_REAL. "
            "Use double/double, long_double/long_double, or double_double/double_double.");
    }
    if (parse_load_data_real_type(load_data_real_name) == LoadDataRealType::MP) {
        throw std::runtime_error("Chunked load-data does not write MP_Real binary samples; use double, long_double, or double_double.");
    }
}

template <typename RealT>
inline size_t auto_node_chunk_size_for_real(size_t node_count, size_t sitepair_count) {
    if (load_data_node_chunk_size > 0) {
        return static_cast<size_t>(load_data_node_chunk_size);
    }
    const long double target_bytes =
        std::max(0.01L, static_cast<long double>(load_data_sample_target_file_gb)) *
        1024.0L * 1024.0L * 1024.0L;
    const long double bytes_per_node =
        static_cast<long double>(sitepair_count) * static_cast<long double>(sizeof(RealT));
    const size_t candidate = static_cast<size_t>(std::max(1.0L, std::floor(target_bytes / bytes_per_node)));
    return std::max<size_t>(1, std::min(node_count, candidate));
}

template <typename RealT>
inline size_t auto_sitepair_chunk_size_for_real(size_t node_count, size_t sitepair_count) {
    if (load_data_sitepair_chunk_size > 0) {
        return static_cast<size_t>(load_data_sitepair_chunk_size);
    }
    const long double target_bytes =
        std::max(0.01L, static_cast<long double>(load_data_transform_target_mem_gb)) *
        1024.0L * 1024.0L * 1024.0L;
    const long double bytes_per_sitepair =
        static_cast<long double>(node_count) * static_cast<long double>(sizeof(RealT)) * 2.5L;
    const size_t candidate = static_cast<size_t>(std::max(1.0L, std::floor(target_bytes / bytes_per_sitepair)));
    return std::max<size_t>(1, std::min(sitepair_count, candidate));
}

inline std::vector<ChunkedNodeJob> read_chunked_node_jobs() {
    std::ifstream in(chunked_node_jobs_path());
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + chunked_node_jobs_path().string());
    }
    std::vector<ChunkedNodeJob> jobs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> tokens = split_tsv_line(line);
        if (tokens.size() < 5) {
            throw std::runtime_error("Malformed node job line: " + line);
        }
        jobs.push_back(ChunkedNodeJob{
            std::stoi(tokens[0]),
            parse_chunked_table_kind(tokens[1]),
            parse_chunked_branch(tokens[2]),
            std::stoi(tokens[3]),
            std::stoi(tokens[4]),
        });
    }
    return jobs;
}

inline std::vector<ChunkedSitepairJob> read_chunked_transform_jobs() {
    std::ifstream in(chunked_transform_jobs_path());
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + chunked_transform_jobs_path().string());
    }
    std::vector<ChunkedSitepairJob> jobs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> tokens = split_tsv_line(line);
        if (tokens.size() < 5) {
            throw std::runtime_error("Malformed transform job line: " + line);
        }
        jobs.push_back(ChunkedSitepairJob{
            std::stoi(tokens[0]),
            parse_chunked_table_kind(tokens[1]),
            parse_chunked_branch(tokens[2]),
            static_cast<size_t>(std::stoull(tokens[3])),
            static_cast<size_t>(std::stoull(tokens[4])),
        });
    }
    return jobs;
}

inline std::vector<ChunkedCornerJob> read_chunked_corner_jobs() {
    std::ifstream in(chunked_corner_jobs_path());
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + chunked_corner_jobs_path().string());
    }
    std::vector<ChunkedCornerJob> jobs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> tokens = split_tsv_line(line);
        if (tokens.size() < 3) {
            throw std::runtime_error("Malformed corner job line: " + line);
        }
        jobs.push_back(ChunkedCornerJob{
            std::stoi(tokens[0]),
            parse_chunked_table_kind(tokens[1]),
            parse_chunked_branch(tokens[2]),
        });
    }
    return jobs;
}

inline int chunked_job_index_from_environment() {
    const char* raw = std::getenv("CDET_LOAD_DATA_JOB_INDEX");
    if (raw == nullptr || raw[0] == '\0') {
        raw = std::getenv("PBS_ARRAY_INDEX");
    }
    if (raw == nullptr || raw[0] == '\0') {
        throw std::runtime_error(
            "Set CDET_LOAD_DATA_JOB_INDEX or run under a PBS array with PBS_ARRAY_INDEX for this chunk stage");
    }
    return std::stoi(raw);
}

template <typename JobT>
inline JobT select_chunked_job_by_index(const std::vector<JobT>& jobs, int job_index) {
    for (const JobT& job : jobs) {
        if (job.index == job_index) {
            return job;
        }
    }
    throw std::runtime_error("No chunked load-data job with index " + std::to_string(job_index));
}

inline MP_Real chunked_table_mu(ChunkedGreenTableKind kind) {
    switch (kind) {
        case ChunkedGreenTableKind::TwistedUp:
        case ChunkedGreenTableKind::OrdinaryUp:
            return mu - alpha_shift_up;
        case ChunkedGreenTableKind::TwistedDown:
        case ChunkedGreenTableKind::OrdinaryDown:
            return mu - alpha_shift_down;
    }
    throw std::runtime_error("Unhandled chunked table kind");
}

inline ReplicaSubregionMask chunked_table_mask(ChunkedGreenTableKind kind) {
    switch (kind) {
        case ChunkedGreenTableKind::TwistedUp:
        case ChunkedGreenTableKind::TwistedDown:
            return parse_subregion_mask(subregion_mask_name);
        case ChunkedGreenTableKind::OrdinaryUp:
        case ChunkedGreenTableKind::OrdinaryDown:
            return ReplicaSubregionMask::Empty;
    }
    throw std::runtime_error("Unhandled chunked table kind");
}

inline int chunked_table_origin_square_radius(ChunkedGreenTableKind kind) {
    const ReplicaSubregionMask mask = chunked_table_mask(kind);
    return mask == ReplicaSubregionMask::OriginSquare ? subregion_square_radius : 0;
}

template <typename RealT>
inline ReplicaGreenCalculator<RealT> make_chunked_calculator(ChunkedGreenTableKind kind) {
    return ReplicaGreenCalculator<RealT>(
        N1,
        N2,
        real_precision_cast<RealT>(beta),
        real_precision_cast<RealT>(chunked_table_mu(kind)),
        pbc,
        chunked_table_mask(kind),
        chunked_table_origin_square_radius(kind));
}

template <typename RealT>
inline void write_chunked_calculator_cache(ChunkedGreenTableKind kind,
                                           const ReplicaGreenCalculator<RealT>& calculator) {
    std::filesystem::create_directories(chunked_calculator_cache_dir());
    const std::filesystem::path output_path = chunked_calculator_cache_path(kind);
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Could not write " + tmp_path.string());
    }

    const char magic[] = "CDET_CALCULATOR_CACHE_V1";
    out.write(magic, sizeof(magic));
    if (!out) {
        throw std::runtime_error("Failed while writing calculator cache magic");
    }

    const std::int32_t n1_cached = static_cast<std::int32_t>(N1);
    const std::int32_t n2_cached = static_cast<std::int32_t>(N2);
    const std::int32_t pbc_cached = pbc ? 1 : 0;
    const std::int32_t mask_cached = static_cast<std::int32_t>(chunked_table_mask(kind));
    const std::int32_t radius_cached = static_cast<std::int32_t>(chunked_table_origin_square_radius(kind));
    const std::int32_t real_bytes = static_cast<std::int32_t>(sizeof(RealT));
    chunked_write_binary_value(out, n1_cached, "calculator cache N1");
    chunked_write_binary_value(out, n2_cached, "calculator cache N2");
    chunked_write_binary_value(out, pbc_cached, "calculator cache PBC");
    chunked_write_binary_value(out, mask_cached, "calculator cache mask");
    chunked_write_binary_value(out, radius_cached, "calculator cache radius");
    chunked_write_binary_value(out, real_bytes, "calculator cache real size");

    const RealT beta_cached = real_precision_cast<RealT>(beta);
    const RealT mu_cached = real_precision_cast<RealT>(chunked_table_mu(kind));
    chunked_write_binary_value(out, beta_cached, "calculator cache beta");
    chunked_write_binary_value(out, mu_cached, "calculator cache mu");
    chunked_write_real_vector(out, calculator.eigenvalues(), "calculator cache eigenvalues");
    chunked_write_real_matrix(out, calculator.eigenvectors(), "calculator cache eigenvectors");
    chunked_write_complex_matrix(out, calculator.sector_mixing_matrix(0), "calculator cache sector 0");
    chunked_write_complex_matrix(out, calculator.sector_mixing_matrix(1), "calculator cache sector 1");
    out.close();
    std::filesystem::rename(tmp_path, output_path);
}

template <typename RealT>
inline ReplicaGreenCalculator<RealT> read_chunked_calculator_cache(ChunkedGreenTableKind kind) {
    const std::filesystem::path path = chunked_calculator_cache_path(kind);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open calculator cache " + path.string());
    }

    char magic[sizeof("CDET_CALCULATOR_CACHE_V1")] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic) != "CDET_CALCULATOR_CACHE_V1") {
        throw std::runtime_error("Invalid calculator cache magic in " + path.string());
    }

    const std::int32_t n1_cached = chunked_read_binary_value<std::int32_t>(in, "calculator cache N1");
    const std::int32_t n2_cached = chunked_read_binary_value<std::int32_t>(in, "calculator cache N2");
    const std::int32_t pbc_cached = chunked_read_binary_value<std::int32_t>(in, "calculator cache PBC");
    const std::int32_t mask_cached = chunked_read_binary_value<std::int32_t>(in, "calculator cache mask");
    const std::int32_t radius_cached = chunked_read_binary_value<std::int32_t>(in, "calculator cache radius");
    const std::int32_t real_bytes = chunked_read_binary_value<std::int32_t>(in, "calculator cache real size");
    const RealT beta_cached = chunked_read_binary_value<RealT>(in, "calculator cache beta");
    const RealT mu_cached = chunked_read_binary_value<RealT>(in, "calculator cache mu");

    const ReplicaSubregionMask expected_mask = chunked_table_mask(kind);
    const int expected_radius = chunked_table_origin_square_radius(kind);
    const RealT expected_beta = real_precision_cast<RealT>(beta);
    const RealT expected_mu = real_precision_cast<RealT>(chunked_table_mu(kind));
    if (n1_cached != N1 || n2_cached != N2 || pbc_cached != (pbc ? 1 : 0) ||
        mask_cached != static_cast<std::int32_t>(expected_mask) || radius_cached != expected_radius ||
        real_bytes != static_cast<std::int32_t>(sizeof(RealT)) ||
        beta_cached != expected_beta || mu_cached != expected_mu) {
        throw std::runtime_error("Calculator cache does not match current parameters: " + path.string());
    }

    RealVector<RealT> eigenvalues = chunked_read_real_vector<RealT>(in, "calculator cache eigenvalues");
    Matrix<RealT> eigenvectors = chunked_read_real_matrix<RealT>(in, "calculator cache eigenvectors");
    std::array<Matrix<ComplexNumber<RealT>>, 2> sector_mixing = {
        chunked_read_complex_matrix<RealT>(in, "calculator cache sector 0"),
        chunked_read_complex_matrix<RealT>(in, "calculator cache sector 1"),
    };

    return ReplicaGreenCalculator<RealT>(
        N1,
        N2,
        expected_beta,
        expected_mu,
        pbc,
        expected_mask,
        expected_radius,
        std::move(eigenvalues),
        std::move(eigenvectors),
        std::move(sector_mixing));
}

template <typename RealT>
inline ReplicaGreenCalculator<RealT> make_chunked_calculator_with_optional_cache(ChunkedGreenTableKind kind) {
    const std::filesystem::path cache_path = chunked_calculator_cache_path(kind);
    if (load_data_calculator_cache != "off" && std::filesystem::exists(cache_path)) {
        std::cout << "Loading calculator cache: " << cache_path << '\n';
        return read_chunked_calculator_cache<RealT>(kind);
    }
    if (load_data_calculator_cache == "require") {
        throw std::runtime_error("Missing required calculator cache: " + cache_path.string());
    }
    std::cout << "Calculator cache not used for " << chunked_table_kind_name(kind)
              << "; building calculator in this job.\n";
    return make_chunked_calculator<RealT>(kind);
}

inline std::array<int, 6> chunked_decode_sitepair(size_t flat_index) {
    return decode_replica_sitepair_index(flat_index, N1, N2);
}

inline bool chunked_sitepair_is_diagonal_lower_line(size_t flat_index) {
    const auto decoded = chunked_decode_sitepair(flat_index);
    return decoded[0] == decoded[1] && decoded[2] == decoded[4] && decoded[3] == decoded[5];
}

template <typename RealT>
inline RealT chunked_replica_entry_for_flat_index(const Matrix<ComplexNumber<RealT>>& sector_m0,
                                                  const Matrix<ComplexNumber<RealT>>& sector_m1,
                                                  size_t flat_index) {
    const auto decoded = chunked_decode_sitepair(flat_index);
    const int site1 = site_index(decoded[2], decoded[3], N1, N2);
    const int site2 = site_index(decoded[4], decoded[5], N1, N2);
    return replica_entry_from_sector_matrices(
        sector_m0, sector_m1, decoded[0], decoded[1], site1, site2);
}

template <typename RealT>
inline void run_chunked_plan_for_real() {
    ensure_chunk_workflow_real_types_match();

    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t node_chunk_size = auto_node_chunk_size_for_real<RealT>(node_count, sitepair_count);
    const size_t sitepair_chunk_size = auto_sitepair_chunk_size_for_real<RealT>(node_count, sitepair_count);
    const size_t node_job_count_per_table_branch =
        (node_count + node_chunk_size - 1) / node_chunk_size;

    std::filesystem::create_directories(chunked_root_dir());
    write_key_value_file(
        chunked_manifest_path(),
        current_chunked_manifest_values(node_chunk_size, sitepair_chunk_size));

    {
        std::ofstream out(chunked_node_jobs_path());
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + chunked_node_jobs_path().string());
        }
        out << "#index\ttable\tbranch\tnode_begin\tnode_end\n";
        int job_index = 1;
        for (ChunkedGreenTableKind kind : chunked_table_kinds()) {
            for (TriangleBranch branch : chunked_branches()) {
                for (size_t chunk = 0; chunk < node_job_count_per_table_branch; ++chunk) {
                    const size_t begin = chunk * node_chunk_size;
                    const size_t end = std::min(node_count, begin + node_chunk_size);
                    out << job_index++ << '\t'
                        << chunked_table_kind_name(kind) << '\t'
                        << chunked_branch_name(branch) << '\t'
                        << begin << '\t' << end << '\n';
                }
            }
        }
    }

    {
        std::ofstream out(chunked_transform_jobs_path());
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + chunked_transform_jobs_path().string());
        }
        out << "#index\ttable\tbranch\tsitepair_begin\tsitepair_end\n";
        int job_index = 1;
        for (ChunkedGreenTableKind kind : chunked_table_kinds()) {
            for (TriangleBranch branch : chunked_branches()) {
                for (size_t begin = 0; begin < sitepair_count; begin += sitepair_chunk_size) {
                    const size_t end = std::min(sitepair_count, begin + sitepair_chunk_size);
                    out << job_index++ << '\t'
                        << chunked_table_kind_name(kind) << '\t'
                        << chunked_branch_name(branch) << '\t'
                        << begin << '\t' << end << '\n';
                }
            }
        }
    }

    {
        std::ofstream out(chunked_corner_jobs_path());
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + chunked_corner_jobs_path().string());
        }
        out << "#index\ttable\tbranch\n";
        int job_index = 1;
        for (ChunkedGreenTableKind kind : chunked_table_kinds()) {
            for (TriangleBranch branch : chunked_branches()) {
                out << job_index++ << '\t'
                    << chunked_table_kind_name(kind) << '\t'
                    << chunked_branch_name(branch) << '\n';
            }
        }
    }

    const long double gib = 1024.0L * 1024.0L * 1024.0L;
    const long double node_sample_file_gib =
        static_cast<long double>(node_chunk_size) *
        static_cast<long double>(sitepair_count) *
        static_cast<long double>(sizeof(RealT)) / gib;
    const long double transform_mem_gib =
        static_cast<long double>(sitepair_chunk_size) *
        static_cast<long double>(node_count) *
        static_cast<long double>(sizeof(RealT)) * 2.5L / gib;
    const size_t n_sites = static_cast<size_t>(N1) * static_cast<size_t>(N2);
    const long double sample_mem_gib =
        static_cast<long double>(16) *
        static_cast<long double>(n_sites) *
        static_cast<long double>(n_sites) *
        static_cast<long double>(sizeof(ComplexNumber<RealT>)) / gib;
    const long double resident_gib =
        (static_cast<long double>(4) * static_cast<long double>(2) *
             static_cast<long double>(sitepair_count) * static_cast<long double>(node_count) *
             static_cast<long double>(sizeof(RealT)) +
         static_cast<long double>(4) * static_cast<long double>(2) *
             static_cast<long double>(sitepair_count) * static_cast<long double>(sizeof(RealT))) /
        gib;

    {
        std::ofstream out(chunked_resource_plan_path());
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + chunked_resource_plan_path().string());
        }
        out << "real_type=" << real_type_name<RealT>() << '\n';
        out << "node_count=" << node_count << '\n';
        out << "replica_sitepair_count=" << sitepair_count << '\n';
        out << "node_chunk_size=" << node_chunk_size << '\n';
        out << "sitepair_chunk_size=" << sitepair_chunk_size << '\n';
        out << "node_jobs=" << read_chunked_node_jobs().size() << '\n';
        out << "transform_jobs=" << read_chunked_transform_jobs().size() << '\n';
        out << "corner_jobs=" << read_chunked_corner_jobs().size() << '\n';
        out << "sample_chunk_file_gib_estimate=" << static_cast<double>(node_sample_file_gib) << '\n';
        out << "sample_subjob_mem_gib_estimate=" << static_cast<double>(std::max(1.0L, sample_mem_gib * 1.5L)) << '\n';
        out << "transform_subjob_mem_gib_estimate=" << static_cast<double>(std::max(1.0L, transform_mem_gib * 1.5L)) << '\n';
        out << "mcmcmc_resident_table_gib_estimate=" << static_cast<double>(resident_gib) << '\n';
        out << "mcmcmc_subjob_mem_gib_recommendation=" << static_cast<double>(resident_gib * 1.35L + 8.0L) << '\n';
    }

    std::cout << "Chunked load-data plan written to " << chunked_root_dir() << '\n';
    std::cout << "  node jobs = " << read_chunked_node_jobs().size()
              << ", transform jobs = " << read_chunked_transform_jobs().size()
              << ", corner jobs = " << read_chunked_corner_jobs().size() << '\n';
    std::cout << "  node chunk size = " << node_chunk_size
              << ", sitepair chunk size = " << sitepair_chunk_size << '\n';
    std::cout << "  resource estimates: sample ~= "
              << static_cast<double>(std::max(1.0L, sample_mem_gib * 1.5L))
              << " GiB/subjob, transform ~= "
              << static_cast<double>(std::max(1.0L, transform_mem_gib * 1.5L))
              << " GiB/subjob, resident MCMCMC ~= "
              << static_cast<double>(resident_gib * 1.35L + 8.0L) << " GiB\n";
}

template <typename RealT>
inline void ensure_manifest_matches_for_real() {
    const auto manifest = read_key_value_file(chunked_manifest_path());
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_chunk_size = manifest.count("node_chunk_size") != 0
                                       ? static_cast<size_t>(std::stoull(manifest.at("node_chunk_size")))
                                       : auto_node_chunk_size_for_real<RealT>(node_count, sitepair_count);
    const size_t sitepair_chunk_size = manifest.count("sitepair_chunk_size") != 0
                                           ? static_cast<size_t>(std::stoull(manifest.at("sitepair_chunk_size")))
                                           : auto_sitepair_chunk_size_for_real<RealT>(node_count, sitepair_count);
    const auto expected = current_chunked_manifest_values(node_chunk_size, sitepair_chunk_size);
    for (const auto& [key, expected_value] : expected) {
        const auto found = manifest.find(key);
        if (found == manifest.end() || found->second != expected_value) {
            std::ostringstream error;
            error << "Chunk manifest mismatch for " << key << ": expected "
                  << expected_value << ", found "
                  << (found == manifest.end() ? "(missing)" : found->second);
            throw std::runtime_error(error.str());
        }
    }
}

template <typename RealT>
inline void run_chunked_precompute_calculators_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    std::filesystem::create_directories(chunked_calculator_cache_dir());

    const int requested_job_index = []() {
        const char* raw = std::getenv("CDET_LOAD_DATA_JOB_INDEX");
        if (raw == nullptr || raw[0] == '\0') {
            raw = std::getenv("PBS_ARRAY_INDEX");
        }
        return (raw == nullptr || raw[0] == '\0') ? 0 : std::stoi(raw);
    }();
    if (requested_job_index < 0 || requested_job_index > 4) {
        throw std::runtime_error("CDET_LOAD_DATA_JOB_INDEX for precompute_calculators must be 1..4");
    }

    int index = 1;
    for (ChunkedGreenTableKind kind : chunked_table_kinds()) {
        if (requested_job_index != 0 && requested_job_index != index) {
            ++index;
            continue;
        }
        const std::filesystem::path cache_path = chunked_calculator_cache_path(kind);
        if (std::filesystem::exists(cache_path)) {
            std::cout << "Calculator cache already exists for " << chunked_table_kind_name(kind)
                      << ": " << cache_path << '\n';
            ++index;
            continue;
        }

        std::cout << "Precomputing calculator cache " << index << ": "
                  << chunked_table_kind_name(kind) << " -> " << cache_path << '\n';
        const ReplicaGreenCalculator<RealT> calculator = make_chunked_calculator<RealT>(kind);
        write_chunked_calculator_cache(kind, calculator);
        std::cout << "Wrote calculator cache: " << cache_path << '\n';
        ++index;
    }
}

template <typename RealT>
inline void run_chunked_sample_node_job_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    const int job_index = chunked_job_index_from_environment();
    const ChunkedNodeJob job = select_chunked_job_by_index(read_chunked_node_jobs(), job_index);
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t io_block_size = std::min<size_t>(sitepair_count, 65536);

    std::filesystem::path output_path =
        chunked_sample_path(job.kind, job.branch, job.node_begin, job.node_end);
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Could not write " + tmp_path.string());
    }

    ReplicaGreenCalculator<RealT> calculator = make_chunked_calculator_with_optional_cache<RealT>(job.kind);
    const ChebyshevNodeTables<RealT> tables = build_chebyshev_node_tables<RealT>(cheb_nu, cheb_nv);
    std::vector<RealT> buffer(io_block_size);

    std::cout << "Chunk sample job " << job.index << ": "
              << chunked_table_kind_name(job.kind) << ' '
              << chunked_branch_name(job.branch) << " nodes ["
              << job.node_begin << ", " << job.node_end << ")\n";

    for (int node = job.node_begin; node < job.node_end; ++node) {
        const int u_index = node / cheb_nv;
        const int v_index = node - u_index * cheb_nv;
        const auto [tau, tau_prime] =
            map_square_to_triangle(tables.cheb_node_u[static_cast<size_t>(u_index)],
                                   tables.cheb_node_v[static_cast<size_t>(v_index)],
                                   job.branch,
                                   calculator.beta());
        const auto sector_matrices = calculator.green_sector_matrices(tau, tau_prime, job.branch);
        for (size_t begin = 0; begin < sitepair_count; begin += io_block_size) {
            const size_t end = std::min(sitepair_count, begin + io_block_size);
            for (size_t flat = begin; flat < end; ++flat) {
                buffer[flat - begin] = chunked_replica_entry_for_flat_index(
                    sector_matrices[0], sector_matrices[1], flat);
            }
            chunked_write_values(out, buffer.data(), end - begin);
        }
    }
    out.close();
    std::filesystem::rename(tmp_path, output_path);
    const size_t expected_size =
        static_cast<size_t>(job.node_end - job.node_begin) * sitepair_count * sizeof(RealT);
    require_file_size(output_path, expected_size);
}

template <typename RealT>
inline void run_chunked_corner_job_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    const int job_index = chunked_job_index_from_environment();
    const ChunkedCornerJob job = select_chunked_job_by_index(read_chunked_corner_jobs(), job_index);
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t io_block_size = std::min<size_t>(sitepair_count, 65536);

    std::filesystem::path output_path = chunked_corner_path(job.kind, job.branch);
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Could not write " + tmp_path.string());
    }

    ReplicaGreenCalculator<RealT> calculator = make_chunked_calculator_with_optional_cache<RealT>(job.kind);
    const auto sector_matrices = calculator.green_sector_matrices(RealT(0), RealT(0), job.branch);
    std::vector<RealT> buffer(io_block_size);

    std::cout << "Chunk corner job " << job.index << ": "
              << chunked_table_kind_name(job.kind) << ' '
              << chunked_branch_name(job.branch) << '\n';

    for (size_t begin = 0; begin < sitepair_count; begin += io_block_size) {
        const size_t end = std::min(sitepair_count, begin + io_block_size);
        for (size_t flat = begin; flat < end; ++flat) {
            buffer[flat - begin] = chunked_replica_entry_for_flat_index(
                sector_matrices[0], sector_matrices[1], flat);
        }
        chunked_apply_green_table_drop_bits(buffer);
        chunked_write_values(out, buffer.data(), end - begin);
    }
    out.close();
    std::filesystem::rename(tmp_path, output_path);
    require_file_size(output_path, sitepair_count * sizeof(RealT));
}

template <typename RealT>
inline void write_chunked_diagonal_line_file(const ChunkedSitepairJob& job,
                                             const std::vector<RealT>& coeff_grid) {
    if (job.branch != TriangleBranch::Lower) {
        return;
    }
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t chunk_sitepair_count = job.sitepair_end - job.sitepair_begin;
    std::vector<size_t> diagonal_local_indices;
    diagonal_local_indices.reserve(
        std::min(chunk_sitepair_count, static_cast<size_t>(2) * static_cast<size_t>(N1) * static_cast<size_t>(N2)));
    for (size_t local_sitepair = 0; local_sitepair < chunk_sitepair_count; ++local_sitepair) {
        const size_t sitepair_index = job.sitepair_begin + local_sitepair;
        if (chunked_sitepair_is_diagonal_lower_line(sitepair_index)) {
            diagonal_local_indices.push_back(local_sitepair);
        }
    }

    const std::filesystem::path output_path =
        chunked_diagonal_line_path(job.kind, job.sitepair_begin, job.sitepair_end);
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + tmp_path.string());
        }
        const std::uint64_t record_count = static_cast<std::uint64_t>(diagonal_local_indices.size());
        out.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));
        if (!out) {
            throw std::runtime_error("Failed while writing diagonal line record count");
        }
        std::vector<RealT> line(static_cast<size_t>(cheb_nu), RealT(0));
        for (size_t local_sitepair : diagonal_local_indices) {
            const std::uint64_t sitepair_index =
                static_cast<std::uint64_t>(job.sitepair_begin + local_sitepair);
            out.write(reinterpret_cast<const char*>(&sitepair_index), sizeof(sitepair_index));
            if (!out) {
                throw std::runtime_error("Failed while writing diagonal line sitepair index");
            }
            const RealT* coeffs = coeff_grid.data() + local_sitepair * node_count;
            for (int u_index = 0; u_index < cheb_nu; ++u_index) {
                RealT sum(0);
                const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
                for (int v_index = 0; v_index < cheb_nv; ++v_index) {
                    sum += coeffs[row_offset + static_cast<size_t>(v_index)];
                }
                line[static_cast<size_t>(u_index)] = sum;
            }
            chunked_write_values(out, line.data(), line.size());
        }
    }
    std::filesystem::rename(tmp_path, output_path);
    const size_t expected_size =
        sizeof(std::uint64_t) +
        diagonal_local_indices.size() * (sizeof(std::uint64_t) + static_cast<size_t>(cheb_nu) * sizeof(RealT));
    require_file_size(output_path, expected_size);
}

template <typename RealT>
inline void run_chunked_transform_sitepair_job_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    const int job_index = chunked_job_index_from_environment();
    const ChunkedSitepairJob job = select_chunked_job_by_index(read_chunked_transform_jobs(), job_index);
    const std::vector<ChunkedNodeJob> node_jobs = read_chunked_node_jobs();
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t chunk_sitepair_count = job.sitepair_end - job.sitepair_begin;
    const ChebyshevNodeTables<RealT> tables = build_chebyshev_node_tables<RealT>(cheb_nu, cheb_nv);

    std::vector<RealT> sample_grid(chunk_sitepair_count * node_count, RealT(0));
    std::vector<RealT> read_buffer(chunk_sitepair_count);

    std::cout << "Chunk transform job " << job.index << ": "
              << chunked_table_kind_name(job.kind) << ' '
              << chunked_branch_name(job.branch) << " sitepairs ["
              << job.sitepair_begin << ", " << job.sitepair_end << ")\n";

    for (const ChunkedNodeJob& node_job : node_jobs) {
        if (node_job.kind != job.kind || node_job.branch != job.branch) {
            continue;
        }
        const std::filesystem::path sample_path =
            chunked_sample_path(node_job.kind, node_job.branch, node_job.node_begin, node_job.node_end);
        std::ifstream in(sample_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open " + sample_path.string());
        }
        const int local_node_count = node_job.node_end - node_job.node_begin;
        require_file_size(sample_path, static_cast<size_t>(local_node_count) * sitepair_count * sizeof(RealT));
        for (int local_node = 0; local_node < local_node_count; ++local_node) {
            const size_t global_node = static_cast<size_t>(node_job.node_begin + local_node);
            const size_t byte_offset =
                (static_cast<size_t>(local_node) * sitepair_count + job.sitepair_begin) * sizeof(RealT);
            in.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
            chunked_read_values(in, read_buffer.data(), chunk_sitepair_count);
            for (size_t local_sitepair = 0; local_sitepair < chunk_sitepair_count; ++local_sitepair) {
                sample_grid[local_sitepair * node_count + global_node] = read_buffer[local_sitepair];
            }
        }
    }

    for (size_t local_sitepair = 0; local_sitepair < chunk_sitepair_count; ++local_sitepair) {
        RealT* block = sample_grid.data() + local_sitepair * node_count;
        transform_sitepair_sample_grid_to_coeffs(block, tables, cheb_nu, cheb_nv);
    }
    chunked_apply_green_table_drop_bits(sample_grid);

    const std::filesystem::path output_path =
        chunked_coeff_path(job.kind, job.branch, job.sitepair_begin, job.sitepair_end);
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + tmp_path.string());
        }
        chunked_write_values(out, sample_grid.data(), sample_grid.size());
    }
    std::filesystem::rename(tmp_path, output_path);
    require_file_size(output_path, chunk_sitepair_count * node_count * sizeof(RealT));
    write_chunked_diagonal_line_file(job, sample_grid);
}

template <typename RealT>
inline std::uint64_t chunked_diagonal_line_record_count(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::uint64_t record_count = 0;
    in.read(reinterpret_cast<char*>(&record_count), sizeof(record_count));
    if (!in) {
        throw std::runtime_error("Could not read diagonal line record count from " + path.string());
    }
    const size_t expected_size =
        sizeof(std::uint64_t) +
        static_cast<size_t>(record_count) *
            (sizeof(std::uint64_t) + static_cast<size_t>(cheb_nu) * sizeof(RealT));
    require_file_size(path, expected_size);
    return record_count;
}

template <typename RealT>
inline std::map<size_t, std::vector<RealT>> read_chunked_diagonal_lines_for_kind(
    ChunkedGreenTableKind kind) {
    std::map<size_t, std::vector<RealT>> lines;
    for (const ChunkedSitepairJob& job : read_chunked_transform_jobs()) {
        if (job.kind != kind || job.branch != TriangleBranch::Lower) {
            continue;
        }
        const std::filesystem::path line_path =
            chunked_diagonal_line_path(kind, job.sitepair_begin, job.sitepair_end);
        const std::uint64_t record_count = chunked_diagonal_line_record_count<RealT>(line_path);
        std::ifstream in(line_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open " + line_path.string());
        }
        std::uint64_t stored_record_count = 0;
        in.read(reinterpret_cast<char*>(&stored_record_count), sizeof(stored_record_count));
        if (stored_record_count != record_count) {
            throw std::runtime_error("Diagonal line record count changed while reading " + line_path.string());
        }
        for (std::uint64_t record = 0; record < record_count; ++record) {
            std::uint64_t sitepair_index = 0;
            in.read(reinterpret_cast<char*>(&sitepair_index), sizeof(sitepair_index));
            if (!in) {
                throw std::runtime_error("Could not read diagonal line sitepair index from " + line_path.string());
            }
            std::vector<RealT> line(static_cast<size_t>(cheb_nu), RealT(0));
            chunked_read_values(in, line.data(), line.size());
            const auto [_, inserted] = lines.emplace(static_cast<size_t>(sitepair_index), std::move(line));
            if (!inserted) {
                throw std::runtime_error("Duplicate diagonal sitepair in " + line_path.string());
            }
        }
    }
    return lines;
}

template <typename RealT>
inline const std::vector<RealT>& chunked_require_diagonal_line(
    const std::map<size_t, std::vector<RealT>>& lines,
    size_t sitepair_index,
    const char* table_name) {
    const auto found = lines.find(sitepair_index);
    if (found == lines.end()) {
        throw std::runtime_error(
            std::string("Missing diagonal line for ") + table_name +
            " sitepair " + std::to_string(sitepair_index));
    }
    return found->second;
}

template <typename RealT>
inline RealT integrate_diagonal_shifted_product_from_lines(const std::vector<RealT>& up_line,
                                                           const std::vector<RealT>& down_line,
                                                           const RealT& beta_t,
                                                           const RealT& diagonal_up_shift,
                                                           const RealT& diagonal_down_shift) {
    const RealT up_integral = integrate_chebyshev_line(up_line, beta_t);
    const RealT down_integral = integrate_chebyshev_line(down_line, beta_t);
    const RealT product_integral = integrate_chebyshev_line_product(up_line, down_line, beta_t);
    return product_integral +
           diagonal_up_shift * down_integral +
           diagonal_down_shift * up_integral;
}

template <typename RealT>
inline DoubleDouble compute_chunked_diagonal_observable_from_lines_for_real() {
    const auto twisted_up = read_chunked_diagonal_lines_for_kind<RealT>(ChunkedGreenTableKind::TwistedUp);
    const auto twisted_down = read_chunked_diagonal_lines_for_kind<RealT>(ChunkedGreenTableKind::TwistedDown);
    const auto ordinary_up = read_chunked_diagonal_lines_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryUp);
    const auto ordinary_down = read_chunked_diagonal_lines_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryDown);

    if (U == MP_Real(0)) {
        throw std::runtime_error("The diagonal observable is undefined for U = 0");
    }

    const RealT beta_t = real_precision_cast<RealT>(beta);
    const RealT u_t = real_precision_cast<RealT>(U);
    const RealT diagonal_up_shift = real_precision_cast<RealT>(alpha_shift_down) / u_t;
    const RealT diagonal_down_shift = real_precision_cast<RealT>(alpha_shift_up) / u_t;

    RealT twisted_sum(0);
    for (int replica_index = 0; replica_index < 2; ++replica_index) {
        for (int site_b2 = 0; site_b2 < N2; ++site_b2) {
            for (int site_b1 = 0; site_b1 < N1; ++site_b1) {
                const size_t sitepair_index =
                    replica_sitepair_index(replica_index,
                                           replica_index,
                                           site_b1,
                                           site_b2,
                                           site_b1,
                                           site_b2,
                                           N1,
                                           N2);
                twisted_sum += integrate_diagonal_shifted_product_from_lines(
                    chunked_require_diagonal_line<RealT>(twisted_up, sitepair_index, "twisted_up"),
                    chunked_require_diagonal_line<RealT>(twisted_down, sitepair_index, "twisted_down"),
                    beta_t,
                    diagonal_up_shift,
                    diagonal_down_shift);
            }
        }
    }

    RealT ordinary_sum(0);
    for (int site_b2 = 0; site_b2 < N2; ++site_b2) {
        for (int site_b1 = 0; site_b1 < N1; ++site_b1) {
            const size_t sitepair_index =
                replica_sitepair_index(0, 0, site_b1, site_b2, site_b1, site_b2, N1, N2);
            ordinary_sum += integrate_diagonal_shifted_product_from_lines(
                chunked_require_diagonal_line<RealT>(ordinary_up, sitepair_index, "ordinary_up"),
                chunked_require_diagonal_line<RealT>(ordinary_down, sitepair_index, "ordinary_down"),
                beta_t,
                diagonal_up_shift,
                diagonal_down_shift);
        }
    }

    const RealT site_count = RealT(N1) * RealT(N2);
    const RealT twisted = -u_t * twisted_sum / site_count;
    const RealT ordinary = -u_t * ordinary_sum / site_count;
    return real_precision_cast<DoubleDouble>(twisted - RealT(2) * ordinary);
}

inline void write_chunked_diagonal_observable_value(const DoubleDouble& value) {
    const std::filesystem::path output_path = chunked_diagonal_observable_path();
    std::filesystem::create_directories(output_path.parent_path());
    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + tmp_path.string());
        }
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "layout_version=1\n";
        out << "order1_hi=" << value.hi << '\n';
        out << "order1_lo=" << value.lo << '\n';
    }
    std::filesystem::rename(tmp_path, output_path);
}

inline DoubleDouble read_chunked_diagonal_observable_value() {
    const std::map<std::string, std::string> values =
        read_key_value_file(chunked_diagonal_observable_path());
    const auto hi_it = values.find("order1_hi");
    const auto lo_it = values.find("order1_lo");
    if (hi_it == values.end() || lo_it == values.end()) {
        throw std::runtime_error("Malformed " + chunked_diagonal_observable_path().string());
    }
    return DoubleDouble(std::stod(hi_it->second), std::stod(lo_it->second));
}

template <typename RealT>
inline void run_chunked_diagonal_reduce_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    const DoubleDouble value = compute_chunked_diagonal_observable_from_lines_for_real<RealT>();
    write_chunked_diagonal_observable_value(value);
    std::cout << "Chunked diagonal order-1 observable written to "
              << chunked_diagonal_observable_path()
              << ": " << chunked_dd_string(value) << '\n';
}

template <typename RealT>
inline void validate_chunked_outputs_for_real() {
    ensure_manifest_matches_for_real<RealT>();
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    for (const ChunkedNodeJob& job : read_chunked_node_jobs()) {
        require_file_size(chunked_sample_path(job.kind, job.branch, job.node_begin, job.node_end),
                          static_cast<size_t>(job.node_end - job.node_begin) * sitepair_count * sizeof(RealT));
    }
    for (const ChunkedSitepairJob& job : read_chunked_transform_jobs()) {
        require_file_size(chunked_coeff_path(job.kind, job.branch, job.sitepair_begin, job.sitepair_end),
                          (job.sitepair_end - job.sitepair_begin) * node_count * sizeof(RealT));
        if (job.branch == TriangleBranch::Lower) {
            (void)chunked_diagonal_line_record_count<RealT>(
                chunked_diagonal_line_path(job.kind, job.sitepair_begin, job.sitepair_end));
        }
    }
    for (const ChunkedCornerJob& job : read_chunked_corner_jobs()) {
        require_file_size(chunked_corner_path(job.kind, job.branch), sitepair_count * sizeof(RealT));
    }
    (void)read_chunked_diagonal_observable_value();
    {
        std::ofstream out(chunked_ready_path());
        if (!out.is_open()) {
            throw std::runtime_error("Could not write " + chunked_ready_path().string());
        }
        out << "ready\n";
    }
    std::cout << "Chunked load-data validation passed; wrote " << chunked_ready_path() << '\n';
}

template <typename RealT>
inline void load_coeff_chunks_for_branch(std::vector<RealT>& destination,
                                         ChunkedGreenTableKind kind,
                                         TriangleBranch branch) {
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    for (const ChunkedSitepairJob& job : read_chunked_transform_jobs()) {
        if (job.kind != kind || job.branch != branch) {
            continue;
        }
        const size_t chunk_sitepair_count = job.sitepair_end - job.sitepair_begin;
        const std::filesystem::path coeff_path =
            chunked_coeff_path(kind, branch, job.sitepair_begin, job.sitepair_end);
        require_file_size(coeff_path, chunk_sitepair_count * node_count * sizeof(RealT));
        std::ifstream in(coeff_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open " + coeff_path.string());
        }
        RealT* target = destination.data() + job.sitepair_begin * node_count;
        chunked_read_values(in, target, chunk_sitepair_count * node_count);
    }
}

template <typename RealT>
inline void load_corner_chunk_for_branch(std::vector<RealT>& destination,
                                         ChunkedGreenTableKind kind,
                                         TriangleBranch branch) {
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const std::filesystem::path corner_path = chunked_corner_path(kind, branch);
    require_file_size(corner_path, sitepair_count * sizeof(RealT));
    std::ifstream in(corner_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + corner_path.string());
    }
    chunked_read_values(in, destination.data(), sitepair_count);
}

template <typename RealT>
inline PiecewiseChebyshevTable<RealT> load_chunked_table_for_kind(ChunkedGreenTableKind kind) {
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    PiecewiseChebyshevTable<RealT> table;
    table.upper_coeffs.assign(sitepair_count * node_count, RealT(0));
    table.lower_coeffs.assign(sitepair_count * node_count, RealT(0));
    table.upper_corners.assign(sitepair_count, RealT(0));
    table.lower_corners.assign(sitepair_count, RealT(0));
    load_coeff_chunks_for_branch(table.upper_coeffs, kind, TriangleBranch::Upper);
    load_coeff_chunks_for_branch(table.lower_coeffs, kind, TriangleBranch::Lower);
    load_corner_chunk_for_branch(table.upper_corners, kind, TriangleBranch::Upper);
    load_corner_chunk_for_branch(table.lower_corners, kind, TriangleBranch::Lower);
    return table;
}

template <typename RealT>
inline DiskBackedPiecewiseChebyshevTable<RealT> open_disk_backed_table_for_kind(
    ChunkedGreenTableKind kind,
    const std::shared_ptr<DiskBackedGreenCoeffCache<RealT>>& cache) {
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    DiskBackedPiecewiseChebyshevTable<RealT> table;
    table.table_kind = chunked_table_kind_name(kind);
    table.node_count = node_count;
    table.coeff_cache = cache;

    for (const ChunkedSitepairJob& job : read_chunked_transform_jobs()) {
        if (job.kind != kind) {
            continue;
        }
        const std::filesystem::path coeff_path =
            chunked_coeff_path(kind, job.branch, job.sitepair_begin, job.sitepair_end);
        require_file_size(coeff_path, (job.sitepair_end - job.sitepair_begin) * node_count * sizeof(RealT));
        DiskBackedCoeffChunk chunk{job.sitepair_begin, job.sitepair_end, coeff_path};
        if (job.branch == TriangleBranch::Upper) {
            table.upper_coeff_chunks.push_back(std::move(chunk));
        } else {
            table.lower_coeff_chunks.push_back(std::move(chunk));
        }
    }

    const auto by_begin = [](const DiskBackedCoeffChunk& lhs, const DiskBackedCoeffChunk& rhs) {
        return lhs.sitepair_begin < rhs.sitepair_begin;
    };
    std::sort(table.upper_coeff_chunks.begin(), table.upper_coeff_chunks.end(), by_begin);
    std::sort(table.lower_coeff_chunks.begin(), table.lower_coeff_chunks.end(), by_begin);
    if (table.upper_coeff_chunks.empty() || table.lower_coeff_chunks.empty()) {
        throw std::runtime_error("Disk-backed table is missing coefficient chunks for " + table.table_kind);
    }

    table.upper_corners.assign(sitepair_count, RealT(0));
    table.lower_corners.assign(sitepair_count, RealT(0));
    load_corner_chunk_for_branch(table.upper_corners, kind, TriangleBranch::Upper);
    load_corner_chunk_for_branch(table.lower_corners, kind, TriangleBranch::Lower);
    return table;
}

template <typename RealT>
inline void load_chunked_data_into_resident_tables_for_real() {
    ensure_chunk_workflow_real_types_match();
    ensure_manifest_matches_for_real<RealT>();
    if (!std::filesystem::exists(chunked_ready_path())) {
        throw std::runtime_error("Chunked load-data directory is missing READY: " + chunked_ready_path().string());
    }
    std::cout << "  - Resident Green-table real type = " << real_type_name<RealT>() << '\n';
    g_table_up = load_chunked_table_for_kind<RealT>(ChunkedGreenTableKind::TwistedUp);
    g_table_down = load_chunked_table_for_kind<RealT>(ChunkedGreenTableKind::TwistedDown);
    g_table_ordinary_up = load_chunked_table_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryUp);
    g_table_ordinary_down = load_chunked_table_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryDown);
}

template <typename RealT>
inline void load_chunked_data_into_disk_lru_tables_for_real() {
    ensure_chunk_workflow_real_types_match();
    ensure_manifest_matches_for_real<RealT>();
    if (!std::filesystem::exists(chunked_ready_path())) {
        throw std::runtime_error("Chunked load-data directory is missing READY: " + chunked_ready_path().string());
    }
    if (!(mcmcmc_lru_cache_gb > 0.0)) {
        throw std::runtime_error("MCMCMC_LRU_CACHE_GB must be positive for disk_lru backend");
    }
    const size_t max_cache_bytes = static_cast<size_t>(
        mcmcmc_lru_cache_gb * 1024.0 * 1024.0 * 1024.0);
    auto cache = std::make_shared<DiskBackedGreenCoeffCache<RealT>>(max_cache_bytes);
    std::cout << "  - Disk-backed Green-table real type = " << real_type_name<RealT>() << '\n';
    std::cout << "  - MCMCMC coefficient LRU cache cap = " << mcmcmc_lru_cache_gb << " GiB\n";
    std::cout << "  - Corner values are loaded resident; coefficient chunks stay on disk\n";

    g_disk_table_up = open_disk_backed_table_for_kind<RealT>(ChunkedGreenTableKind::TwistedUp, cache);
    g_disk_table_down = open_disk_backed_table_for_kind<RealT>(ChunkedGreenTableKind::TwistedDown, cache);
    g_disk_table_ordinary_up = open_disk_backed_table_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryUp, cache);
    g_disk_table_ordinary_down = open_disk_backed_table_for_kind<RealT>(ChunkedGreenTableKind::OrdinaryDown, cache);
    g_disk_tables_initialized = true;
    g_disk_diagonal_order1_value_available = false;
    if (std::filesystem::exists(chunked_diagonal_observable_path())) {
        g_disk_diagonal_order1_value = read_chunked_diagonal_observable_value();
        g_disk_diagonal_order1_value_available = true;
        std::cout << "  - Precomputed diagonal order-1 observable = "
                  << chunked_dd_string(g_disk_diagonal_order1_value) << '\n';
    } else {
        std::cout << "  - No precomputed diagonal observable found; diagonal will fall back to LRU lookup\n";
    }
}

template <typename Fn>
inline void dispatch_chunked_real_type(Fn&& fn) {
    ensure_chunk_workflow_real_types_match();
    switch (parse_vector_real_type(vector_real_name)) {
        case IntegrandRealType::Double:
            fn.template operator()<double>();
            return;
        case IntegrandRealType::LongDouble:
            fn.template operator()<long double>();
            return;
        case IntegrandRealType::DoubleDouble:
            fn.template operator()<DoubleDouble>();
            return;
    }
    throw std::runtime_error("Unhandled VECTOR_REAL for chunked workflow");
}

inline void run_chunked_load_data_planner() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_plan_for_real<RealT>();
    });
}

inline void run_chunked_load_data_precompute_calculators() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_precompute_calculators_for_real<RealT>();
    });
}

inline void run_chunked_load_data_sample_node_job() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_sample_node_job_for_real<RealT>();
    });
}

inline void run_chunked_load_data_transform_sitepair_job() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_transform_sitepair_job_for_real<RealT>();
    });
}

inline void run_chunked_load_data_diagonal_reduce() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_diagonal_reduce_for_real<RealT>();
    });
}

inline void run_chunked_load_data_corner_job() {
    dispatch_chunked_real_type([]<typename RealT>() {
        run_chunked_corner_job_for_real<RealT>();
    });
}

inline void validate_chunked_load_data_outputs() {
    dispatch_chunked_real_type([]<typename RealT>() {
        validate_chunked_outputs_for_real<RealT>();
    });
}

inline void load_chunked_data_into_resident_tables() {
    dispatch_chunked_real_type([]<typename RealT>() {
        load_chunked_data_into_resident_tables_for_real<RealT>();
    });
}

inline void load_chunked_data_into_disk_lru_tables() {
    dispatch_chunked_real_type([]<typename RealT>() {
        load_chunked_data_into_disk_lru_tables_for_real<RealT>();
    });
}

#endif
