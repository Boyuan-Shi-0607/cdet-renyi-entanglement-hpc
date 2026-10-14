#ifndef REPLICA_CHEBYSHEV_TABLES_H
#define REPLICA_CHEBYSHEV_TABLES_H

#include "cdet/integrand_parallel.h"
#include "cdet/precision_config.h"
#include "cdet/replica_free_greens.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

template <typename T>
struct PiecewiseChebyshevTable {
    using value_type = T;

    std::vector<T> upper_coeffs;
    std::vector<T> lower_coeffs;
    std::vector<T> upper_corners;
    std::vector<T> lower_corners;
};

using GreenFunctionTableMP = PiecewiseChebyshevTable<MP_Real>;
using GreenFunctionTableDD = PiecewiseChebyshevTable<DoubleDouble>;
using GreenFunctionTableLongDouble = PiecewiseChebyshevTable<long double>;
using GreenFunctionTableDouble = PiecewiseChebyshevTable<double>;

struct DiskBackedGreenCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t cached_bytes = 0;
    std::uint64_t max_bytes = 0;
    std::uint64_t entry_count = 0;
};

struct DiskBackedCoeffChunk {
    size_t sitepair_begin = 0;
    size_t sitepair_end = 0;
    std::filesystem::path path;
};

template <typename RealT>
class DiskBackedGreenCoeffCache {
public:
    using BlockPtr = std::shared_ptr<const std::vector<RealT>>;

    explicit DiskBackedGreenCoeffCache(size_t max_bytes)
        : max_bytes_(max_bytes) {}

    BlockPtr get_block(const std::string& table_kind,
                       TriangleBranch branch,
                       const DiskBackedCoeffChunk& chunk,
                       size_t node_count) {
        const std::string key = make_key(table_kind, branch, chunk.sitepair_begin, chunk.sitepair_end);
        const size_t value_count = (chunk.sitepair_end - chunk.sitepair_begin) * node_count;
        const size_t byte_count = value_count * sizeof(RealT);
        if (byte_count == 0) {
            throw std::runtime_error("Disk-backed Green chunk is empty: " + chunk.path.string());
        }
        if (byte_count > max_bytes_) {
            throw std::runtime_error(
                "MCMCMC_LRU_CACHE_GB is smaller than one coefficient chunk; reduce "
                "LOAD_DATA_SITEPAIR_CHUNK_SIZE or increase MCMCMC_LRU_CACHE_GB");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            ++stats_.hits;
            touch(found->second);
            return found->second.values;
        }

        ++stats_.misses;
        auto values = std::make_shared<std::vector<RealT>>(value_count);
        std::ifstream in(chunk.path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open disk-backed Green chunk: " + chunk.path.string());
        }
        in.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(byte_count));
        if (in.gcount() != static_cast<std::streamsize>(byte_count) || !in) {
            throw std::runtime_error("Short read from disk-backed Green chunk: " + chunk.path.string());
        }

        while (cached_bytes_ + byte_count > max_bytes_ && !lru_keys_.empty()) {
            const std::string evict_key = lru_keys_.back();
            auto evict_it = entries_.find(evict_key);
            if (evict_it != entries_.end()) {
                cached_bytes_ -= evict_it->second.bytes;
                entries_.erase(evict_it);
                ++stats_.evictions;
            }
            lru_keys_.pop_back();
        }

        lru_keys_.push_front(key);
        entries_.emplace(key, Entry{values, byte_count, lru_keys_.begin()});
        cached_bytes_ += byte_count;
        stats_.bytes_read += static_cast<std::uint64_t>(byte_count);
        return values;
    }

    DiskBackedGreenCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        DiskBackedGreenCacheStats snapshot = stats_;
        snapshot.cached_bytes = static_cast<std::uint64_t>(cached_bytes_);
        snapshot.max_bytes = static_cast<std::uint64_t>(max_bytes_);
        snapshot.entry_count = static_cast<std::uint64_t>(entries_.size());
        return snapshot;
    }

private:
    struct Entry {
        BlockPtr values;
        size_t bytes = 0;
        std::list<std::string>::iterator lru_it;
    };

    static std::string make_key(const std::string& table_kind,
                                TriangleBranch branch,
                                size_t begin,
                                size_t end) {
        return table_kind + "|" + (branch == TriangleBranch::Upper ? "upper" : "lower") +
               "|" + std::to_string(begin) + "|" + std::to_string(end);
    }

    void touch(Entry& entry) {
        lru_keys_.splice(lru_keys_.begin(), lru_keys_, entry.lru_it);
        entry.lru_it = lru_keys_.begin();
    }

    size_t max_bytes_ = 0;
    size_t cached_bytes_ = 0;
    mutable std::mutex mutex_;
    std::list<std::string> lru_keys_;
    std::unordered_map<std::string, Entry> entries_;
    DiskBackedGreenCacheStats stats_;
};

template <typename RealT>
struct DiskBackedPiecewiseChebyshevTable {
    using value_type = RealT;

    std::string table_kind;
    size_t node_count = 0;
    std::shared_ptr<DiskBackedGreenCoeffCache<RealT>> coeff_cache;
    std::vector<DiskBackedCoeffChunk> upper_coeff_chunks;
    std::vector<DiskBackedCoeffChunk> lower_coeff_chunks;
    std::vector<RealT> upper_corners;
    std::vector<RealT> lower_corners;

    const DiskBackedCoeffChunk& find_chunk(TriangleBranch branch, size_t sitepair_index) const {
        const std::vector<DiskBackedCoeffChunk>& chunks =
            branch == TriangleBranch::Upper ? upper_coeff_chunks : lower_coeff_chunks;
        auto it = std::upper_bound(
            chunks.begin(),
            chunks.end(),
            sitepair_index,
            [](size_t value, const DiskBackedCoeffChunk& chunk) {
                return value < chunk.sitepair_end;
            });
        if (it == chunks.end() || sitepair_index < it->sitepair_begin) {
            throw std::runtime_error("Disk-backed Green table is missing requested sitepair chunk");
        }
        return *it;
    }

    std::shared_ptr<const std::vector<RealT>> coeff_block(TriangleBranch branch,
                                                          size_t sitepair_index) const {
        if (!coeff_cache) {
            throw std::runtime_error("Disk-backed Green table cache is not initialized");
        }
        const DiskBackedCoeffChunk& chunk = find_chunk(branch, sitepair_index);
        return coeff_cache->get_block(table_kind, branch, chunk, node_count);
    }

    void copy_coefficients(TriangleBranch branch,
                           size_t sitepair_index,
                           std::vector<RealT>& destination) const {
        if (!coeff_cache) {
            throw std::runtime_error("Disk-backed Green table cache is not initialized");
        }
        const DiskBackedCoeffChunk& chunk = find_chunk(branch, sitepair_index);
        const auto block = coeff_cache->get_block(table_kind, branch, chunk, node_count);
        const size_t local_offset = (sitepair_index - chunk.sitepair_begin) * node_count;
        destination.assign(block->begin() + static_cast<std::ptrdiff_t>(local_offset),
                           block->begin() + static_cast<std::ptrdiff_t>(local_offset + node_count));
    }
};

using DiskBackedGreenFunctionTable =
    std::variant<DiskBackedPiecewiseChebyshevTable<double>,
                 DiskBackedPiecewiseChebyshevTable<long double>,
                 DiskBackedPiecewiseChebyshevTable<DoubleDouble>>;

using ResidentGreenFunctionTable =
    std::variant<GreenFunctionTableDouble, GreenFunctionTableLongDouble, GreenFunctionTableDD>;

struct GreenTableTruncationSummary {
    std::size_t value_count = 0;
    int native_digits = 0;
    int keep_bits = 0;
    long double max_abs_delta = 0.0L;
    long double max_rel_delta = 0.0L;

    void merge(const GreenTableTruncationSummary& other) {
        value_count += other.value_count;
        native_digits = std::max(native_digits, other.native_digits);
        keep_bits = std::max(keep_bits, other.keep_bits);
        max_abs_delta = std::max(max_abs_delta, other.max_abs_delta);
        max_rel_delta = std::max(max_rel_delta, other.max_rel_delta);
    }
};

template <typename RealT>
inline constexpr int resident_green_table_mantissa_digits() {
    if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return 106;
    } else {
        return std::numeric_limits<RealT>::digits;
    }
}

template <typename RealT>
inline int resident_green_table_keep_bits(int drop_bits) {
    const int native_digits = resident_green_table_mantissa_digits<RealT>();
    return std::max(1, native_digits - std::max(0, drop_bits));
}

template <typename RealT>
inline RealT truncate_real_mantissa_bits(const RealT& value, int drop_bits) {
    if (drop_bits <= 0) {
        return value;
    }

    const int keep_bits = resident_green_table_keep_bits<RealT>(drop_bits);
    if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        if (!std::isfinite(value.hi) || !std::isfinite(value.lo) ||
            (value.hi == 0.0 && value.lo == 0.0)) {
            return value;
        }

        const MP_Real mp_value = value.to_mp_real();
        if (mp_value == MP_Real(0)) {
            return value;
        }

        const bool negative = mp_value < MP_Real(0);
        const MP_Real abs_value = negative ? -mp_value : mp_value;
        const double exponent_probe = std::fabs(value.hi != 0.0 ? value.hi : value.lo);
        int exponent = 0;
        std::frexp(exponent_probe, &exponent);
        const MP_Real scale =
            boost::multiprecision::ldexp(MP_Real(1), keep_bits - exponent);
        const MP_Real truncated_abs =
            boost::multiprecision::floor(abs_value * scale) / scale;
        return DoubleDouble(negative ? -truncated_abs : truncated_abs);
    } else {
        if (!std::isfinite(value) || value == RealT(0)) {
            return value;
        }

        int exponent = 0;
        const RealT mantissa = std::frexp(value, &exponent);
        const RealT scaled = std::ldexp(mantissa, keep_bits);
        const RealT truncated_scaled = std::trunc(scaled);
        return std::ldexp(truncated_scaled, exponent - keep_bits);
    }
}

template <typename RealT>
inline void update_green_table_truncation_summary(GreenTableTruncationSummary& summary,
                                                  const RealT& before,
                                                  const RealT& after) {
    if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        const MP_Real before_mp = before.to_mp_real();
        const MP_Real after_mp = after.to_mp_real();
        const MP_Real delta_mp = boost::multiprecision::abs(before_mp - after_mp);
        summary.max_abs_delta =
            std::max(summary.max_abs_delta, static_cast<long double>(delta_mp));
        const MP_Real denom_mp = boost::multiprecision::abs(before_mp);
        if (denom_mp > MP_Real(0)) {
            summary.max_rel_delta =
                std::max(summary.max_rel_delta, static_cast<long double>(delta_mp / denom_mp));
        }
    } else {
        const long double before_ld = real_precision_cast<long double>(before);
        const long double after_ld = real_precision_cast<long double>(after);
        const long double delta = std::fabs(before_ld - after_ld);
        summary.max_abs_delta = std::max(summary.max_abs_delta, delta);
        const long double denom = std::fabs(before_ld);
        if (denom > 0.0L) {
            summary.max_rel_delta = std::max(summary.max_rel_delta, delta / denom);
        }
    }
}

template <typename RealT>
inline void truncate_green_table_vector_mantissa(std::vector<RealT>& values,
                                                 int drop_bits,
                                                 GreenTableTruncationSummary& summary) {
    for (RealT& value : values) {
        const RealT before = value;
        value = truncate_real_mantissa_bits(value, drop_bits);
        update_green_table_truncation_summary(summary, before, value);
    }
    summary.value_count += values.size();
}

template <typename RealT>
inline GreenTableTruncationSummary truncate_green_function_table_mantissa(
    PiecewiseChebyshevTable<RealT>& table,
    int drop_bits) {
    GreenTableTruncationSummary summary;
    summary.native_digits = resident_green_table_mantissa_digits<RealT>();
    summary.keep_bits = resident_green_table_keep_bits<RealT>(drop_bits);

    if (drop_bits <= 0) {
        return summary;
    }

    truncate_green_table_vector_mantissa(table.upper_coeffs, drop_bits, summary);
    truncate_green_table_vector_mantissa(table.lower_coeffs, drop_bits, summary);
    truncate_green_table_vector_mantissa(table.upper_corners, drop_bits, summary);
    truncate_green_table_vector_mantissa(table.lower_corners, drop_bits, summary);
    return summary;
}

struct ReplicaSitepairMeta {
    std::uint16_t site1 = 0;
    std::uint16_t site2 = 0;
    std::uint8_t replica_alpha = 0;
    std::uint8_t replica_beta = 0;
};

template <typename RealT>
struct ChebyshevNodeTables {
    RealVector<RealT> cheb_node_u;
    RealVector<RealT> cheb_node_v;
    RealVector<RealT> alpha_u;
    RealVector<RealT> alpha_v;
    RealVector<RealT> cos_table_u;
    RealVector<RealT> cos_table_v;
};

inline size_t replica_sitepair_count(int N1, int N2) {
    return static_cast<size_t>(2) * static_cast<size_t>(2) * static_cast<size_t>(N1) * static_cast<size_t>(N2) *
           static_cast<size_t>(N1) * static_cast<size_t>(N2);
}

inline size_t replica_sitepair_index(int replica_alpha,
                                     int replica_beta,
                                     int R1_b1,
                                     int R1_b2,
                                     int R2_b1,
                                     int R2_b2,
                                     int N1,
                                     int N2) {
    return ((((((static_cast<size_t>(replica_alpha) * 2 + static_cast<size_t>(replica_beta)) *
                    static_cast<size_t>(N1) +
                static_cast<size_t>(R1_b1)) *
                   static_cast<size_t>(N2) +
               static_cast<size_t>(R1_b2)) *
                  static_cast<size_t>(N1) +
              static_cast<size_t>(R2_b1)) *
                 static_cast<size_t>(N2) +
             static_cast<size_t>(R2_b2)));
}

inline size_t full_coeff_index(int cheb_u,
                               int cheb_v,
                               int replica_alpha,
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
               coeff_count_per_replica_sitepair +
           static_cast<size_t>(cheb_u) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(cheb_v);
}

inline std::array<int, 6> decode_replica_sitepair_index(size_t replica_sitepair_flat_index, int N1, int N2) {
    const size_t dimN2 = static_cast<size_t>(N2);
    const size_t dimN1 = static_cast<size_t>(N1);

    const int R2_b2 = static_cast<int>(replica_sitepair_flat_index % dimN2);
    replica_sitepair_flat_index /= dimN2;
    const int R2_b1 = static_cast<int>(replica_sitepair_flat_index % dimN1);
    replica_sitepair_flat_index /= dimN1;
    const int R1_b2 = static_cast<int>(replica_sitepair_flat_index % dimN2);
    replica_sitepair_flat_index /= dimN2;
    const int R1_b1 = static_cast<int>(replica_sitepair_flat_index % dimN1);
    replica_sitepair_flat_index /= dimN1;
    const int replica_beta = static_cast<int>(replica_sitepair_flat_index % 2U);
    replica_sitepair_flat_index /= 2U;
    const int replica_alpha = static_cast<int>(replica_sitepair_flat_index % 2U);

    return {replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2};
}

inline size_t corner_replica_sitepair_index(int replica_alpha,
                                            int replica_beta,
                                            int R1_b1,
                                            int R1_b2,
                                            int R2_b1,
                                            int R2_b2,
                                            int N1,
                                            int N2) {
    return replica_sitepair_index(replica_alpha, replica_beta, R1_b1, R1_b2, R2_b1, R2_b2, N1, N2);
}

template <typename RealT>
inline ChebyshevNodeTables<RealT> build_chebyshev_node_tables(int cheb_nu, int cheb_nv) {
    ChebyshevNodeTables<RealT> tables;
    tables.cheb_node_u.resize(static_cast<size_t>(cheb_nu));
    tables.cheb_node_v.resize(static_cast<size_t>(cheb_nv));
    tables.alpha_u.resize(static_cast<size_t>(cheb_nu));
    tables.alpha_v.resize(static_cast<size_t>(cheb_nv));
    tables.cos_table_u.resize(static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nu));
    tables.cos_table_v.resize(static_cast<size_t>(cheb_nv) * static_cast<size_t>(cheb_nv));

    const RealT pi(get_pi());
    for (int i = 0; i < cheb_nu; ++i) {
        const RealT theta = (RealT(2 * i + 1) * pi) / RealT(2 * cheb_nu);
        tables.cheb_node_u[static_cast<size_t>(i)] = real_cos_value(theta);
    }
    for (int j = 0; j < cheb_nv; ++j) {
        const RealT theta = (RealT(2 * j + 1) * pi) / RealT(2 * cheb_nv);
        tables.cheb_node_v[static_cast<size_t>(j)] = real_cos_value(theta);
    }

    for (int a = 0; a < cheb_nu; ++a) {
        tables.alpha_u[static_cast<size_t>(a)] = (a == 0) ? RealT(1) : RealT(2);
        for (int i = 0; i < cheb_nu; ++i) {
            const RealT theta = (RealT(2 * i + 1) * pi) / RealT(2 * cheb_nu);
            tables.cos_table_u[static_cast<size_t>(a) * static_cast<size_t>(cheb_nu) + static_cast<size_t>(i)] =
                real_cos_value(RealT(RealT(a) * theta));
        }
    }

    for (int b = 0; b < cheb_nv; ++b) {
        tables.alpha_v[static_cast<size_t>(b)] = (b == 0) ? RealT(1) : RealT(2);
        for (int j = 0; j < cheb_nv; ++j) {
            const RealT theta = (RealT(2 * j + 1) * pi) / RealT(2 * cheb_nv);
            tables.cos_table_v[static_cast<size_t>(b) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(j)] =
                real_cos_value(RealT(RealT(b) * theta));
        }
    }

    return tables;
}

template <typename RealT>
inline std::pair<RealT, RealT> map_square_to_triangle(const RealT& cheb_u,
                                                      const RealT& cheb_v,
                                                      TriangleBranch branch,
                                                      const RealT& beta_s) {
    if (branch == TriangleBranch::Upper) {
        const RealT tau = beta_s * (RealT(1) + cheb_u) / RealT(2);
        const RealT tau_prime = tau * (RealT(1) + cheb_v) / RealT(2);
        return {tau, tau_prime};
    }

    const RealT tau_prime = beta_s * (RealT(1) + cheb_u) / RealT(2);
    const RealT tau = tau_prime * (RealT(1) + cheb_v) / RealT(2);
    return {tau, tau_prime};
}

template <typename RealT>
inline RealT replica_entry_from_sector_matrices(const Matrix<ComplexNumber<RealT>>& sector_m0,
                                                const Matrix<ComplexNumber<RealT>>& sector_m1,
                                                int replica_alpha,
                                                int replica_beta,
                                                int site1,
                                                int site2) {
    using ComplexT = ComplexNumber<RealT>;

    const ComplexT& g0 = sector_m0(site1, site2);
    const ComplexT& g1 = sector_m1(site1, site2);

    if (replica_alpha == replica_beta) {
        return real_part_if_nearly_real((g0 + g1) / ComplexT(RealT(2)));
    }

    if (replica_alpha == 0 && replica_beta == 1) {
        return real_part_if_nearly_real(complex_i<RealT>() * (g0 - g1) / ComplexT(RealT(2)));
    }

    return real_part_if_nearly_real(-complex_i<RealT>() * (g0 - g1) / ComplexT(RealT(2)));
}

inline std::vector<ReplicaSitepairMeta> build_replica_sitepair_meta(int N1, int N2) {
    const size_t count = replica_sitepair_count(N1, N2);
    std::vector<ReplicaSitepairMeta> meta(count);

    for (size_t flat_index = 0; flat_index < count; ++flat_index) {
        const auto decoded = decode_replica_sitepair_index(flat_index, N1, N2);
        meta[flat_index] = ReplicaSitepairMeta{
            static_cast<std::uint16_t>(site_index(decoded[2], decoded[3], N1, N2)),
            static_cast<std::uint16_t>(site_index(decoded[4], decoded[5], N1, N2)),
            static_cast<std::uint8_t>(decoded[0]),
            static_cast<std::uint8_t>(decoded[1]),
        };
    }

    return meta;
}

template <typename RealT>
inline std::vector<std::pair<RealT, RealT>> build_branch_node_times(const ChebyshevNodeTables<RealT>& tables,
                                                                    int cheb_nu,
                                                                    int cheb_nv,
                                                                    TriangleBranch branch,
                                                                    const RealT& beta_s) {
    const size_t node_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    std::vector<std::pair<RealT, RealT>> node_times(node_count);

    for (int i = 0; i < cheb_nu; ++i) {
        for (int j = 0; j < cheb_nv; ++j) {
            const size_t node_index = static_cast<size_t>(i) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(j);
            node_times[node_index] = map_square_to_triangle(
                tables.cheb_node_u[static_cast<size_t>(i)],
                tables.cheb_node_v[static_cast<size_t>(j)],
                branch,
                beta_s);
        }
    }

    return node_times;
}

template <typename RealT>
inline void transform_sitepair_sample_grid_to_coeffs(RealT* block,
                                                     const ChebyshevNodeTables<RealT>& tables,
                                                     int cheb_nu,
                                                     int cheb_nv) {
    const size_t coeff_count = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    thread_local std::vector<RealT> temp_buffer;
    temp_buffer.assign(coeff_count, RealT(0));

    for (int i = 0; i < cheb_nu; ++i) {
        const RealT* sample_row = block + static_cast<size_t>(i) * static_cast<size_t>(cheb_nv);
        for (int b = 0; b < cheb_nv; ++b) {
            const RealT prefactor_v = tables.alpha_v[static_cast<size_t>(b)] / RealT(cheb_nv);
            const RealT* cos_v_row =
                tables.cos_table_v.data() + static_cast<size_t>(b) * static_cast<size_t>(cheb_nv);
            RealT total = RealT(0);
            for (int j = 0; j < cheb_nv; ++j) {
                total += sample_row[static_cast<size_t>(j)] * cos_v_row[static_cast<size_t>(j)];
            }
            temp_buffer[static_cast<size_t>(i) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(b)] =
                prefactor_v * total;
        }
    }

    for (int a = 0; a < cheb_nu; ++a) {
        const RealT prefactor_u = tables.alpha_u[static_cast<size_t>(a)] / RealT(cheb_nu);
        const RealT* cos_u_row =
            tables.cos_table_u.data() + static_cast<size_t>(a) * static_cast<size_t>(cheb_nu);
        for (int b = 0; b < cheb_nv; ++b) {
            RealT total = RealT(0);
            for (int i = 0; i < cheb_nu; ++i) {
                total += temp_buffer[static_cast<size_t>(i) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(b)] *
                         cos_u_row[static_cast<size_t>(i)];
            }
            block[static_cast<size_t>(a) * static_cast<size_t>(cheb_nv) + static_cast<size_t>(b)] =
                prefactor_u * total;
        }
    }
}

template <typename RealT>
inline PiecewiseChebyshevTable<RealT> build_green_function_table(const ReplicaGreenCalculator<RealT>& calculator,
                                                                 int cheb_nu,
                                                                 int cheb_nv,
                                                                 int build_threads = 1) {
    using ComplexT = ComplexNumber<RealT>;

    const int N1 = calculator.N1();
    const int N2 = calculator.N2();
    const size_t sitepair_count = replica_sitepair_count(N1, N2);
    const size_t coeff_count_per_sitepair = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t total_coeff_count = coeff_count_per_sitepair * sitepair_count;
    const ChebyshevNodeTables<RealT> tables = build_chebyshev_node_tables<RealT>(cheb_nu, cheb_nv);
    const std::vector<ReplicaSitepairMeta> sitepair_meta = build_replica_sitepair_meta(N1, N2);
    const std::vector<std::pair<RealT, RealT>> upper_node_times =
        build_branch_node_times(tables, cheb_nu, cheb_nv, TriangleBranch::Upper, calculator.beta());
    const std::vector<std::pair<RealT, RealT>> lower_node_times =
        build_branch_node_times(tables, cheb_nu, cheb_nv, TriangleBranch::Lower, calculator.beta());

    PiecewiseChebyshevTable<RealT> table;
    table.upper_coeffs.assign(total_coeff_count, RealT(0));
    table.lower_coeffs.assign(total_coeff_count, RealT(0));
    table.upper_corners.assign(sitepair_count, RealT(0));
    table.lower_corners.assign(sitepair_count, RealT(0));

    const size_t requested_threads = static_cast<size_t>(std::max(1, build_threads));

    const auto fill_corner_table = [&](TriangleBranch branch, std::vector<RealT>& destination) {
        const auto sector_matrices = calculator.green_sector_matrices(RealT(0), RealT(0), branch);
        const Matrix<ComplexT>& sector_m0 = sector_matrices[0];
        const Matrix<ComplexT>& sector_m1 = sector_matrices[1];

        for (size_t flat_index = 0; flat_index < sitepair_count; ++flat_index) {
            const auto& meta = sitepair_meta[flat_index];
            destination[flat_index] = replica_entry_from_sector_matrices(
                sector_m0,
                sector_m1,
                static_cast<int>(meta.replica_alpha),
                static_cast<int>(meta.replica_beta),
                static_cast<int>(meta.site1),
                static_cast<int>(meta.site2));
        }
    };

    fill_corner_table(TriangleBranch::Upper, table.upper_corners);
    fill_corner_table(TriangleBranch::Lower, table.lower_corners);

    const auto fill_branch_sample_grid = [&](TriangleBranch branch,
                                             const std::vector<std::pair<RealT, RealT>>& node_times,
                                             std::vector<RealT>& destination) {
        const size_t node_count = node_times.size();

        auto fill_node_range = [&](size_t /*worker_index*/, size_t begin, size_t end) {
            for (size_t node_index = begin; node_index < end; ++node_index) {
                const auto& [tau, tau_prime] = node_times[node_index];
                const auto sector_matrices = calculator.green_sector_matrices(tau, tau_prime, branch);
                const Matrix<ComplexT>& sector_m0 = sector_matrices[0];
                const Matrix<ComplexT>& sector_m1 = sector_matrices[1];

                for (size_t flat_index = 0; flat_index < sitepair_count; ++flat_index) {
                    const auto& meta = sitepair_meta[flat_index];
                    destination[flat_index * coeff_count_per_sitepair + node_index] =
                        replica_entry_from_sector_matrices(
                            sector_m0,
                            sector_m1,
                            static_cast<int>(meta.replica_alpha),
                            static_cast<int>(meta.replica_beta),
                            static_cast<int>(meta.site1),
                            static_cast<int>(meta.site2));
                }
            }
        };

        if (requested_threads > 1 && node_count > 1) {
            IntegrandParallelExecutor::instance().parallel_for(requested_threads, node_count, fill_node_range);
        } else {
            fill_node_range(0, 0, node_count);
        }
    };

    const auto transform_branch_samples_to_coeffs = [&](std::vector<RealT>& destination) {
        auto transform_sitepair_range = [&](size_t /*worker_index*/, size_t begin, size_t end) {
            for (size_t flat_index = begin; flat_index < end; ++flat_index) {
                RealT* block = destination.data() + flat_index * coeff_count_per_sitepair;
                transform_sitepair_sample_grid_to_coeffs(block, tables, cheb_nu, cheb_nv);
            }
        };

        if (requested_threads > 1 && sitepair_count > 1) {
            IntegrandParallelExecutor::instance().parallel_for(
                requested_threads, sitepair_count, transform_sitepair_range);
        } else {
            transform_sitepair_range(0, 0, sitepair_count);
        }
    };

    fill_branch_sample_grid(TriangleBranch::Upper, upper_node_times, table.upper_coeffs);
    transform_branch_samples_to_coeffs(table.upper_coeffs);

    fill_branch_sample_grid(TriangleBranch::Lower, lower_node_times, table.lower_coeffs);
    transform_branch_samples_to_coeffs(table.lower_coeffs);

    return table;
}

template <typename RealT>
inline PiecewiseChebyshevTable<RealT> build_ordinary_replica00_green_function_table(
    const ReplicaGreenCalculator<RealT>& calculator,
    int cheb_nu,
    int cheb_nv,
    int build_threads = 1) {
    const int N1 = calculator.N1();
    const int N2 = calculator.N2();
    const int n_sites = calculator.n_sites();
    const size_t full_sitepair_count = replica_sitepair_count(N1, N2);
    const size_t active_sitepair_count = static_cast<size_t>(n_sites) * static_cast<size_t>(n_sites);
    const size_t coeff_count_per_sitepair = static_cast<size_t>(cheb_nu) * static_cast<size_t>(cheb_nv);
    const size_t total_coeff_count = coeff_count_per_sitepair * full_sitepair_count;
    const ChebyshevNodeTables<RealT> tables = build_chebyshev_node_tables<RealT>(cheb_nu, cheb_nv);
    const std::vector<std::pair<RealT, RealT>> upper_node_times =
        build_branch_node_times(tables, cheb_nu, cheb_nv, TriangleBranch::Upper, calculator.beta());
    const std::vector<std::pair<RealT, RealT>> lower_node_times =
        build_branch_node_times(tables, cheb_nu, cheb_nv, TriangleBranch::Lower, calculator.beta());

    PiecewiseChebyshevTable<RealT> table;
    table.upper_coeffs.assign(total_coeff_count, RealT(0));
    table.lower_coeffs.assign(total_coeff_count, RealT(0));
    table.upper_corners.assign(full_sitepair_count, RealT(0));
    table.lower_corners.assign(full_sitepair_count, RealT(0));

    const size_t requested_threads = static_cast<size_t>(std::max(1, build_threads));

    const auto fill_corner_table = [&](TriangleBranch branch, std::vector<RealT>& destination) {
        const Matrix<ComplexNumber<RealT>> sector_m0 =
            calculator.green_sector_matrix(0, RealT(0), RealT(0), branch);
        for (int site1 = 0; site1 < n_sites; ++site1) {
            for (int site2 = 0; site2 < n_sites; ++site2) {
                const size_t flat_index =
                    static_cast<size_t>(site1) * static_cast<size_t>(n_sites) +
                    static_cast<size_t>(site2);
                destination[flat_index] = real_part_if_nearly_real(sector_m0(site1, site2));
            }
        }
    };

    fill_corner_table(TriangleBranch::Upper, table.upper_corners);
    fill_corner_table(TriangleBranch::Lower, table.lower_corners);

    const auto fill_branch_sample_grid = [&](TriangleBranch branch,
                                             const std::vector<std::pair<RealT, RealT>>& node_times,
                                             std::vector<RealT>& destination) {
        const size_t node_count = node_times.size();

        auto fill_node_range = [&](size_t /*worker_index*/, size_t begin, size_t end) {
            for (size_t node_index = begin; node_index < end; ++node_index) {
                const auto& [tau, tau_prime] = node_times[node_index];
                const Matrix<ComplexNumber<RealT>> sector_m0 =
                    calculator.green_sector_matrix(0, tau, tau_prime, branch);

                for (int site1 = 0; site1 < n_sites; ++site1) {
                    for (int site2 = 0; site2 < n_sites; ++site2) {
                        const size_t flat_index =
                            static_cast<size_t>(site1) * static_cast<size_t>(n_sites) +
                            static_cast<size_t>(site2);
                        destination[flat_index * coeff_count_per_sitepair + node_index] =
                            real_part_if_nearly_real(sector_m0(site1, site2));
                    }
                }
            }
        };

        if (requested_threads > 1 && node_count > 1) {
            IntegrandParallelExecutor::instance().parallel_for(requested_threads, node_count, fill_node_range);
        } else {
            fill_node_range(0, 0, node_count);
        }
    };

    const auto transform_branch_samples_to_coeffs = [&](std::vector<RealT>& destination) {
        auto transform_sitepair_range = [&](size_t /*worker_index*/, size_t begin, size_t end) {
            for (size_t flat_index = begin; flat_index < end; ++flat_index) {
                RealT* block = destination.data() + flat_index * coeff_count_per_sitepair;
                transform_sitepair_sample_grid_to_coeffs(block, tables, cheb_nu, cheb_nv);
            }
        };

        if (requested_threads > 1 && active_sitepair_count > 1) {
            IntegrandParallelExecutor::instance().parallel_for(
                requested_threads, active_sitepair_count, transform_sitepair_range);
        } else {
            transform_sitepair_range(0, 0, active_sitepair_count);
        }
    };

    fill_branch_sample_grid(TriangleBranch::Upper, upper_node_times, table.upper_coeffs);
    transform_branch_samples_to_coeffs(table.upper_coeffs);

    fill_branch_sample_grid(TriangleBranch::Lower, lower_node_times, table.lower_coeffs);
    transform_branch_samples_to_coeffs(table.lower_coeffs);

    return table;
}

template <typename TargetRealT, typename SourceRealT>
inline PiecewiseChebyshevTable<TargetRealT> convert_green_function_table(
    PiecewiseChebyshevTable<SourceRealT>&& source) {
    if constexpr (std::is_same_v<TargetRealT, SourceRealT>) {
        return std::move(source);
    } else {
        PiecewiseChebyshevTable<TargetRealT> target;
        target.upper_coeffs.reserve(source.upper_coeffs.size());
        target.lower_coeffs.reserve(source.lower_coeffs.size());
        target.upper_corners.reserve(source.upper_corners.size());
        target.lower_corners.reserve(source.lower_corners.size());

        for (const SourceRealT& value : source.upper_coeffs) {
            target.upper_coeffs.push_back(real_precision_cast<TargetRealT>(value));
        }
        for (const SourceRealT& value : source.lower_coeffs) {
            target.lower_coeffs.push_back(real_precision_cast<TargetRealT>(value));
        }
        for (const SourceRealT& value : source.upper_corners) {
            target.upper_corners.push_back(real_precision_cast<TargetRealT>(value));
        }
        for (const SourceRealT& value : source.lower_corners) {
            target.lower_corners.push_back(real_precision_cast<TargetRealT>(value));
        }

        return target;
    }
}

template <typename RealT>
inline GreenFunctionTableDD convert_green_function_table_to_dd(PiecewiseChebyshevTable<RealT>&& source) {
    return convert_green_function_table<DoubleDouble>(std::move(source));
}

template <typename BuildRealT, typename TargetRealT>
inline PiecewiseChebyshevTable<TargetRealT> build_green_function_table_as(
    const ReplicaGreenCalculator<BuildRealT>& calculator,
    int cheb_nu,
    int cheb_nv,
    int build_threads = 1) {
    return convert_green_function_table<TargetRealT>(
        build_green_function_table(calculator, cheb_nu, cheb_nv, build_threads));
}

template <typename RealT>
inline GreenFunctionTableDD build_green_function_table_dd_from_calculator(
    const ReplicaGreenCalculator<RealT>& calculator,
    int cheb_nu,
    int cheb_nv,
    int build_threads = 1) {
    return convert_green_function_table_to_dd(
        build_green_function_table(calculator, cheb_nu, cheb_nv, build_threads));
}

inline GreenFunctionTableMP build_green_function_table_mp(const ReplicaGreenCalculatorMP& calculator,
                                                          int cheb_nu,
                                                          int cheb_nv,
                                                          int build_threads = 1) {
    return build_green_function_table(calculator, cheb_nu, cheb_nv, build_threads);
}

inline GreenFunctionTableDD build_green_function_table_dd(const ReplicaGreenCalculatorDD& calculator,
                                                          int cheb_nu,
                                                          int cheb_nv,
                                                          int build_threads = 1) {
    return build_green_function_table(calculator, cheb_nu, cheb_nv, build_threads);
}

#endif
