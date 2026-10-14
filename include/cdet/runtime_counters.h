#ifndef CDET_RUNTIME_COUNTERS_H
#define CDET_RUNTIME_COUNTERS_H

#include <atomic>
#include <cstdint>

struct CDetRuntimeCountersSnapshot {
    std::uint64_t integrand_calls = 0;
    std::uint64_t fpm_fallbacks = 0;
    std::uint64_t green_pair_cache_hits = 0;
    std::uint64_t green_pair_cache_misses = 0;
};

inline std::atomic<std::uint64_t> cdet_integrand_call_counter{0};
inline std::atomic<std::uint64_t> cdet_fpm_fallback_counter{0};
inline std::atomic<std::uint64_t> cdet_green_pair_cache_hit_counter{0};
inline std::atomic<std::uint64_t> cdet_green_pair_cache_miss_counter{0};

inline void record_cdet_integrand_call() noexcept {
    cdet_integrand_call_counter.fetch_add(1, std::memory_order_relaxed);
}

inline void record_cdet_integrand_calls(std::uint64_t count) noexcept {
    cdet_integrand_call_counter.fetch_add(count, std::memory_order_relaxed);
}

inline void record_cdet_fpm_fallback() noexcept {
    cdet_fpm_fallback_counter.fetch_add(1, std::memory_order_relaxed);
}

inline void record_cdet_green_pair_cache_counts(std::uint64_t hits,
                                                std::uint64_t misses) noexcept {
    cdet_green_pair_cache_hit_counter.fetch_add(hits, std::memory_order_relaxed);
    cdet_green_pair_cache_miss_counter.fetch_add(misses, std::memory_order_relaxed);
}

inline CDetRuntimeCountersSnapshot snapshot_cdet_runtime_counters() noexcept {
    return {
        cdet_integrand_call_counter.load(std::memory_order_relaxed),
        cdet_fpm_fallback_counter.load(std::memory_order_relaxed),
        cdet_green_pair_cache_hit_counter.load(std::memory_order_relaxed),
        cdet_green_pair_cache_miss_counter.load(std::memory_order_relaxed),
    };
}

inline void reset_cdet_runtime_counters() noexcept {
    cdet_integrand_call_counter.store(0, std::memory_order_relaxed);
    cdet_fpm_fallback_counter.store(0, std::memory_order_relaxed);
    cdet_green_pair_cache_hit_counter.store(0, std::memory_order_relaxed);
    cdet_green_pair_cache_miss_counter.store(0, std::memory_order_relaxed);
}

#endif
