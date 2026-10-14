#ifndef CDET_MCMCMC_SAMPLER_H
#define CDET_MCMCMC_SAMPLER_H

#include "cdet/cdet_integrand_evaluator.h"
#include "cdet/mcmcmc_seed.h"
#include "cdet/parameters.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Set heat-bath MCMCMC move used in FastFeynmanDiagrammatics.  Uniform seed is
// the special case where the conditional generation factor Pgen(full | subset)
// is one for every subset.
struct RunningStatsLongDouble {
    std::uint64_t count = 0;
    long double mean = 0.0L;
    long double m2 = 0.0L;

    void add(long double value) {
        ++count;
        const long double delta = value - mean;
        mean += delta / static_cast<long double>(count);
        const long double delta2 = value - mean;
        m2 += delta * delta2;
    }

    void merge(const RunningStatsLongDouble& other) {
        if (other.count == 0) {
            return;
        }
        if (count == 0) {
            *this = other;
            return;
        }

        const std::uint64_t combined_count = count + other.count;
        const long double delta = other.mean - mean;
        mean += delta * static_cast<long double>(other.count) /
                static_cast<long double>(combined_count);
        m2 += other.m2 +
              delta * delta * static_cast<long double>(count) *
                  static_cast<long double>(other.count) /
                  static_cast<long double>(combined_count);
        count = combined_count;
    }

    long double variance() const {
        if (count < 2) {
            return std::numeric_limits<long double>::quiet_NaN();
        }
        return m2 / static_cast<long double>(count - 1);
    }

    long double stderr() const {
        if (count < 2) {
            return std::numeric_limits<long double>::quiet_NaN();
        }
        return std::sqrt(variance() / static_cast<long double>(count));
    }
};

struct MCMCMCConfig {
    int order_min = 2;
    int order_max = 4;
    int envelope_order = 5;
    int normalization_order = 2;
    std::vector<double> lambdas;
    int lambda_warmup_steps = 0;
    int lambda_adapt_interval = 1000;
    double lambda_adapt_rate = 0.25;
    int production_steps = 100000;
    int chain_count = 1;
    std::uint64_t base_seed = 1;
    std::string seed_mode = "uniform";
    int histogram_learning_steps = 0;
    int histogram_tau_bins = 16;
    double histogram_uniform_mix = 0.05;
    double histogram_pseudo_count = 0.0;
    bool compare_uniform_seed = false;
    bool checkpoint_enabled = false;
    std::string checkpoint_dir = "mcmcmc_checkpoints";
    std::uint64_t checkpoint_interval_steps = 10'000'000ULL;
    std::string resume_mode = "off";
};

struct MCMCMCStepResult {
    std::uint64_t chosen_mask = 0;
    int chosen_order = 0;
    double local_normalization = 0.0;
};

struct MCMCMCMaskPlan {
    std::uint64_t subset_count = 0;
    std::uint64_t full_mask = 0;
    std::vector<std::uint64_t> sampled_masks;
    std::vector<int> sampled_orders;
    std::vector<double> inverse_binomial_by_order;
};

struct MCMCMCStepWorkspace {
    std::vector<double> fluxes;
    std::vector<double> selection_weights;
    std::vector<double> integrand_values;
    std::vector<double> conditional_probabilities;
    std::vector<double> order_signal;
    std::vector<UnitVertex> next_vertices;
};

struct MCMCMCChainResult {
    std::size_t chain_index = 0;
    std::uint64_t seed = 0;
    std::vector<double> final_lambdas;
    std::uint64_t warmup_proposed_steps = 0;
    std::uint64_t warmup_accepted_steps = 0;
    std::uint64_t production_proposed_steps = 0;
    std::uint64_t production_accepted_steps = 0;
    std::vector<std::uint64_t> warmup_chosen_counts;
    std::vector<std::uint64_t> production_chosen_counts;
    RunningStatsLongDouble warmup_retained_fraction_stats;
    RunningStatsLongDouble production_retained_fraction_stats;
    std::vector<RunningStatsLongDouble> order_signal_stats;
};

struct MCMCMCSummary {
    MCMCMCConfig config;
    std::vector<MCMCMCChainResult> chains;
    std::vector<RunningStatsLongDouble> cross_chain_order_signal_stats;
    std::map<int, long double> ratios_to_normalization;
    bool completed = true;
    bool checkpointed = false;
};

struct PlainMCBenchmarkSummary {
    int order_min = 2;
    int order_max = 4;
    int normalization_order = 2;
    int samples_per_order = 0;
    std::vector<RunningStatsLongDouble> order_integral_stats;
    std::map<int, long double> ratios_to_normalization;
};

inline std::uint64_t splitmix64_mcmcmc(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

inline std::uint64_t derive_mcmcmc_seed(std::uint64_t base_seed,
                                        std::uint64_t stream_tag,
                                        std::uint64_t sequence_index = 0) {
    std::uint64_t mixed = splitmix64_mcmcmc(base_seed ^ stream_tag);
    mixed ^= 0x9e3779b97f4a7c15ULL * (sequence_index + 1ULL);
    return splitmix64_mcmcmc(mixed);
}

inline double binomial_as_double(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    k = std::min(k, n - k);
    long double result = 1.0L;
    for (int i = 1; i <= k; ++i) {
        result *= static_cast<long double>(n - k + i);
        result /= static_cast<long double>(i);
    }
    return static_cast<double>(result);
}

inline UnitVertex sample_uniform_unit_vertex(std::mt19937_64& rng) {
    static thread_local std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    return UnitVertex{uniform01(rng), uniform01(rng), uniform01(rng), uniform01(rng)};
}

inline std::vector<UnitVertex> sample_uniform_vertex_set(int order, std::mt19937_64& rng) {
    std::vector<UnitVertex> vertices(static_cast<std::size_t>(order));
    for (auto& vertex : vertices) {
        vertex = sample_uniform_unit_vertex(rng);
    }
    return vertices;
}

inline std::uint64_t sample_weighted_mask(const std::vector<double>& weights,
                                          double total_weight,
                                          std::mt19937_64& rng) {
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::runtime_error("MCMCMC local normalization is zero or non-finite");
    }
    std::uniform_real_distribution<double> uniform(0.0, total_weight);
    const double threshold = uniform(rng);
    double cumulative = 0.0;
    std::uint64_t last_positive_mask = 0;
    for (std::size_t mask = 0; mask < weights.size(); ++mask) {
        if (!(weights[mask] > 0.0)) {
            continue;
        }
        if (std::isfinite(weights[mask])) {
            last_positive_mask = static_cast<std::uint64_t>(mask);
        }
        cumulative += weights[mask];
        if (cumulative > threshold) {
            return static_cast<std::uint64_t>(mask);
        }
    }
    return last_positive_mask;
}

inline std::uint64_t sample_weighted_candidate_mask(const std::vector<std::uint64_t>& masks,
                                                    const std::vector<double>& weights,
                                                    double total_weight,
                                                    std::mt19937_64& rng) {
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw std::runtime_error("MCMCMC local normalization is zero or non-finite");
    }
    std::uniform_real_distribution<double> uniform(0.0, total_weight);
    const double threshold = uniform(rng);
    double cumulative = 0.0;
    std::uint64_t last_positive_mask = 0;
    for (std::size_t index = 0; index < masks.size(); ++index) {
        const double weight = weights[index];
        if (!(weight > 0.0)) {
            continue;
        }
        if (std::isfinite(weight)) {
            last_positive_mask = masks[index];
        }
        cumulative += weight;
        if (cumulative > threshold) {
            return masks[index];
        }
    }
    return last_positive_mask;
}

template <typename RecursionRealT>
inline void compute_conditional_generation_probabilities_typed_into(
    const std::vector<UnitVertex>& vertices,
    const MCMCMCSeedKernel& seed_kernel,
    std::vector<RecursionRealT>& probabilities) {
    const int envelope_order = static_cast<int>(vertices.size());
    const std::uint64_t subset_count = std::uint64_t{1} << envelope_order;
    probabilities.assign(static_cast<std::size_t>(subset_count), RecursionRealT(1));
    if (seed_kernel.is_uniform()) {
        return;
    }

    thread_local std::vector<RecursionRealT> seed_density;
    seed_density.assign(static_cast<std::size_t>(envelope_order * envelope_order), RecursionRealT(0));
    for (int child = 0; child < envelope_order; ++child) {
        for (int parent = 0; parent < envelope_order; ++parent) {
            if (child == parent) {
                continue;
            }
            const double density_double =
                seed_kernel.density_child_given_parent(vertices[static_cast<std::size_t>(child)],
                                                       vertices[static_cast<std::size_t>(parent)]);
            if (density_double < 0.0 || !std::isfinite(density_double)) {
                throw std::runtime_error("MCMCMC seed density is negative or non-finite");
            }
            seed_density[static_cast<std::size_t>(child * envelope_order + parent)] =
                real_precision_cast<RecursionRealT>(density_double);
        }
    }

    thread_local std::vector<RecursionRealT> complement_density;
    complement_density.assign(
        static_cast<std::size_t>(subset_count) * static_cast<std::size_t>(envelope_order),
        RecursionRealT(0));
    for (int child = 0; child < envelope_order; ++child) {
        RecursionRealT density_sum = RecursionRealT(0);
        for (int parent = 0; parent < envelope_order; ++parent) {
            density_sum += seed_density[static_cast<std::size_t>(child * envelope_order + parent)];
        }
        complement_density[static_cast<std::size_t>(child)] = density_sum;
    }

    probabilities.assign(static_cast<std::size_t>(subset_count), RecursionRealT(0));
    probabilities[0] = RecursionRealT(1);
    const std::uint64_t full_mask = subset_count - 1U;
    for (std::uint64_t generated_mask = 1; generated_mask < full_mask; ++generated_mask) {
        const std::uint64_t removed_parent_bit = generated_mask & (~generated_mask + 1U);
        const int removed_parent = __builtin_ctzll(removed_parent_bit);
        const std::uint64_t previous_mask = generated_mask ^ removed_parent_bit;
        const std::size_t mask_offset =
            static_cast<std::size_t>(generated_mask) * static_cast<std::size_t>(envelope_order);
        const std::size_t previous_offset =
            static_cast<std::size_t>(previous_mask) * static_cast<std::size_t>(envelope_order);
        for (int child = 0; child < envelope_order; ++child) {
            RecursionRealT density =
                complement_density[previous_offset + static_cast<std::size_t>(child)] -
                seed_density[static_cast<std::size_t>(child * envelope_order + removed_parent)];
            if (density < RecursionRealT(0) && density > RecursionRealT(-1.0e-12)) {
                density = RecursionRealT(0);
            }
            complement_density[mask_offset + static_cast<std::size_t>(child)] = density;
        }

        const int generated_count = __builtin_popcountll(generated_mask);
        const int parent_count = envelope_order - generated_count;
        RecursionRealT sum = RecursionRealT(0);
        for (int child = 0; child < envelope_order; ++child) {
            const std::uint64_t child_bit = std::uint64_t{1} << child;
            if ((generated_mask & child_bit) == 0U) {
                continue;
            }
            sum += probabilities[static_cast<std::size_t>(generated_mask ^ child_bit)] *
                   complement_density[mask_offset + static_cast<std::size_t>(child)];
        }
        probabilities[static_cast<std::size_t>(generated_mask)] =
            sum / (RecursionRealT(generated_count) * RecursionRealT(parent_count));
    }
    probabilities[static_cast<std::size_t>(full_mask)] = RecursionRealT(0);
}

template <typename RecursionRealT>
inline void compute_conditional_generation_probabilities_as_double(
    const std::vector<UnitVertex>& vertices,
    const MCMCMCSeedKernel& seed_kernel,
    std::vector<double>& probabilities) {
    if constexpr (std::is_same_v<RecursionRealT, double>) {
        compute_conditional_generation_probabilities_typed_into<double>(
            vertices, seed_kernel, probabilities);
    } else {
        thread_local std::vector<RecursionRealT> typed_probabilities;
        compute_conditional_generation_probabilities_typed_into<RecursionRealT>(
            vertices, seed_kernel, typed_probabilities);
        probabilities.resize(typed_probabilities.size());
        for (std::size_t index = 0; index < typed_probabilities.size(); ++index) {
            probabilities[index] = real_precision_cast<double>(typed_probabilities[index]);
        }
    }
}

inline void compute_conditional_generation_probabilities_into(
    const std::vector<UnitVertex>& vertices,
    const MCMCMCSeedKernel& seed_kernel,
    std::vector<double>& probabilities) {
    switch (selected_integrand_real_type()) {
        case IntegrandRealType::Double:
            compute_conditional_generation_probabilities_as_double<double>(
                vertices, seed_kernel, probabilities);
            return;
        case IntegrandRealType::LongDouble:
            compute_conditional_generation_probabilities_as_double<long double>(
                vertices, seed_kernel, probabilities);
            return;
        case IntegrandRealType::DoubleDouble:
            compute_conditional_generation_probabilities_as_double<DoubleDouble>(
                vertices, seed_kernel, probabilities);
            return;
    }
    throw std::runtime_error("Unhandled integrand real type");
}

inline std::vector<double> compute_conditional_generation_probabilities(
    const std::vector<UnitVertex>& vertices,
    const MCMCMCSeedKernel& seed_kernel) {
    std::vector<double> probabilities;
    compute_conditional_generation_probabilities_into(vertices, seed_kernel, probabilities);
    return probabilities;
}

inline void create_next_vertices_from_seed_into(const std::vector<UnitVertex>& vertices,
                                                std::uint64_t chosen_mask,
                                                int envelope_order,
                                                const MCMCMCSeedKernel& seed_kernel,
                                                std::mt19937_64& rng,
                                                std::vector<UnitVertex>& next_vertices) {
    next_vertices.clear();
    next_vertices.reserve(static_cast<std::size_t>(envelope_order));
    for (int i = 0; i < envelope_order; ++i) {
        if ((chosen_mask & (std::uint64_t{1} << i)) != 0U) {
            next_vertices.push_back(vertices[static_cast<std::size_t>(i)]);
        }
    }
    while (static_cast<int>(next_vertices.size()) < envelope_order) {
        std::uniform_int_distribution<int> parent_distribution(
            0, static_cast<int>(next_vertices.size()) - 1);
        const UnitVertex& parent = next_vertices[static_cast<std::size_t>(parent_distribution(rng))];
        next_vertices.push_back(seed_kernel.sample_child_from_parent(parent, rng));
    }
}

inline std::vector<UnitVertex> create_next_vertices_from_seed(const std::vector<UnitVertex>& vertices,
                                                              std::uint64_t chosen_mask,
                                                              int envelope_order,
                                                              const MCMCMCSeedKernel& seed_kernel,
                                                              std::mt19937_64& rng) {
    std::vector<UnitVertex> next_vertices;
    create_next_vertices_from_seed_into(vertices, chosen_mask, envelope_order, seed_kernel, rng, next_vertices);
    return next_vertices;
}

inline MCMCMCMaskPlan make_mcmcmc_mask_plan(const MCMCMCConfig& config) {
    MCMCMCMaskPlan plan;
    plan.subset_count = std::uint64_t{1} << config.envelope_order;
    plan.full_mask = plan.subset_count - 1U;
    plan.inverse_binomial_by_order.assign(static_cast<std::size_t>(config.envelope_order + 1), 0.0);
    for (int order = config.order_min; order <= config.order_max; ++order) {
        plan.inverse_binomial_by_order[static_cast<std::size_t>(order)] =
            1.0 / binomial_as_double(config.envelope_order, order);
    }

    plan.sampled_masks.reserve(static_cast<std::size_t>(plan.subset_count));
    plan.sampled_orders.reserve(static_cast<std::size_t>(plan.subset_count));
    for (std::uint64_t mask = 1; mask < plan.subset_count; ++mask) {
        const int order = __builtin_popcountll(mask);
        if (order < config.order_min || order > config.order_max) {
            continue;
        }
        plan.sampled_masks.push_back(mask);
        plan.sampled_orders.push_back(order);
    }
    return plan;
}

inline MCMCMCConfig make_mcmcmc_config_from_globals() {
    MCMCMCConfig config;
    config.order_min = mcmcmc_order_min;
    config.order_max = mcmcmc_order_max;
    config.envelope_order = mcmcmc_envelope_order;
    config.normalization_order = mcmcmc_normalization_order;
    config.lambdas.assign(static_cast<std::size_t>(config.envelope_order + 1), 0.0);
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const int index = order - config.order_min;
        config.lambdas[static_cast<std::size_t>(order)] =
            mcmcmc_order_lambdas[static_cast<std::size_t>(index)];
    }
    config.chain_count = mcmcmc_chain_count;
    config.base_seed = mcmcmc_base_seed;
    config.lambda_warmup_steps = mcmcmc_lambda_warmup_steps;
    config.lambda_adapt_interval = mcmcmc_lambda_adapt_interval;
    config.lambda_adapt_rate = mcmcmc_lambda_adapt_rate;
    config.production_steps = mcmcmc_production_steps;
    config.seed_mode = mcmcmc_seed_mode;
    config.histogram_learning_steps = mcmcmc_seed_histogram_learning_steps;
    config.histogram_tau_bins = mcmcmc_seed_histogram_tau_bins;
    config.histogram_uniform_mix = mcmcmc_seed_histogram_uniform_mix;
    config.histogram_pseudo_count = mcmcmc_seed_histogram_pseudo_count;
    config.compare_uniform_seed = mcmcmc_compare_uniform_seed;
    config.checkpoint_enabled = mcmcmc_checkpoint_enabled;
    config.checkpoint_dir = mcmcmc_checkpoint_dir;
    config.checkpoint_interval_steps = mcmcmc_checkpoint_interval_steps;
    config.resume_mode = mcmcmc_resume_mode;
    return config;
}

inline void validate_mcmcmc_config(const MCMCMCConfig& config) {
    if (config.order_min < 1) {
        throw std::runtime_error("MCMCMC_ORDER_MIN must be at least 1 for the replica integrand");
    }
    if (config.order_max < config.order_min) {
        throw std::runtime_error("MCMCMC_ORDER_MAX must be >= MCMCMC_ORDER_MIN");
    }
    if (config.envelope_order <= config.order_max) {
        throw std::runtime_error("MCMCMC_ENVELOPE_ORDER must be larger than MCMCMC_ORDER_MAX");
    }
    if (config.envelope_order >= 63) {
        throw std::runtime_error("MCMCMC_ENVELOPE_ORDER is too large for 64-bit subset masks");
    }
    if (config.envelope_order > 48) {
        throw std::runtime_error("MCMCMC_ENVELOPE_ORDER must be <= 48 for the current principal-minor backend");
    }
    if (config.normalization_order < config.order_min ||
        config.normalization_order > config.order_max) {
        throw std::runtime_error("MCMCMC_NORMALIZATION_ORDER must lie inside the measured order range");
    }
    if (config.chain_count <= 0) {
        throw std::runtime_error("MCMCMC_CHAIN_COUNT must be positive");
    }
    if (config.lambda_warmup_steps < 0 || config.production_steps <= 0) {
        throw std::runtime_error("MCMCMC warmup must be nonnegative and production must be positive");
    }
    if (config.lambda_adapt_interval <= 0) {
        throw std::runtime_error("MCMCMC_LAMBDA_ADAPT_INTERVAL must be positive");
    }
    if (config.seed_mode != "uniform" && config.seed_mode != "histogram") {
        throw std::runtime_error("MCMCMC_SEED_MODE must be uniform or histogram");
    }
    if (config.resume_mode != "off" && config.resume_mode != "auto" &&
        config.resume_mode != "require") {
        throw std::runtime_error("MCMCMC_RESUME must be off, auto, or require");
    }
    if (config.checkpoint_enabled && config.checkpoint_interval_steps == 0) {
        throw std::runtime_error("MCMCMC_CHECKPOINT_INTERVAL_STEPS must be positive when checkpointing is enabled");
    }
    if (config.seed_mode == "histogram" && config.histogram_learning_steps <= 0) {
        throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_LEARNING_STEPS must be positive for histogram seed");
    }
    if (config.histogram_tau_bins <= 0) {
        throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_TAU_BINS must be positive");
    }
    if (config.histogram_uniform_mix < 0.0 || config.histogram_uniform_mix > 1.0) {
        throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_UNIFORM_MIX must lie in [0, 1]");
    }
    if (config.histogram_pseudo_count < 0.0 || !std::isfinite(config.histogram_pseudo_count)) {
        throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_PSEUDO_COUNT must be nonnegative and finite");
    }
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const double lambda = config.lambdas[static_cast<std::size_t>(order)];
        if (!(lambda > 0.0) || !std::isfinite(lambda)) {
            throw std::runtime_error("MCMCMC_LAMBDAS entries must be positive finite numbers");
        }
    }
}

inline MCMCMCStepResult run_mcmcmc_step(const std::vector<UnitVertex>& vertices,
                                        const MCMCMCConfig& config,
                                        const MCMCMCMaskPlan& mask_plan,
                                        const std::vector<double>& lambdas,
                                        const MCMCMCSeedKernel& seed_kernel,
                                        std::mt19937_64& rng,
                                        MCMCMCStepWorkspace& workspace) {
    const int envelope_order = config.envelope_order;
    const std::size_t candidate_count = mask_plan.sampled_masks.size();

    workspace.fluxes.resize(candidate_count);
    workspace.selection_weights.resize(candidate_count);
    workspace.order_signal.assign(static_cast<std::size_t>(envelope_order + 1), 0.0);

    evaluate_cdet_density_all_subset_integrands_double_into(vertices, workspace.integrand_values);

    const bool uniform_seed = seed_kernel.is_uniform();
    if (!uniform_seed) {
        compute_conditional_generation_probabilities_into(
            vertices, seed_kernel, workspace.conditional_probabilities);
    }

    double local_normalization = 0.0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        const std::uint64_t mask = mask_plan.sampled_masks[index];
        const int order = mask_plan.sampled_orders[index];
        const double integrand_value = workspace.integrand_values[static_cast<std::size_t>(mask)];
        const double conditional_probability =
            uniform_seed
                ? 1.0
                : workspace.conditional_probabilities[static_cast<std::size_t>(mask_plan.full_mask ^ mask)];
        const double flux =
            lambdas[static_cast<std::size_t>(order)] * integrand_value * conditional_probability;
        const double selection_weight =
            std::abs(flux) * mask_plan.inverse_binomial_by_order[static_cast<std::size_t>(order)];
        workspace.fluxes[index] = flux;
        workspace.selection_weights[index] = selection_weight;
        local_normalization += selection_weight;
    }

    const std::uint64_t chosen_mask =
        sample_weighted_candidate_mask(mask_plan.sampled_masks,
                                       workspace.selection_weights,
                                       local_normalization,
                                       rng);
    const int chosen_order = __builtin_popcountll(chosen_mask);

    for (std::size_t index = 0; index < candidate_count; ++index) {
        const int order = mask_plan.sampled_orders[index];
        const double time = workspace.fluxes[index] / local_normalization;
        workspace.order_signal[static_cast<std::size_t>(order)] +=
            time * mask_plan.inverse_binomial_by_order[static_cast<std::size_t>(order)];
    }

    create_next_vertices_from_seed_into(
        vertices, chosen_mask, envelope_order, seed_kernel, rng, workspace.next_vertices);

    return MCMCMCStepResult{chosen_mask, chosen_order, local_normalization};
}

inline void adapt_mcmcmc_lambdas(std::vector<double>& lambdas,
                                 const MCMCMCConfig& config,
                                 const std::vector<std::uint64_t>& block_counts) {
    std::uint64_t total = 0;
    for (int order = config.order_min; order <= config.order_max; ++order) {
        total += block_counts[static_cast<std::size_t>(order)];
    }
    if (total == 0) {
        return;
    }

    const int order_count = config.order_max - config.order_min + 1;
    const double target_fraction = 1.0 / static_cast<double>(order_count);
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const double actual_fraction =
            (static_cast<double>(block_counts[static_cast<std::size_t>(order)]) + 1.0) /
            (static_cast<double>(total) + static_cast<double>(order_count));
        const double update =
            std::exp(config.lambda_adapt_rate * std::log(target_fraction / actual_fraction));
        lambdas[static_cast<std::size_t>(order)] *= update;
    }
}

inline std::shared_ptr<const MCMCMCSeedKernel> learn_factorized_histogram_seed(
    const MCMCMCConfig& config,
    long double* total_pair_weight_out = nullptr) {
    FactorizedHistogramSeedSettings settings;
    settings.tau_bins = config.histogram_tau_bins;
    settings.N1 = N1;
    settings.N2 = N2;
    settings.uniform_mixing = config.histogram_uniform_mix;
    settings.pseudo_count = config.histogram_pseudo_count;
    FactorizedHistogramSeedAccumulator accumulator(settings);

    constexpr std::uint64_t stream_tag = 0x484953544F534545ULL;
    std::mt19937_64 rng(derive_mcmcmc_seed(config.base_seed, stream_tag));
    const std::uint64_t subset_count = std::uint64_t{1} << config.envelope_order;

    for (int step = 0; step < config.histogram_learning_steps; ++step) {
        const std::vector<UnitVertex> vertices = sample_uniform_vertex_set(config.envelope_order, rng);
        const std::vector<double> integrand_values =
            evaluate_cdet_density_all_subset_integrands_double(vertices);

        for (std::uint64_t mask = 1; mask < subset_count; ++mask) {
            const int order = __builtin_popcountll(mask);
            if (order < config.order_min || order > config.order_max || order < 2) {
                continue;
            }

            const double integrand_value = integrand_values[static_cast<std::size_t>(mask)];
            if (!std::isfinite(integrand_value)) {
                continue;
            }
            const long double subset_weight =
                static_cast<long double>(std::abs(integrand_value)) /
                static_cast<long double>(binomial_as_double(config.envelope_order, order));
            const long double pair_weight =
                subset_weight / static_cast<long double>(order * (order - 1));
            if (!(pair_weight > 0.0L)) {
                continue;
            }

            for (int parent = 0; parent < config.envelope_order; ++parent) {
                const std::uint64_t parent_bit = std::uint64_t{1} << parent;
                if ((mask & parent_bit) == 0U) {
                    continue;
                }
                for (int child = 0; child < config.envelope_order; ++child) {
                    if (child == parent) {
                        continue;
                    }
                    const std::uint64_t child_bit = std::uint64_t{1} << child;
                    if ((mask & child_bit) == 0U) {
                        continue;
                    }
                    accumulator.add_pair(vertices[static_cast<std::size_t>(parent)],
                                         vertices[static_cast<std::size_t>(child)],
                                         pair_weight);
                }
            }
        }
    }

    if (total_pair_weight_out != nullptr) {
        *total_pair_weight_out = accumulator.total_pair_weight();
    }
    return accumulator.finalize();
}

struct MCMCMCChainState {
    std::size_t chain_index = 0;
    std::uint64_t seed = 0;
    std::mt19937_64 rng;
    std::vector<UnitVertex> vertices;
    std::vector<double> lambdas;
    std::vector<std::uint64_t> warmup_block_counts;
    MCMCMCChainResult result;
};

inline void initialize_mcmcmc_chain_result(MCMCMCChainResult& result,
                                           const MCMCMCConfig& config,
                                           std::size_t chain_index,
                                           std::uint64_t seed,
                                           const std::vector<double>& lambdas) {
    result = {};
    result.chain_index = chain_index;
    result.seed = seed;
    result.final_lambdas = lambdas;
    result.warmup_chosen_counts.assign(static_cast<std::size_t>(config.envelope_order + 1), 0);
    result.production_chosen_counts.assign(static_cast<std::size_t>(config.envelope_order + 1), 0);
    result.order_signal_stats.assign(static_cast<std::size_t>(config.envelope_order + 1), {});
}

inline MCMCMCChainState make_initial_mcmcmc_chain_state(const MCMCMCConfig& config,
                                                        std::size_t chain_index,
                                                        std::uint64_t seed) {
    MCMCMCChainState state;
    state.chain_index = chain_index;
    state.seed = seed;
    state.rng.seed(seed);
    state.vertices = sample_uniform_vertex_set(config.envelope_order, state.rng);
    state.lambdas = config.lambdas;
    state.warmup_block_counts.assign(static_cast<std::size_t>(config.envelope_order + 1), 0);
    initialize_mcmcmc_chain_result(state.result, config, chain_index, seed, state.lambdas);
    return state;
}

inline std::filesystem::path mcmcmc_checkpoint_path(const MCMCMCConfig& config,
                                                    std::size_t chain_index) {
    std::ostringstream name;
    name << "chain_" << std::setw(6) << std::setfill('0') << chain_index << ".chk";
    return std::filesystem::path(config.checkpoint_dir) / name.str();
}

inline void write_mcmcmc_stats(std::ostream& out, const RunningStatsLongDouble& stats) {
    out << stats.count << ' ' << std::hexfloat << stats.mean << ' ' << stats.m2
        << std::defaultfloat << '\n';
}

inline long double parse_mcmcmc_checkpoint_float_token(const std::string& token,
                                                       const char* context) {
    errno = 0;
    char* end = nullptr;
    const long double value = std::strtold(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || errno == ERANGE) {
        throw std::runtime_error("Could not parse MCMCMC checkpoint floating value for " +
                                 std::string(context) + ": " + token);
    }
    return value;
}

inline long double read_mcmcmc_checkpoint_float_token(std::istream& in, const char* context) {
    std::string token;
    in >> token;
    if (!in) {
        throw std::runtime_error("Missing MCMCMC checkpoint floating value for " +
                                 std::string(context));
    }
    return parse_mcmcmc_checkpoint_float_token(token, context);
}

inline void read_mcmcmc_stats(std::istream& in, RunningStatsLongDouble& stats) {
    in >> stats.count;
    if (!in) {
        throw std::runtime_error("Malformed MCMCMC checkpoint running-stat count");
    }
    stats.mean = read_mcmcmc_checkpoint_float_token(in, "running-stat mean");
    stats.m2 = read_mcmcmc_checkpoint_float_token(in, "running-stat m2");
}

inline void write_mcmcmc_u64_vector(std::ostream& out,
                                    const char* label,
                                    const std::vector<std::uint64_t>& values) {
    out << label << ' ' << values.size();
    for (std::uint64_t value : values) {
        out << ' ' << value;
    }
    out << '\n';
}

inline void read_mcmcmc_u64_vector(std::istream& in,
                                   const char* expected_label,
                                   std::vector<std::uint64_t>& values) {
    std::string label;
    std::size_t size = 0;
    in >> label >> size;
    if (label != expected_label) {
        throw std::runtime_error("MCMCMC checkpoint expected label " +
                                 std::string(expected_label) + " but found " + label);
    }
    values.assign(size, 0);
    for (auto& value : values) {
        in >> value;
    }
}

inline void write_mcmcmc_double_vector(std::ostream& out,
                                       const char* label,
                                       const std::vector<double>& values) {
    out << label << ' ' << values.size() << std::hexfloat;
    for (double value : values) {
        out << ' ' << value;
    }
    out << std::defaultfloat << '\n';
}

inline void read_mcmcmc_double_vector(std::istream& in,
                                      const char* expected_label,
                                      std::vector<double>& values) {
    std::string label;
    std::size_t size = 0;
    in >> label >> size;
    if (label != expected_label) {
        throw std::runtime_error("MCMCMC checkpoint expected label " +
                                 std::string(expected_label) + " but found " + label);
    }
    values.assign(size, 0.0);
    for (auto& value : values) {
        value = static_cast<double>(
            read_mcmcmc_checkpoint_float_token(in, expected_label));
    }
}

inline void save_mcmcmc_chain_checkpoint(const MCMCMCConfig& config,
                                         const MCMCMCChainState& state) {
    if (!config.checkpoint_enabled) {
        return;
    }

    std::filesystem::create_directories(config.checkpoint_dir);
    const std::filesystem::path checkpoint_path =
        mcmcmc_checkpoint_path(config, state.chain_index);
    const std::filesystem::path tmp_path = checkpoint_path.string() + ".tmp";

    std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open MCMCMC checkpoint for writing: " +
                                 tmp_path.string());
    }
    out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    out << "CDET_MCMCMC_CHAIN_CHECKPOINT_V1\n";
    out << "chain_index " << state.chain_index << '\n';
    out << "seed " << state.seed << '\n';
    out << "order_min " << config.order_min << '\n';
    out << "order_max " << config.order_max << '\n';
    out << "envelope_order " << config.envelope_order << '\n';
    out << "normalization_order " << config.normalization_order << '\n';
    out << "lambda_warmup_steps " << config.lambda_warmup_steps << '\n';
    out << "production_steps " << config.production_steps << '\n';
    out << "seed_mode " << config.seed_mode << '\n';
    out << "rng_state\n" << state.rng << '\n';
    write_mcmcmc_double_vector(out, "lambdas", state.lambdas);
    write_mcmcmc_u64_vector(out, "warmup_block_counts", state.warmup_block_counts);
    out << "vertices " << state.vertices.size() << '\n' << std::hexfloat;
    for (const auto& vertex : state.vertices) {
        out << vertex.tau << ' ' << vertex.x << ' ' << vertex.y << ' '
            << vertex.replica << '\n';
    }
    out << std::defaultfloat;

    const auto& result = state.result;
    out << "warmup_proposed_steps " << result.warmup_proposed_steps << '\n';
    out << "warmup_accepted_steps " << result.warmup_accepted_steps << '\n';
    out << "production_proposed_steps " << result.production_proposed_steps << '\n';
    out << "production_accepted_steps " << result.production_accepted_steps << '\n';
    write_mcmcmc_u64_vector(out, "warmup_chosen_counts", result.warmup_chosen_counts);
    write_mcmcmc_u64_vector(out, "production_chosen_counts", result.production_chosen_counts);
    out << "warmup_retained_fraction_stats ";
    write_mcmcmc_stats(out, result.warmup_retained_fraction_stats);
    out << "production_retained_fraction_stats ";
    write_mcmcmc_stats(out, result.production_retained_fraction_stats);
    out << "order_signal_stats " << result.order_signal_stats.size() << '\n';
    for (const auto& stats : result.order_signal_stats) {
        write_mcmcmc_stats(out, stats);
    }
    out << "END\n";
    out.close();
    if (!out) {
        throw std::runtime_error("Failed while writing MCMCMC checkpoint: " +
                                 tmp_path.string());
    }
    std::filesystem::rename(tmp_path, checkpoint_path);
}

inline void expect_mcmcmc_checkpoint_label(std::istream& in, const char* expected_label) {
    std::string label;
    in >> label;
    if (label != expected_label) {
        throw std::runtime_error("MCMCMC checkpoint expected label " +
                                 std::string(expected_label) + " but found " + label);
    }
}

template <typename T>
inline T read_mcmcmc_labeled_scalar(std::istream& in, const char* expected_label) {
    expect_mcmcmc_checkpoint_label(in, expected_label);
    T value{};
    in >> value;
    return value;
}

inline bool load_mcmcmc_chain_checkpoint(const MCMCMCConfig& config,
                                         std::size_t chain_index,
                                         std::uint64_t seed,
                                         MCMCMCChainState& state) {
    const std::filesystem::path checkpoint_path =
        mcmcmc_checkpoint_path(config, chain_index);
    std::ifstream in(checkpoint_path);
    if (!in.is_open()) {
        return false;
    }

    std::string magic;
    in >> magic;
    if (magic != "CDET_MCMCMC_CHAIN_CHECKPOINT_V1") {
        throw std::runtime_error("Unsupported MCMCMC checkpoint format: " +
                                 checkpoint_path.string());
    }

    state.chain_index = read_mcmcmc_labeled_scalar<std::size_t>(in, "chain_index");
    state.seed = read_mcmcmc_labeled_scalar<std::uint64_t>(in, "seed");
    const int saved_order_min = read_mcmcmc_labeled_scalar<int>(in, "order_min");
    const int saved_order_max = read_mcmcmc_labeled_scalar<int>(in, "order_max");
    const int saved_envelope_order = read_mcmcmc_labeled_scalar<int>(in, "envelope_order");
    const int saved_normalization_order =
        read_mcmcmc_labeled_scalar<int>(in, "normalization_order");
    const int saved_warmup_steps =
        read_mcmcmc_labeled_scalar<int>(in, "lambda_warmup_steps");
    const int saved_production_steps =
        read_mcmcmc_labeled_scalar<int>(in, "production_steps");
    expect_mcmcmc_checkpoint_label(in, "seed_mode");
    std::string saved_seed_mode;
    in >> saved_seed_mode;

    if (state.chain_index != chain_index || state.seed != seed ||
        saved_order_min != config.order_min ||
        saved_order_max != config.order_max ||
        saved_envelope_order != config.envelope_order ||
        saved_normalization_order != config.normalization_order ||
        saved_warmup_steps != config.lambda_warmup_steps ||
        saved_production_steps > config.production_steps ||
        saved_seed_mode != config.seed_mode) {
        throw std::runtime_error("MCMCMC checkpoint does not match current run settings: " +
                                 checkpoint_path.string());
    }

    expect_mcmcmc_checkpoint_label(in, "rng_state");
    in >> state.rng;
    read_mcmcmc_double_vector(in, "lambdas", state.lambdas);
    read_mcmcmc_u64_vector(in, "warmup_block_counts", state.warmup_block_counts);

    expect_mcmcmc_checkpoint_label(in, "vertices");
    std::size_t vertex_count = 0;
    in >> vertex_count;
    state.vertices.assign(vertex_count, {});
    for (auto& vertex : state.vertices) {
        vertex.tau = static_cast<double>(
            read_mcmcmc_checkpoint_float_token(in, "vertex tau"));
        vertex.x = static_cast<double>(
            read_mcmcmc_checkpoint_float_token(in, "vertex x"));
        vertex.y = static_cast<double>(
            read_mcmcmc_checkpoint_float_token(in, "vertex y"));
        vertex.replica = static_cast<double>(
            read_mcmcmc_checkpoint_float_token(in, "vertex replica"));
    }

    initialize_mcmcmc_chain_result(state.result, config, chain_index, seed, state.lambdas);
    state.result.warmup_proposed_steps =
        read_mcmcmc_labeled_scalar<std::uint64_t>(in, "warmup_proposed_steps");
    state.result.warmup_accepted_steps =
        read_mcmcmc_labeled_scalar<std::uint64_t>(in, "warmup_accepted_steps");
    state.result.production_proposed_steps =
        read_mcmcmc_labeled_scalar<std::uint64_t>(in, "production_proposed_steps");
    state.result.production_accepted_steps =
        read_mcmcmc_labeled_scalar<std::uint64_t>(in, "production_accepted_steps");
    read_mcmcmc_u64_vector(in, "warmup_chosen_counts", state.result.warmup_chosen_counts);
    read_mcmcmc_u64_vector(in, "production_chosen_counts", state.result.production_chosen_counts);
    expect_mcmcmc_checkpoint_label(in, "warmup_retained_fraction_stats");
    read_mcmcmc_stats(in, state.result.warmup_retained_fraction_stats);
    expect_mcmcmc_checkpoint_label(in, "production_retained_fraction_stats");
    read_mcmcmc_stats(in, state.result.production_retained_fraction_stats);

    expect_mcmcmc_checkpoint_label(in, "order_signal_stats");
    std::size_t stats_count = 0;
    in >> stats_count;
    state.result.order_signal_stats.assign(stats_count, {});
    for (auto& stats : state.result.order_signal_stats) {
        read_mcmcmc_stats(in, stats);
    }

    std::string end_label;
    in >> end_label;
    if (end_label != "END") {
        throw std::runtime_error("Malformed MCMCMC checkpoint footer: " +
                                 checkpoint_path.string());
    }
    state.result.final_lambdas = state.lambdas;
    return true;
}

inline bool mcmcmc_has_pending_warmup_block(const std::vector<std::uint64_t>& block_counts) {
    return std::any_of(block_counts.begin(), block_counts.end(), [](std::uint64_t count) {
        return count != 0;
    });
}

inline std::uint64_t mcmcmc_completed_step_count(const MCMCMCChainState& state) {
    return state.result.warmup_proposed_steps + state.result.production_proposed_steps;
}

inline MCMCMCChainResult run_single_mcmcmc_chain(const MCMCMCConfig& config,
                                                std::size_t chain_index,
                                                std::uint64_t seed,
                                                std::shared_ptr<const MCMCMCSeedKernel> seed_kernel) {
    MCMCMCChainState state;
    bool resumed = false;
    if (config.checkpoint_enabled && config.resume_mode != "off") {
        resumed = load_mcmcmc_chain_checkpoint(config, chain_index, seed, state);
        if (!resumed && config.resume_mode == "require") {
            throw std::runtime_error("MCMCMC_RESUME=require but checkpoint is missing for chain " +
                                     std::to_string(chain_index));
        }
    }
    if (!resumed) {
        state = make_initial_mcmcmc_chain_state(config, chain_index, seed);
    }

    const MCMCMCMaskPlan mask_plan = make_mcmcmc_mask_plan(config);
    MCMCMCStepWorkspace workspace;
    auto next_checkpoint_step = [&]() {
        if (!config.checkpoint_enabled) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const std::uint64_t completed_steps = mcmcmc_completed_step_count(state);
        return ((completed_steps / config.checkpoint_interval_steps) + 1U) *
               config.checkpoint_interval_steps;
    }();

    auto checkpoint_if_due = [&](bool force) {
        if (!config.checkpoint_enabled) {
            return;
        }
        const std::uint64_t completed_steps = mcmcmc_completed_step_count(state);
        if (force || completed_steps >= next_checkpoint_step) {
            state.result.final_lambdas = state.lambdas;
            save_mcmcmc_chain_checkpoint(config, state);
            if (!force) {
                while (next_checkpoint_step <= completed_steps) {
                    next_checkpoint_step += config.checkpoint_interval_steps;
                }
            }
        }
    };

    while (state.result.warmup_proposed_steps <
           static_cast<std::uint64_t>(config.lambda_warmup_steps)) {
        const std::uint64_t next_step = state.result.warmup_proposed_steps + 1;
        MCMCMCStepResult step_result =
            run_mcmcmc_step(state.vertices, config, mask_plan, state.lambdas, *seed_kernel,
                            state.rng, workspace);
        state.vertices.swap(workspace.next_vertices);
        ++state.result.warmup_proposed_steps;
        ++state.result.warmup_accepted_steps;
        ++state.result.warmup_chosen_counts[static_cast<std::size_t>(step_result.chosen_order)];
        state.result.warmup_retained_fraction_stats.add(
            static_cast<long double>(step_result.chosen_order) /
            static_cast<long double>(config.envelope_order));
        ++state.warmup_block_counts[static_cast<std::size_t>(step_result.chosen_order)];
        if (next_step % static_cast<std::uint64_t>(config.lambda_adapt_interval) == 0) {
            adapt_mcmcmc_lambdas(state.lambdas, config, state.warmup_block_counts);
            std::fill(state.warmup_block_counts.begin(), state.warmup_block_counts.end(), 0);
        }
        checkpoint_if_due(false);
    }
    if (config.lambda_warmup_steps % config.lambda_adapt_interval != 0 &&
        mcmcmc_has_pending_warmup_block(state.warmup_block_counts)) {
        adapt_mcmcmc_lambdas(state.lambdas, config, state.warmup_block_counts);
        std::fill(state.warmup_block_counts.begin(), state.warmup_block_counts.end(), 0);
    }
    state.result.final_lambdas = state.lambdas;

    while (state.result.production_proposed_steps <
           static_cast<std::uint64_t>(config.production_steps)) {
        MCMCMCStepResult step_result =
            run_mcmcmc_step(state.vertices, config, mask_plan, state.lambdas, *seed_kernel,
                            state.rng, workspace);
        state.vertices.swap(workspace.next_vertices);
        ++state.result.production_proposed_steps;
        ++state.result.production_accepted_steps;
        ++state.result.production_chosen_counts[static_cast<std::size_t>(step_result.chosen_order)];
        state.result.production_retained_fraction_stats.add(
            static_cast<long double>(step_result.chosen_order) /
            static_cast<long double>(config.envelope_order));
        for (int order = config.order_min; order <= config.order_max; ++order) {
            state.result.order_signal_stats[static_cast<std::size_t>(order)].add(
                workspace.order_signal[static_cast<std::size_t>(order)] /
                state.lambdas[static_cast<std::size_t>(order)]);
        }
        checkpoint_if_due(false);
    }

    checkpoint_if_due(true);
    state.result.final_lambdas = state.lambdas;
    return state.result;
}

inline MCMCMCSummary run_mcmcmc_chains_parallel(const MCMCMCConfig& config,
                                                std::shared_ptr<const MCMCMCSeedKernel> seed_kernel) {
    validate_mcmcmc_config(config);
    if (!seed_kernel) {
        throw std::runtime_error("MCMCMC seed kernel is null");
    }

    MCMCMCSummary summary;
    summary.config = config;
    summary.cross_chain_order_signal_stats.assign(static_cast<std::size_t>(config.envelope_order + 1), {});

    std::vector<std::future<MCMCMCChainResult>> futures;
    futures.reserve(static_cast<std::size_t>(config.chain_count));
    constexpr std::uint64_t stream_tag = 0x4D434D434D435348ULL;
    for (int chain = 0; chain < config.chain_count; ++chain) {
        const std::uint64_t seed =
            derive_mcmcmc_seed(config.base_seed, stream_tag, static_cast<std::uint64_t>(chain));
        futures.emplace_back(std::async(std::launch::async, [&, chain, seed, seed_kernel]() {
            return run_single_mcmcmc_chain(config, static_cast<std::size_t>(chain), seed, seed_kernel);
        }));
    }

    summary.chains.reserve(static_cast<std::size_t>(config.chain_count));
    for (auto& future : futures) {
        summary.chains.push_back(future.get());
    }

    summary.completed = true;
    for (const auto& chain : summary.chains) {
        if (chain.warmup_proposed_steps < static_cast<std::uint64_t>(config.lambda_warmup_steps) ||
            chain.production_proposed_steps < static_cast<std::uint64_t>(config.production_steps)) {
            summary.completed = false;
        }
        for (int order = config.order_min; order <= config.order_max; ++order) {
            const auto& stats = chain.order_signal_stats[static_cast<std::size_t>(order)];
            // Treat each chain mean as one independent CLT sample; production
            // steps inside a chain are correlated and must not be pooled here.
            if (stats.count > 0) {
                summary.cross_chain_order_signal_stats[static_cast<std::size_t>(order)].add(stats.mean);
            }
        }
    }
    summary.checkpointed = !summary.completed;

    const int ref = config.normalization_order;
    const long double ref_mean =
        summary.cross_chain_order_signal_stats[static_cast<std::size_t>(ref)].mean;
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const long double mean =
            summary.cross_chain_order_signal_stats[static_cast<std::size_t>(order)].mean;
        summary.ratios_to_normalization[order] =
            (ref_mean == 0.0L ? std::numeric_limits<long double>::quiet_NaN() : mean / ref_mean);
    }

    return summary;
}

inline PlainMCBenchmarkSummary run_plain_mc_benchmark(int order_min,
                                                      int order_max,
                                                      int normalization_order,
                                                      int samples_per_order,
                                                      std::uint64_t seed) {
    PlainMCBenchmarkSummary summary;
    summary.order_min = order_min;
    summary.order_max = order_max;
    summary.normalization_order = normalization_order;
    summary.samples_per_order = samples_per_order;
    summary.order_integral_stats.assign(static_cast<std::size_t>(order_max + 1), {});
    if (samples_per_order <= 0) {
        return summary;
    }

    std::mt19937_64 rng(seed);
    for (int order = order_min; order <= order_max; ++order) {
        for (int sample = 0; sample < samples_per_order; ++sample) {
            const std::vector<UnitVertex> vertices = sample_uniform_vertex_set(order, rng);
            summary.order_integral_stats[static_cast<std::size_t>(order)].add(
                evaluate_cdet_density_integrand_double(vertices));
        }
    }

    const long double ref_mean =
        summary.order_integral_stats[static_cast<std::size_t>(normalization_order)].mean;
    for (int order = order_min; order <= order_max; ++order) {
        summary.ratios_to_normalization[order] =
            summary.order_integral_stats[static_cast<std::size_t>(order)].mean / ref_mean;
    }
    return summary;
}

#endif
