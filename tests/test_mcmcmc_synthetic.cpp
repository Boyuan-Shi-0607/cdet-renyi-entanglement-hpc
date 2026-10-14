#include "audit_test_utils.h"

#include "cdet/mcmcmc_sampler.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct SyntheticRunResult {
    std::vector<RunningStatsLongDouble> signals;
    std::vector<std::uint64_t> chosen_orders;
};

void synthetic_subset_integrands(const std::vector<UnitVertex>& vertices,
                                 std::vector<double>& values) {
    const std::size_t subset_count = std::size_t{1} << vertices.size();
    values.assign(subset_count, 1.0);
    for (std::size_t mask = 1; mask < subset_count; ++mask) {
        const std::size_t bit = mask & (~mask + 1U);
        const int vertex = __builtin_ctzll(static_cast<unsigned long long>(bit));
        values[mask] = values[mask ^ bit] *
                       (0.5 + vertices[static_cast<std::size_t>(vertex)].tau);
    }
}

SyntheticRunResult run_synthetic_chain(
    const MCMCMCConfig& config,
    const std::shared_ptr<const MCMCMCSeedKernel>& seed_kernel,
    std::uint64_t seed,
    int burn_in,
    int production_steps) {
    const MCMCMCMaskPlan plan = make_mcmcmc_mask_plan(config);
    std::mt19937_64 rng(seed);
    std::vector<UnitVertex> vertices = sample_uniform_vertex_set(config.envelope_order, rng);
    std::vector<double> integrands;
    std::vector<double> conditional;
    std::vector<double> weights(plan.sampled_masks.size(), 0.0);
    std::vector<double> fluxes(plan.sampled_masks.size(), 0.0);
    std::vector<UnitVertex> next_vertices;

    SyntheticRunResult result;
    result.signals.assign(static_cast<std::size_t>(config.envelope_order + 1), {});
    result.chosen_orders.assign(static_cast<std::size_t>(config.envelope_order + 1), 0);

    for (int step = -burn_in; step < production_steps; ++step) {
        synthetic_subset_integrands(vertices, integrands);
        if (!seed_kernel->is_uniform()) {
            compute_conditional_generation_probabilities_into(vertices, *seed_kernel, conditional);
        }

        double normalization = 0.0;
        for (std::size_t index = 0; index < plan.sampled_masks.size(); ++index) {
            const std::uint64_t mask = plan.sampled_masks[index];
            const int order = plan.sampled_orders[index];
            const double q = seed_kernel->is_uniform()
                                 ? 1.0
                                 : conditional[static_cast<std::size_t>(plan.full_mask ^ mask)];
            fluxes[index] = config.lambdas[static_cast<std::size_t>(order)] *
                            integrands[static_cast<std::size_t>(mask)] * q;
            weights[index] = std::abs(fluxes[index]) *
                             plan.inverse_binomial_by_order[static_cast<std::size_t>(order)];
            normalization += weights[index];
        }

        const std::uint64_t chosen_mask = sample_weighted_candidate_mask(
            plan.sampled_masks, weights, normalization, rng);
        const int chosen_order = __builtin_popcountll(chosen_mask);

        if (step >= 0) {
            std::vector<long double> signal(
                static_cast<std::size_t>(config.envelope_order + 1), 0.0L);
            for (std::size_t index = 0; index < plan.sampled_masks.size(); ++index) {
                const int order = plan.sampled_orders[index];
                signal[static_cast<std::size_t>(order)] +=
                    static_cast<long double>(fluxes[index] / normalization) *
                    static_cast<long double>(
                        plan.inverse_binomial_by_order[static_cast<std::size_t>(order)]) /
                    static_cast<long double>(config.lambdas[static_cast<std::size_t>(order)]);
            }
            for (int order = config.order_min; order <= config.order_max; ++order) {
                result.signals[static_cast<std::size_t>(order)].add(
                    signal[static_cast<std::size_t>(order)]);
            }
            ++result.chosen_orders[static_cast<std::size_t>(chosen_order)];
        }

        create_next_vertices_from_seed_into(
            vertices, chosen_mask, config.envelope_order, *seed_kernel, rng, next_vertices);
        vertices.swap(next_vertices);
    }
    return result;
}

std::shared_ptr<const MCMCMCSeedKernel> make_positive_histogram_seed() {
    FactorizedHistogramSeedSettings settings;
    settings.tau_bins = 4;
    settings.N1 = 2;
    settings.N2 = 2;
    settings.uniform_mixing = 0.2;

    std::vector<double> tau(16, 0.1);
    std::vector<double> space(16, 0.1);
    std::vector<double> replica(4, 0.2);
    for (int row = 0; row < 4; ++row) {
        tau[static_cast<std::size_t>(row * 4 + row)] = 0.7;
        space[static_cast<std::size_t>(row * 4 + row)] = 0.7;
    }
    replica[0] = 0.8;
    replica[3] = 0.8;
    return std::make_shared<FactorizedHistogramSeed>(
        settings, std::move(tau), std::move(space), std::move(replica));
}

void require_equal_integral_ratios(const SyntheticRunResult& result,
                                   const MCMCMCConfig& config,
                                   double tolerance,
                                   const std::string& label) {
    const long double reference = result.signals[1].mean;
    audit_require(reference > 0.0L, label + " has a non-positive reference signal");
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const double ratio = static_cast<double>(
            result.signals[static_cast<std::size_t>(order)].mean / reference);
        audit_require_close(ratio, 1.0, tolerance, 0.0,
                            label + " failed the known equal-integral ratio at order " +
                                std::to_string(order));
        audit_require(result.chosen_orders[static_cast<std::size_t>(order)] > 100,
                      label + " did not visit every retained vertex count");
    }
}

void run_mcmcmc_synthetic_tests() {
    N1 = 2;
    N2 = 2;
    integrand_real_name = "double";
    MCMCMCConfig config;
    config.order_min = 1;
    config.order_max = 4;
    config.envelope_order = 5;
    config.normalization_order = 1;
    config.lambdas.assign(6, 0.0);
    config.lambdas[1] = 0.7;
    config.lambdas[2] = 1.1;
    config.lambdas[3] = 1.6;
    config.lambdas[4] = 2.0;
    validate_mcmcmc_config(config);

    const bool long_mode = std::getenv("CDET_AUDIT_LONG") != nullptr;
    const int production_steps = long_mode ? 2'000'000 : 400'000;
    const int burn_in = long_mode ? 100'000 : 20'000;

    const auto uniform_result = run_synthetic_chain(
        config, make_uniform_mcmcmc_seed(), 0x7812bc91ULL, burn_in, production_steps);
    require_equal_integral_ratios(uniform_result, config,
                                  long_mode ? 0.015 : 0.04,
                                  "uniform-seed MCMCMC");

    const auto histogram_seed = make_positive_histogram_seed();
    const std::vector<UnitVertex> support_probe = {
        UnitVertex{0.01, 0.01, 0.01, 0.01},
        UnitVertex{0.29, 0.29, 0.29, 0.29},
        UnitVertex{0.51, 0.51, 0.51, 0.51},
        UnitVertex{0.79, 0.79, 0.79, 0.79},
        UnitVertex{0.99, 0.99, 0.99, 0.99},
    };
    const auto conditional =
        compute_conditional_generation_probabilities(support_probe, *histogram_seed);
    for (std::size_t mask = 1; mask + 1 < conditional.size(); ++mask) {
        audit_require(conditional[mask] > 0.0 && std::isfinite(conditional[mask]),
                      "positive histogram proposal lost support for a generated subset");
    }

    const auto histogram_result = run_synthetic_chain(
        config, histogram_seed, 0xd9024ad7ULL, burn_in, production_steps);
    require_equal_integral_ratios(histogram_result, config,
                                  long_mode ? 0.025 : 0.07,
                                  "histogram-seed MCMCMC");
}

int main() {
    return audit_test_main("mcmcmc_synthetic", run_mcmcmc_synthetic_tests);
}
