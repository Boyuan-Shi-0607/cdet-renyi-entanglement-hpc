#ifndef CDET_MCMCMC_SEED_H
#define CDET_MCMCMC_SEED_H

#include "cdet/cdet_integrand_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class MCMCMCSeedKernel {
public:
    virtual ~MCMCMCSeedKernel() = default;
    virtual std::string name() const = 0;
    virtual bool is_uniform() const {
        return false;
    }
    virtual UnitVertex sample_child_from_parent(const UnitVertex& parent,
                                                std::mt19937_64& rng) const = 0;
    virtual double density_child_given_parent(const UnitVertex& child,
                                              const UnitVertex& parent) const = 0;
};

inline int unit_coordinate_to_index(double value, int extent) {
    int index = static_cast<int>(std::floor(value * static_cast<double>(extent)));
    if (index < 0) {
        return 0;
    }
    if (index >= extent) {
        return extent - 1;
    }
    return index;
}

class UniformMCMCMCSeed final : public MCMCMCSeedKernel {
public:
    std::string name() const override {
        return "uniform";
    }

    bool is_uniform() const override {
        return true;
    }

    UnitVertex sample_child_from_parent(const UnitVertex& /*parent*/,
                                        std::mt19937_64& rng) const override {
        static thread_local std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        return UnitVertex{uniform01(rng), uniform01(rng), uniform01(rng), uniform01(rng)};
    }

    double density_child_given_parent(const UnitVertex& /*child*/,
                                      const UnitVertex& /*parent*/) const override {
        return 1.0;
    }
};

struct FactorizedHistogramSeedSettings {
    int tau_bins = 16;
    int N1 = 1;
    int N2 = 1;
    double uniform_mixing = 0.05;
    double pseudo_count = 0.0;
};

class FactorizedHistogramSeed final : public MCMCMCSeedKernel {
public:
    FactorizedHistogramSeed(FactorizedHistogramSeedSettings settings,
                            std::vector<double> tau_prob,
                            std::vector<double> space_prob,
                            std::vector<double> replica_prob)
        : settings_(settings),
          site_count_(settings.N1 * settings.N2),
          tau_prob_(std::move(tau_prob)),
          space_prob_(std::move(space_prob)),
          replica_prob_(std::move(replica_prob)),
          tau_cdf_(tau_prob_.size(), 0.0),
          space_cdf_(space_prob_.size(), 0.0),
          replica_cdf_(replica_prob_.size(), 0.0) {
        if (settings_.tau_bins <= 0 || settings_.N1 <= 0 || settings_.N2 <= 0) {
            throw std::runtime_error("Histogram seed dimensions must be positive");
        }
        if (tau_prob_.size() != static_cast<std::size_t>(settings_.tau_bins * settings_.tau_bins)) {
            throw std::runtime_error("Invalid tau histogram size");
        }
        if (space_prob_.size() != static_cast<std::size_t>(site_count_ * site_count_)) {
            throw std::runtime_error("Invalid space histogram size");
        }
        if (replica_prob_.size() != 4U) {
            throw std::runtime_error("Invalid replica histogram size");
        }
        build_cdfs(tau_prob_, settings_.tau_bins, tau_cdf_);
        build_cdfs(space_prob_, site_count_, space_cdf_);
        build_cdfs(replica_prob_, 2, replica_cdf_);
    }

    std::string name() const override {
        return "factorized_histogram";
    }

    UnitVertex sample_child_from_parent(const UnitVertex& parent,
                                        std::mt19937_64& rng) const override {
        static thread_local std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        const int parent_tau = tau_bin(parent.tau);
        const int child_tau = sample_from_row(tau_cdf_, settings_.tau_bins, parent_tau, rng);

        const int parent_site = site_index(parent);
        const int child_site = sample_from_row(space_cdf_, site_count_, parent_site, rng);
        const int child_x = child_site / settings_.N2;
        const int child_y = child_site - child_x * settings_.N2;

        const int parent_replica = replica_index(parent);
        const int child_replica = sample_from_row(replica_cdf_, 2, parent_replica, rng);

        return UnitVertex{(static_cast<double>(child_tau) + uniform01(rng)) /
                              static_cast<double>(settings_.tau_bins),
                          (static_cast<double>(child_x) + uniform01(rng)) /
                              static_cast<double>(settings_.N1),
                          (static_cast<double>(child_y) + uniform01(rng)) /
                              static_cast<double>(settings_.N2),
                          (static_cast<double>(child_replica) + uniform01(rng)) / 2.0};
    }

    double density_child_given_parent(const UnitVertex& child,
                                      const UnitVertex& parent) const override {
        const int parent_tau = tau_bin(parent.tau);
        const int child_tau = tau_bin(child.tau);
        const double tau_density =
            tau_prob_[row_offset(parent_tau, settings_.tau_bins) + child_tau] *
            static_cast<double>(settings_.tau_bins);

        const int parent_site = site_index(parent);
        const int child_site = site_index(child);
        const double space_density =
            space_prob_[row_offset(parent_site, site_count_) + child_site] *
            static_cast<double>(site_count_);

        const int parent_replica = replica_index(parent);
        const int child_replica = replica_index(child);
        const double replica_density =
            replica_prob_[row_offset(parent_replica, 2) + child_replica] * 2.0;

        return tau_density * space_density * replica_density;
    }

    const FactorizedHistogramSeedSettings& settings() const {
        return settings_;
    }

    int site_count() const {
        return site_count_;
    }

    const std::vector<double>& tau_probabilities() const {
        return tau_prob_;
    }

    const std::vector<double>& space_probabilities() const {
        return space_prob_;
    }

    const std::vector<double>& replica_probabilities() const {
        return replica_prob_;
    }

private:
    static std::size_t row_offset(int row, int column_count) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(column_count);
    }

    int tau_bin(double tau_unit) const {
        return unit_coordinate_to_index(tau_unit, settings_.tau_bins);
    }

    int site_index(const UnitVertex& vertex) const {
        const int x = unit_coordinate_to_index(vertex.x, settings_.N1);
        const int y = unit_coordinate_to_index(vertex.y, settings_.N2);
        return x * settings_.N2 + y;
    }

    static int replica_index(const UnitVertex& vertex) {
        return unit_coordinate_to_index(vertex.replica, 2);
    }

    static void build_cdfs(const std::vector<double>& probabilities,
                           int column_count,
                           std::vector<double>& cdf) {
        const int row_count = static_cast<int>(probabilities.size()) / column_count;
        for (int row = 0; row < row_count; ++row) {
            double cumulative = 0.0;
            for (int column = 0; column < column_count; ++column) {
                const std::size_t index = row_offset(row, column_count) + static_cast<std::size_t>(column);
                cumulative += probabilities[index];
                cdf[index] = cumulative;
            }
            cdf[row_offset(row, column_count) + static_cast<std::size_t>(column_count - 1)] = 1.0;
        }
    }

    static int sample_from_row(const std::vector<double>& cdf,
                               int column_count,
                               int row,
                               std::mt19937_64& rng) {
        static thread_local std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        const double threshold = uniform01(rng);
        const auto begin = cdf.begin() + static_cast<std::ptrdiff_t>(row_offset(row, column_count));
        const auto end = begin + column_count;
        const auto it = std::lower_bound(begin, end, threshold);
        if (it == end) {
            return column_count - 1;
        }
        return static_cast<int>(it - begin);
    }

    FactorizedHistogramSeedSettings settings_;
    int site_count_ = 1;
    std::vector<double> tau_prob_;
    std::vector<double> space_prob_;
    std::vector<double> replica_prob_;
    std::vector<double> tau_cdf_;
    std::vector<double> space_cdf_;
    std::vector<double> replica_cdf_;
};

class FactorizedHistogramSeedAccumulator {
public:
    explicit FactorizedHistogramSeedAccumulator(FactorizedHistogramSeedSettings settings)
        : settings_(settings),
          site_count_(settings.N1 * settings.N2),
          tau_counts_(static_cast<std::size_t>(settings.tau_bins * settings.tau_bins), 0.0L),
          space_counts_(static_cast<std::size_t>(site_count_ * site_count_), 0.0L),
          replica_counts_(4U, 0.0L) {
        if (settings_.tau_bins <= 0 || settings_.N1 <= 0 || settings_.N2 <= 0) {
            throw std::runtime_error("Histogram seed dimensions must be positive");
        }
        if (settings_.uniform_mixing < 0.0 || settings_.uniform_mixing > 1.0) {
            throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_UNIFORM_MIX must lie in [0, 1]");
        }
        if (settings_.pseudo_count < 0.0 || !std::isfinite(settings_.pseudo_count)) {
            throw std::runtime_error("MCMCMC_SEED_HISTOGRAM_PSEUDO_COUNT must be nonnegative and finite");
        }
    }

    void add_pair(const UnitVertex& parent, const UnitVertex& child, long double weight) {
        if (!(weight > 0.0L) || !std::isfinite(static_cast<double>(weight))) {
            return;
        }
        tau_counts_[index(tau_bin(parent.tau), tau_bin(child.tau), settings_.tau_bins)] += weight;
        space_counts_[index(site_index(parent), site_index(child), site_count_)] += weight;
        replica_counts_[index(replica_index(parent), replica_index(child), 2)] += weight;
        total_pair_weight_ += weight;
    }

    long double total_pair_weight() const {
        return total_pair_weight_;
    }

    std::shared_ptr<const MCMCMCSeedKernel> finalize() const {
        return std::make_shared<FactorizedHistogramSeed>(
            settings_,
            normalize_rows(tau_counts_, settings_.tau_bins),
            normalize_rows(space_counts_, site_count_),
            normalize_rows(replica_counts_, 2));
    }

private:
    static std::size_t index(int row, int column, int column_count) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(column_count) +
               static_cast<std::size_t>(column);
    }

    int tau_bin(double tau_unit) const {
        return unit_coordinate_to_index(tau_unit, settings_.tau_bins);
    }

    int site_index(const UnitVertex& vertex) const {
        const int x = unit_coordinate_to_index(vertex.x, settings_.N1);
        const int y = unit_coordinate_to_index(vertex.y, settings_.N2);
        return x * settings_.N2 + y;
    }

    static int replica_index(const UnitVertex& vertex) {
        return unit_coordinate_to_index(vertex.replica, 2);
    }

    std::vector<double> normalize_rows(const std::vector<long double>& counts,
                                       int column_count) const {
        const int row_count = static_cast<int>(counts.size()) / column_count;
        std::vector<double> probabilities(counts.size(), 0.0);
        const long double uniform_probability = 1.0L / static_cast<long double>(column_count);
        const long double mix = static_cast<long double>(settings_.uniform_mixing);
        const long double pseudo = static_cast<long double>(settings_.pseudo_count);

        for (int row = 0; row < row_count; ++row) {
            long double row_sum = 0.0L;
            for (int column = 0; column < column_count; ++column) {
                row_sum += counts[index(row, column, column_count)];
            }
            const long double denominator =
                row_sum + pseudo * static_cast<long double>(column_count);

            for (int column = 0; column < column_count; ++column) {
                long double learned = uniform_probability;
                if (denominator > 0.0L) {
                    learned = (counts[index(row, column, column_count)] + pseudo) / denominator;
                }
                const long double probability =
                    (1.0L - mix) * learned + mix * uniform_probability;
                probabilities[index(row, column, column_count)] = static_cast<double>(probability);
            }

            double row_total = 0.0;
            for (int column = 0; column < column_count; ++column) {
                row_total += probabilities[index(row, column, column_count)];
            }
            if (row_total > 0.0 && std::isfinite(row_total)) {
                for (int column = 0; column < column_count; ++column) {
                    probabilities[index(row, column, column_count)] /= row_total;
                }
            }
        }
        return probabilities;
    }

    FactorizedHistogramSeedSettings settings_;
    int site_count_ = 1;
    std::vector<long double> tau_counts_;
    std::vector<long double> space_counts_;
    std::vector<long double> replica_counts_;
    long double total_pair_weight_ = 0.0L;
};

inline std::shared_ptr<const MCMCMCSeedKernel> make_uniform_mcmcmc_seed() {
    return std::make_shared<UniformMCMCMCSeed>();
}

#endif
