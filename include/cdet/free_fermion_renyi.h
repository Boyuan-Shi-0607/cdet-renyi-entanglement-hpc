#ifndef CDET_FREE_FERMION_RENYI_H
#define CDET_FREE_FERMION_RENYI_H

#include <Eigen/Dense>

#include "cdet/parameters.h"
#include "cdet/replica_free_greens.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cdet_free {

struct FreeRenyiResult {
    int n_sites = 0;
    int subregion_sites = 0;
    int boundary_edges = 0;
    double renyi_order = 2.0;
    double mu_eff_up = 0.0;
    double mu_eff_down = 0.0;
    double entropy_up = 0.0;
    double entropy_down = 0.0;
    double entropy_total = 0.0;
};

inline double stable_fermi_dirac(double beta_value, double energy_minus_mu) {
    const double x = beta_value * energy_minus_mu;
    if (x >= 0.0) {
        const double exp_neg = std::exp(-x);
        return exp_neg / (1.0 + exp_neg);
    }

    const double exp_pos = std::exp(x);
    return 1.0 / (1.0 + exp_pos);
}

inline double renyi_entropy_from_correlation_eigenvalues(const Eigen::VectorXd& nu,
                                                         double renyi_order) {
    if (renyi_order <= 0.0) {
        throw std::runtime_error("Free-system Renyi order must be positive");
    }

    double entropy = 0.0;
    if (std::abs(renyi_order - 1.0) < 1.0e-12) {
        for (int i = 0; i < nu.size(); ++i) {
            const double p = std::clamp(nu(i), 1.0e-300, 1.0);
            const double q = std::clamp(1.0 - nu(i), 1.0e-300, 1.0);
            entropy += -(p * std::log(p) + q * std::log(q));
        }
        return entropy;
    }

    for (int i = 0; i < nu.size(); ++i) {
        const double p = std::clamp(nu(i), 0.0, 1.0);
        const double q = 1.0 - p;
        const double weight = std::max(std::pow(p, renyi_order) + std::pow(q, renyi_order),
                                       1.0e-300);
        entropy += std::log(weight) / (1.0 - renyi_order);
    }
    return entropy;
}

inline std::vector<int> subregion_site_indices(const std::vector<int>& mask) {
    std::vector<int> sites;
    sites.reserve(mask.size());
    for (int site = 0; site < static_cast<int>(mask.size()); ++site) {
        if (mask[static_cast<std::size_t>(site)] != 0) {
            sites.push_back(site);
        }
    }
    return sites;
}

inline int free_boundary_edges(const std::vector<int>& mask, int n1, int n2, bool use_pbc) {
    int total = 0;
    for (int y = 0; y < n2; ++y) {
        for (int x = 0; x < n1; ++x) {
            const bool here = mask[static_cast<std::size_t>(::site_index(x, y, n1, n2))] != 0;

            if (use_pbc || x + 1 < n1) {
                const int x_next = (x + 1) % n1;
                const bool there =
                    mask[static_cast<std::size_t>(::site_index(x_next, y, n1, n2))] != 0;
                total += here != there ? 1 : 0;
            }
            if (use_pbc || y + 1 < n2) {
                const int y_next = (y + 1) % n2;
                const bool there =
                    mask[static_cast<std::size_t>(::site_index(x, y_next, n1, n2))] != 0;
                total += here != there ? 1 : 0;
            }
        }
    }
    return total;
}

inline double pbc_single_spin_renyi_entropy(int n1,
                                            int n2,
                                            double beta_value,
                                            double mu_eff,
                                            double hopping_t,
                                            const std::vector<int>& a_sites,
                                            double renyi_order) {
    const int a_count = static_cast<int>(a_sites.size());
    if (a_count == 0) {
        return 0.0;
    }

    constexpr std::complex<double> imaginary_unit(0.0, 1.0);
    const double pi = std::acos(-1.0);
    const double inv_n = 1.0 / static_cast<double>(n1 * n2);

    Eigen::MatrixXcd corr_a = Eigen::MatrixXcd::Zero(a_count, a_count);
    for (int row = 0; row < a_count; ++row) {
        const int site_i = a_sites[static_cast<std::size_t>(row)];
        const int x_i = site_i % n1;
        const int y_i = site_i / n1;
        for (int col = 0; col <= row; ++col) {
            const int site_j = a_sites[static_cast<std::size_t>(col)];
            const int x_j = site_j % n1;
            const int y_j = site_j / n1;
            const int dx = x_i - x_j;
            const int dy = y_i - y_j;

            std::complex<double> value(0.0, 0.0);
            for (int ky_index = 0; ky_index < n2; ++ky_index) {
                const double ky = 2.0 * pi * static_cast<double>(ky_index) / static_cast<double>(n2);
                for (int kx_index = 0; kx_index < n1; ++kx_index) {
                    const double kx = 2.0 * pi * static_cast<double>(kx_index) / static_cast<double>(n1);
                    const double epsilon =
                        -2.0 * hopping_t * (std::cos(kx) + std::cos(ky));
                    const double occupation =
                        stable_fermi_dirac(beta_value, epsilon - mu_eff);
                    const double phase = kx * static_cast<double>(dx) +
                                         ky * static_cast<double>(dy);
                    value += occupation * std::exp(imaginary_unit * phase);
                }
            }
            value *= inv_n;
            corr_a(row, col) = value;
            if (row != col) {
                corr_a(col, row) = std::conj(value);
            }
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(corr_a);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Free-system PBC correlation diagonalization failed");
    }
    return renyi_entropy_from_correlation_eigenvalues(solver.eigenvalues(), renyi_order);
}

inline Eigen::MatrixXd build_open_one_body_hopping(int n1, int n2, double hopping_t, bool use_pbc) {
    const int n_sites = n1 * n2;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n_sites, n_sites);

    auto add_bond = [&](int x0, int y0, int x1, int y1) {
        const int i = ::site_index(x0, y0, n1, n2);
        const int j = ::site_index(x1, y1, n1, n2);
        h(i, j) += -hopping_t;
        h(j, i) += -hopping_t;
    };

    for (int y = 0; y < n2; ++y) {
        for (int x = 0; x < n1; ++x) {
            if (use_pbc || x + 1 < n1) {
                add_bond(x, y, (x + 1) % n1, y);
            }
            if (use_pbc || y + 1 < n2) {
                add_bond(x, y, x, (y + 1) % n2);
            }
        }
    }
    return h;
}

inline double dense_single_spin_renyi_entropy(int n1,
                                              int n2,
                                              double beta_value,
                                              double mu_eff,
                                              double hopping_t,
                                              bool use_pbc,
                                              const std::vector<int>& a_sites,
                                              double renyi_order) {
    const int a_count = static_cast<int>(a_sites.size());
    if (a_count == 0) {
        return 0.0;
    }

    const Eigen::MatrixXd h = build_open_one_body_hopping(n1, n2, hopping_t, use_pbc);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Free-system one-body diagonalization failed");
    }

    const Eigen::VectorXd& eps = solver.eigenvalues();
    const Eigen::MatrixXd& eigvecs = solver.eigenvectors();
    Eigen::VectorXd occupations(eps.size());
    for (int i = 0; i < eps.size(); ++i) {
        occupations(i) = stable_fermi_dirac(beta_value, eps(i) - mu_eff);
    }

    const Eigen::MatrixXd corr = eigvecs * occupations.asDiagonal() * eigvecs.transpose();
    Eigen::MatrixXd corr_a(a_count, a_count);
    for (int row = 0; row < a_count; ++row) {
        for (int col = 0; col < a_count; ++col) {
            corr_a(row, col) = corr(a_sites[static_cast<std::size_t>(row)],
                                    a_sites[static_cast<std::size_t>(col)]);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> corr_solver(corr_a);
    if (corr_solver.info() != Eigen::Success) {
        throw std::runtime_error("Free-system restricted correlation diagonalization failed");
    }
    return renyi_entropy_from_correlation_eigenvalues(corr_solver.eigenvalues(), renyi_order);
}

inline FreeRenyiResult compute_free_spinful_renyi_entropy(int n1,
                                                          int n2,
                                                          double beta_value,
                                                          double mu_value,
                                                          double alpha_up,
                                                          double alpha_down,
                                                          bool use_pbc,
                                                          ReplicaSubregionMask mask_type,
                                                          int origin_square_radius,
                                                          double renyi_order = 2.0,
                                                          double hopping_t = 1.0) {
    if (n1 <= 0 || n2 <= 0) {
        throw std::runtime_error("Free-system Renyi requires positive N1 and N2");
    }

    const std::vector<int> mask =
        ::build_subregion_mask(n1, n2, mask_type, origin_square_radius);
    const std::vector<int> a_sites = subregion_site_indices(mask);

    FreeRenyiResult result;
    result.n_sites = n1 * n2;
    result.subregion_sites = static_cast<int>(a_sites.size());
    result.boundary_edges = free_boundary_edges(mask, n1, n2, use_pbc);
    result.renyi_order = renyi_order;
    result.mu_eff_up = mu_value - alpha_up;
    result.mu_eff_down = mu_value - alpha_down;

    if (use_pbc) {
        result.entropy_up = pbc_single_spin_renyi_entropy(
            n1, n2, beta_value, result.mu_eff_up, hopping_t, a_sites, renyi_order);
        result.entropy_down = pbc_single_spin_renyi_entropy(
            n1, n2, beta_value, result.mu_eff_down, hopping_t, a_sites, renyi_order);
    } else {
        result.entropy_up = dense_single_spin_renyi_entropy(
            n1, n2, beta_value, result.mu_eff_up, hopping_t, use_pbc, a_sites, renyi_order);
        result.entropy_down = dense_single_spin_renyi_entropy(
            n1, n2, beta_value, result.mu_eff_down, hopping_t, use_pbc, a_sites, renyi_order);
    }
    result.entropy_total = result.entropy_up + result.entropy_down;
    return result;
}

inline FreeRenyiResult compute_free_spinful_renyi_entropy_from_cdet_globals() {
    const ReplicaSubregionMask mask_type = parse_subregion_mask(subregion_mask_name);
    const int origin_square_radius =
        mask_type == ReplicaSubregionMask::OriginSquare ? subregion_square_radius : 0;
    return compute_free_spinful_renyi_entropy(N1,
                                              N2,
                                              beta.convert_to<double>(),
                                              mu.convert_to<double>(),
                                              alpha_shift_up.convert_to<double>(),
                                              alpha_shift_down.convert_to<double>(),
                                              pbc,
                                              mask_type,
                                              origin_square_radius,
                                              2.0,
                                              1.0);
}

inline void print_free_spinful_renyi_entropy_from_cdet_globals() {
    const auto start = std::chrono::steady_clock::now();
    const FreeRenyiResult result = compute_free_spinful_renyi_entropy_from_cdet_globals();
    const double n_sites = static_cast<double>(result.n_sites);
    const double log_z2_norm_density = -result.entropy_total / n_sites;
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    std::cout << "\nFree system ln Z2 norm / N = "
              << std::showpos << std::setprecision(16) << log_z2_norm_density
              << std::noshowpos << '\n';
    std::cout << std::setprecision(15);
    std::cout << "  elapsed time = " << elapsed.count() << " s\n";
}

}  // namespace cdet_free

#endif
