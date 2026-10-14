#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_DIAGONAL_GREEN_OBSERVABLE_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_DIAGONAL_GREEN_OBSERVABLE_H

#include "cdet/g_funcs.h"
#include "cdet/parameters.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/*
Computes the replica contribution to the diagonal lower-branch observable

    -U * sum_alpha sum_i integral_0^beta d tau
      [ (Gbar_up;alpha,alpha(i,tau;i,tau^+) + alpha_down / U)
        (Gbar_down;alpha,alpha(i,tau;i,tau^+) + alpha_up / U)
        - alpha_up * alpha_down / U^2 ].

This quantity is the first-order contribution coming from the twisted
two-replica partition function itself. To obtain the normalized derivative
of ln Z_2[A] = ln Tr_A(rho_A^2), one must subtract 2 times the corresponding
ordinary single-replica thermal contribution d ln Z_1 / d xi.

The current codebase stores two replica sectors, so the alpha-sum runs
over alpha in {0, 1}. The site sum runs over all lattice coordinates
(R_b1, R_b2). The diagonal limit (tau, tau^+) is evaluated from the already
built lower-branch Chebyshev table on the diagonal by setting

    tau_1 = tau_2 = tau,
    replica_alpha = replica_beta,
    (R1_b1, R1_b2) = (R2_b1, R2_b2).
*/

template <typename RealT>
inline RealT evaluate_diagonal_lower_branch_green_table(const PiecewiseChebyshevTable<RealT>& table,
                                                        int replica_index,
                                                        int site_b1,
                                                        int site_b2,
                                                        const RealT& tau) {
    return g_func_flattened(replica_index,
                            replica_index,
                            tau,
                            tau,
                            RealT(site_b1),
                            RealT(site_b2),
                            RealT(site_b1),
                            RealT(site_b2),
                            table,
                            real_precision_cast<RealT>(beta),
                            N1,
                            N2,
                            real_precision_cast<RealT>(branch_epsilon),
                            cheb_nu,
                            cheb_nv,
                            false);
}

template <typename RealT>
inline RealT chebyshev_uniform_integral_weight(int degree) {
    if (degree % 2 != 0) {
        return RealT(0);
    }
    if (degree == 0) {
        return RealT(2);
    }
    const RealT degree_t = RealT(degree);
    return RealT(2) / (RealT(1) - degree_t * degree_t);
}

template <typename RealT>
inline void lower_diagonal_line_coefficients(const PiecewiseChebyshevTable<RealT>& table,
                                             int replica_index,
                                             int site_b1,
                                             int site_b2,
                                             std::vector<RealT>& line_coeffs) {
    line_coeffs.assign(static_cast<std::size_t>(cheb_nu), RealT(0));
    const size_t base =
        coeff_replica_sitepair_offset(replica_index,
                                      replica_index,
                                      site_b1,
                                      site_b2,
                                      site_b1,
                                      site_b2,
                                      cheb_nu,
                                      cheb_nv,
                                      N1,
                                      N2);
    const RealT* coeffs = table.lower_coeffs.data() + base;
    for (int u_index = 0; u_index < cheb_nu; ++u_index) {
        RealT sum(0);
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
        for (int v_index = 0; v_index < cheb_nv; ++v_index) {
            sum += coeffs[row_offset + static_cast<size_t>(v_index)];
        }
        line_coeffs[static_cast<std::size_t>(u_index)] = sum;
    }
}

template <typename RealT>
inline void lower_diagonal_line_coefficients(const DiskBackedPiecewiseChebyshevTable<RealT>& table,
                                             int replica_index,
                                             int site_b1,
                                             int site_b2,
                                             std::vector<RealT>& line_coeffs) {
    line_coeffs.assign(static_cast<std::size_t>(cheb_nu), RealT(0));
    const size_t sitepair_index =
        replica_sitepair_index(replica_index,
                               replica_index,
                               site_b1,
                               site_b2,
                               site_b1,
                               site_b2,
                               N1,
                               N2);
    thread_local std::vector<RealT> coeffs;
    table.copy_coefficients(TriangleBranch::Lower, sitepair_index, coeffs);
    for (int u_index = 0; u_index < cheb_nu; ++u_index) {
        RealT sum(0);
        const size_t row_offset = static_cast<size_t>(u_index) * static_cast<size_t>(cheb_nv);
        for (int v_index = 0; v_index < cheb_nv; ++v_index) {
            sum += coeffs[row_offset + static_cast<size_t>(v_index)];
        }
        line_coeffs[static_cast<std::size_t>(u_index)] = sum;
    }
}

template <typename RealT>
inline RealT integrate_chebyshev_line(const std::vector<RealT>& line_coeffs,
                                      const RealT& beta_t) {
    RealT integral(0);
    for (int degree = 0; degree < cheb_nu; ++degree) {
        integral += line_coeffs[static_cast<std::size_t>(degree)] *
                    chebyshev_uniform_integral_weight<RealT>(degree);
    }
    return beta_t * integral / RealT(2);
}

template <typename RealT>
inline RealT integrate_chebyshev_line_product(const std::vector<RealT>& lhs_coeffs,
                                              const std::vector<RealT>& rhs_coeffs,
                                              const RealT& beta_t) {
    RealT integral(0);
    for (int lhs_degree = 0; lhs_degree < cheb_nu; ++lhs_degree) {
        const RealT lhs_value = lhs_coeffs[static_cast<std::size_t>(lhs_degree)];
        for (int rhs_degree = 0; rhs_degree < cheb_nu; ++rhs_degree) {
            const RealT rhs_value = rhs_coeffs[static_cast<std::size_t>(rhs_degree)];
            const int sum_degree = lhs_degree + rhs_degree;
            const int diff_degree = std::abs(lhs_degree - rhs_degree);
            const RealT product_weight =
                (chebyshev_uniform_integral_weight<RealT>(sum_degree) +
                 chebyshev_uniform_integral_weight<RealT>(diff_degree)) /
                RealT(2);
            integral += lhs_value * rhs_value * product_weight;
        }
    }
    return beta_t * integral / RealT(2);
}

template <typename RealT, typename UpTableT, typename DownTableT>
inline RealT integrate_diagonal_shifted_product_from_tables(
    const UpTableT& up_table,
    const DownTableT& down_table,
    int replica_index,
    int site_b1,
    int site_b2,
    const RealT& beta_t,
    const RealT& diagonal_up_shift,
    const RealT& diagonal_down_shift) {
    thread_local std::vector<RealT> up_line;
    thread_local std::vector<RealT> down_line;
    lower_diagonal_line_coefficients(up_table, replica_index, site_b1, site_b2, up_line);
    lower_diagonal_line_coefficients(down_table, replica_index, site_b1, site_b2, down_line);

    const RealT up_integral = integrate_chebyshev_line(up_line, beta_t);
    const RealT down_integral = integrate_chebyshev_line(down_line, beta_t);
    const RealT product_integral = integrate_chebyshev_line_product(up_line, down_line, beta_t);
    return product_integral +
           diagonal_up_shift * down_integral +
           diagonal_down_shift * up_integral;
}

inline DoubleDouble evaluate_diagonal_lower_branch_green_resident(const ResidentGreenFunctionTable& table,
                                                                  int replica_index,
                                                                  int site_b1,
                                                                  int site_b2,
                                                                  const DoubleDouble& tau) {
    return std::visit(
        [&](const auto& typed_table) -> DoubleDouble {
            using TableT = std::decay_t<decltype(typed_table)>;
            using RealT = typename TableT::value_type;
            const RealT tau_t = real_precision_cast<RealT>(tau);
            const RealT beta_t = real_precision_cast<RealT>(beta);
            const RealT branch_eps_t = real_precision_cast<RealT>(branch_epsilon);
            const RealT value =
                g_func_flattened(replica_index,
                                 replica_index,
                                 tau_t,
                                 tau_t,
                                 RealT(site_b1),
                                 RealT(site_b2),
                                 RealT(site_b1),
                                 RealT(site_b2),
                                 typed_table,
                                 beta_t,
                                 N1,
                                 N2,
                                 branch_eps_t,
                                 cheb_nu,
                                 cheb_nv,
                                 false);
            return real_precision_cast<DoubleDouble>(value);
        },
        table);
}

template <typename RealT>
inline const PiecewiseChebyshevTable<RealT>& require_diagonal_resident_table(
    const ResidentGreenFunctionTable& table,
    const char* table_name) {
    const auto* typed_table = std::get_if<PiecewiseChebyshevTable<RealT>>(&table);
    if (typed_table == nullptr) {
        throw std::runtime_error(
            std::string("Resident Green-function table ") + table_name +
            " type does not match the table selected for diagonal observable evaluation");
    }
    return *typed_table;
}

template <typename RealT, typename UpTableT, typename DownTableT>
inline RealT replica_log_partition_first_order_density_typed(
    const UpTableT& up_table,
    const DownTableT& down_table) {
    if (U == MP_Real(0)) {
        throw std::runtime_error("The diagonal observable is undefined for U = 0");
    }

    constexpr int replica_count = 2;

    const RealT beta_t = real_precision_cast<RealT>(beta);
    const RealT u_t = real_precision_cast<RealT>(U);
    const RealT alpha_up_t = real_precision_cast<RealT>(alpha_shift_up);
    const RealT alpha_down_t = real_precision_cast<RealT>(alpha_shift_down);
    const RealT diagonal_up_shift = alpha_down_t / u_t;
    const RealT diagonal_down_shift = alpha_up_t / u_t;

    RealT accumulated_sum(0);

    for (int replica_index = 0; replica_index < replica_count; ++replica_index) {
        for (int site_b2 = 0; site_b2 < N2; ++site_b2) {
            for (int site_b1 = 0; site_b1 < N1; ++site_b1) {
                accumulated_sum += integrate_diagonal_shifted_product_from_tables(
                    up_table,
                    down_table,
                    replica_index,
                    site_b1,
                    site_b2,
                    beta_t,
                    diagonal_up_shift,
                    diagonal_down_shift);
            }
        }
    }

    return -u_t * accumulated_sum / (RealT(N1) * RealT(N2));
}

template <typename RealT, typename UpTableT, typename DownTableT>
inline RealT ordinary_log_partition_first_order_density_typed(
    const UpTableT& up_table,
    const DownTableT& down_table) {
    if (U == MP_Real(0)) {
        throw std::runtime_error("The ordinary thermal observable is undefined for U = 0");
    }

    const RealT beta_t = real_precision_cast<RealT>(beta);
    const RealT u_t = real_precision_cast<RealT>(U);
    const RealT alpha_up_t = real_precision_cast<RealT>(alpha_shift_up);
    const RealT alpha_down_t = real_precision_cast<RealT>(alpha_shift_down);
    const RealT diagonal_up_shift = alpha_down_t / u_t;
    const RealT diagonal_down_shift = alpha_up_t / u_t;

    RealT site_sum(0);
    for (int site_b2 = 0; site_b2 < N2; ++site_b2) {
        for (int site_b1 = 0; site_b1 < N1; ++site_b1) {
            site_sum += integrate_diagonal_shifted_product_from_tables(
                up_table,
                down_table,
                0,
                site_b1,
                site_b2,
                beta_t,
                diagonal_up_shift,
                diagonal_down_shift);
        }
    }

    return -u_t * site_sum / (RealT(N1) * RealT(N2));
}

inline DoubleDouble normalized_log_partition_first_order_density_dd() {
    return std::visit(
        [&](const auto& twisted_up) -> DoubleDouble {
            using TableT = std::decay_t<decltype(twisted_up)>;
            using RealT = typename TableT::value_type;
            const auto& twisted_down =
                require_diagonal_resident_table<RealT>(g_table_down, "g_table_down");
            const auto& ordinary_up =
                require_diagonal_resident_table<RealT>(g_table_ordinary_up, "g_table_ordinary_up");
            const auto& ordinary_down =
                require_diagonal_resident_table<RealT>(g_table_ordinary_down, "g_table_ordinary_down");

            const RealT twisted =
                replica_log_partition_first_order_density_typed<RealT>(twisted_up, twisted_down);
            const RealT ordinary =
                ordinary_log_partition_first_order_density_typed<RealT>(ordinary_up, ordinary_down);
            return real_precision_cast<DoubleDouble>(twisted - RealT(2) * ordinary);
        },
        g_table_up);
}

template <typename RealT>
inline const DiskBackedPiecewiseChebyshevTable<RealT>& require_diagonal_disk_table(
    const DiskBackedGreenFunctionTable& table,
    const char* table_name) {
    const auto* typed_table = std::get_if<DiskBackedPiecewiseChebyshevTable<RealT>>(&table);
    if (typed_table == nullptr) {
        throw std::runtime_error(
            std::string("Disk-backed Green-function table ") + table_name +
            " type does not match the table selected for diagonal observable evaluation");
    }
    if (!g_disk_tables_initialized) {
        throw std::runtime_error("Disk-backed Green-function tables have not been initialized");
    }
    return *typed_table;
}

inline DoubleDouble normalized_log_partition_first_order_density_disk_dd() {
    return std::visit(
        [&](const auto& twisted_up) -> DoubleDouble {
            using TableT = std::decay_t<decltype(twisted_up)>;
            using RealT = typename TableT::value_type;
            const auto& twisted_down =
                require_diagonal_disk_table<RealT>(g_disk_table_down, "g_disk_table_down");
            const auto& ordinary_up =
                require_diagonal_disk_table<RealT>(g_disk_table_ordinary_up, "g_disk_table_ordinary_up");
            const auto& ordinary_down =
                require_diagonal_disk_table<RealT>(g_disk_table_ordinary_down, "g_disk_table_ordinary_down");

            const RealT twisted =
                replica_log_partition_first_order_density_typed<RealT>(twisted_up, twisted_down);
            const RealT ordinary =
                ordinary_log_partition_first_order_density_typed<RealT>(ordinary_up, ordinary_down);
            return real_precision_cast<DoubleDouble>(twisted - RealT(2) * ordinary);
        },
        g_disk_table_up);
}

inline DoubleDouble diagonal_observable_density_dd() {
    // The deterministic order-1 value used by MCMCMC is already the normalized
    // coefficient of ln Z_2[A], not the twisted coefficient alone.
    if (mcmcmc_table_backend == "disk_lru") {
        if (g_disk_diagonal_order1_value_available) {
            return g_disk_diagonal_order1_value;
        }
        return normalized_log_partition_first_order_density_disk_dd();
    }
    return normalized_log_partition_first_order_density_dd();
}

#endif
