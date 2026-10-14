#ifndef REPLICA_FREE_GREENS_H
#define REPLICA_FREE_GREENS_H

#include "cdet/fermionic_graded_complex.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

enum class TriangleBranch {
    Upper,
    Lower,
};

enum class ReplicaSubregionMask {
    Empty,
    Checkerboard,
    FirstRow,
    FirstColumn,
    OriginSquare,
};

inline int site_index(int x, int y, int N1, int N2) {
    if (x < 0 || x >= N1 || y < 0 || y >= N2) {
        throw std::runtime_error("Site coordinate out of range");
    }
    return y * N1 + x;
}

inline std::string normalize_subregion_mask_name(std::string mask_name) {
    for (char& ch : mask_name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-') {
            ch = '_';
        }
    }
    return mask_name;
}

inline ReplicaSubregionMask parse_subregion_mask(const std::string& mask_name) {
    const std::string normalized = normalize_subregion_mask_name(mask_name);

    if (normalized == "empty" || normalized == "none" || normalized == "ordinary") {
        return ReplicaSubregionMask::Empty;
    }
    if (normalized == "checkerboard") {
        return ReplicaSubregionMask::Checkerboard;
    }
    if (normalized == "first_row") {
        return ReplicaSubregionMask::FirstRow;
    }
    if (normalized == "first_column") {
        return ReplicaSubregionMask::FirstColumn;
    }
    if (normalized == "origin_square" || normalized == "center_square" || normalized == "centered_square") {
        return ReplicaSubregionMask::OriginSquare;
    }

    throw std::runtime_error(
        "Unknown SUBREGION_MASK: " + mask_name +
        ". Supported values are empty, checkerboard, first_row, first_column, origin_square.");
}

inline const char* subregion_mask_to_string(ReplicaSubregionMask mask_type) {
    switch (mask_type) {
        case ReplicaSubregionMask::Empty:
            return "empty";
        case ReplicaSubregionMask::Checkerboard:
            return "checkerboard";
        case ReplicaSubregionMask::FirstRow:
            return "first_row";
        case ReplicaSubregionMask::FirstColumn:
            return "first_column";
        case ReplicaSubregionMask::OriginSquare:
            return "origin_square";
    }

    throw std::runtime_error("Unhandled ReplicaSubregionMask enum value");
}

inline bool coordinate_in_origin_square_interval(int coordinate, int extent, int radius) {
    if (radius <= 0) {
        return false;
    }
    if (2 * radius >= extent) {
        return true;
    }
    return coordinate < radius || coordinate >= extent - radius;
}

inline bool site_in_origin_square_subregion(int x, int y, int N1, int N2, int radius) {
    return coordinate_in_origin_square_interval(x, N1, radius) &&
           coordinate_in_origin_square_interval(y, N2, radius);
}

inline std::vector<int> build_subregion_mask(int N1,
                                             int N2,
                                             ReplicaSubregionMask mask_type,
                                             int origin_square_radius = 0) {
    std::vector<int> mask(static_cast<size_t>(N1 * N2), 0);
    for (int y = 0; y < N2; ++y) {
        for (int x = 0; x < N1; ++x) {
            bool in_subregion = false;
            switch (mask_type) {
                case ReplicaSubregionMask::Empty:
                    in_subregion = false;
                    break;
                case ReplicaSubregionMask::Checkerboard:
                    in_subregion = ((x + y) % 2 == 0);
                    break;
                case ReplicaSubregionMask::FirstRow:
                    in_subregion = (y == 0);
                    break;
                case ReplicaSubregionMask::FirstColumn:
                    in_subregion = (x == 0);
                    break;
                case ReplicaSubregionMask::OriginSquare:
                    in_subregion = site_in_origin_square_subregion(x, y, N1, N2, origin_square_radius);
                    break;
            }

            mask[static_cast<size_t>(site_index(x, y, N1, N2))] = in_subregion ? 1 : 0;
        }
    }
    return mask;
}

template <typename RealT>
inline Matrix<RealT> build_square_lattice_hamiltonian(int N1,
                                                      int N2,
                                                      const RealT& mu_s,
                                                      const RealT& hopping_t,
                                                      bool pbc) {
    const int n_sites = N1 * N2;
    Matrix<RealT> h(n_sites, n_sites, RealT(0));

    for (int y = 0; y < N2; ++y) {
        for (int x = 0; x < N1; ++x) {
            const int i = site_index(x, y, N1, N2);
            h(i, i) = -mu_s;

            std::vector<std::pair<int, int>> neighbors;
            if (x + 1 < N1) {
                neighbors.emplace_back(x + 1, y);
            } else if (pbc) {
                neighbors.emplace_back(0, y);
            }

            if (y + 1 < N2) {
                neighbors.emplace_back(x, y + 1);
            } else if (pbc) {
                neighbors.emplace_back(x, 0);
            }

            for (const auto& [xn, yn] : neighbors) {
                const int j = site_index(xn, yn, N1, N2);
                h(i, j) -= hopping_t;
                h(j, i) -= hopping_t;
            }
        }
    }

    return h;
}

template <typename RealT>
class ReplicaGreenCalculator {
public:
    using ComplexT = ComplexNumber<RealT>;

    ReplicaGreenCalculator(int N1,
                           int N2,
                           const RealT& beta_s,
                           const RealT& mu_s,
                           bool pbc,
                           ReplicaSubregionMask subregion_mask_type = ReplicaSubregionMask::Checkerboard,
                           int origin_square_radius = 0)
        : N1_(N1),
          N2_(N2),
          n_sites_(N1 * N2),
          beta_(beta_s),
          mu_(mu_s),
          hopping_t_(RealT(1)),
          pbc_(pbc),
          subregion_mask_type_(subregion_mask_type),
          origin_square_radius_(origin_square_radius),
          subregion_mask_(build_subregion_mask(N1, N2, subregion_mask_type, origin_square_radius)),
          h_(build_square_lattice_hamiltonian(N1, N2, mu_s, hopping_t_, pbc)) {
        const EigenDecomposition<RealT> eig = symmetric_eigen_decomposition(h_);
        eigenvalues_ = eig.eigenvalues;
        eigenvectors_ = eig.eigenvectors;
        initialize_blas_cache();

        for (int m = 0; m < 2; ++m) {
            sector_mixing_[static_cast<size_t>(m)] = build_sector_mixing_matrix(m);
        }
    }

    ReplicaGreenCalculator(int N1,
                           int N2,
                           const RealT& beta_s,
                           const RealT& mu_s,
                           bool pbc,
                           ReplicaSubregionMask subregion_mask_type,
                           int origin_square_radius,
                           RealVector<RealT> eigenvalues,
                           Matrix<RealT> eigenvectors,
                           std::array<Matrix<ComplexT>, 2> sector_mixing)
        : N1_(N1),
          N2_(N2),
          n_sites_(N1 * N2),
          beta_(beta_s),
          mu_(mu_s),
          hopping_t_(RealT(1)),
          pbc_(pbc),
          subregion_mask_type_(subregion_mask_type),
          origin_square_radius_(origin_square_radius),
          subregion_mask_(build_subregion_mask(N1, N2, subregion_mask_type, origin_square_radius)),
          h_(build_square_lattice_hamiltonian(N1, N2, mu_s, hopping_t_, pbc)),
          eigenvalues_(std::move(eigenvalues)),
          eigenvectors_(std::move(eigenvectors)),
          sector_mixing_(std::move(sector_mixing)) {
        validate_precomputed_dimensions();
        initialize_blas_cache();
    }

    int N1() const {
        return N1_;
    }

    int N2() const {
        return N2_;
    }

    int n_sites() const {
        return n_sites_;
    }

    const RealT& beta() const {
        return beta_;
    }

    bool pbc() const {
        return pbc_;
    }

    ReplicaSubregionMask subregion_mask_type() const {
        return subregion_mask_type_;
    }

    int origin_square_radius() const {
        return origin_square_radius_;
    }

    const std::vector<int>& subregion_mask() const {
        return subregion_mask_;
    }

    const Matrix<RealT>& hamiltonian() const {
        return h_;
    }

    const RealVector<RealT>& eigenvalues() const {
        return eigenvalues_;
    }

    const Matrix<RealT>& eigenvectors() const {
        return eigenvectors_;
    }

    const Matrix<ComplexT>& sector_mixing_matrix(int m) const {
        if (m != 0 && m != 1) {
            throw std::runtime_error("Replica sector index must be 0 or 1");
        }
        return sector_mixing_[static_cast<size_t>(m)];
    }

    std::array<Matrix<ComplexT>, 2> green_sector_matrices(const RealT& tau,
                                                          const RealT& tau_prime,
                                                          TriangleBranch branch) const {
        return {
            transform_eigen_kernel_to_site(stable_sector_kernel_eigenbasis(0, tau, tau_prime, branch)),
            transform_eigen_kernel_to_site(stable_sector_kernel_eigenbasis(1, tau, tau_prime, branch)),
        };
    }

    Matrix<ComplexT> green_sector_matrix(int m,
                                         const RealT& tau,
                                         const RealT& tau_prime,
                                         TriangleBranch branch) const {
        if (m != 0 && m != 1) {
            throw std::runtime_error("Replica sector index must be 0 or 1");
        }
        return transform_eigen_kernel_to_site(stable_sector_kernel_eigenbasis(m, tau, tau_prime, branch));
    }

    Matrix<RealT> green_replica_matrix(int replica_alpha,
                                       int replica_beta,
                                       const RealT& tau,
                                       const RealT& tau_prime,
                                       TriangleBranch branch) const {
        if (replica_alpha < 0 || replica_alpha > 1 || replica_beta < 0 || replica_beta > 1) {
            throw std::runtime_error("Replica indices must be 0 or 1");
        }

        const auto sector_matrices = green_sector_matrices(tau, tau_prime, branch);
        const Matrix<ComplexT>& g_m0 = sector_matrices[0];
        const Matrix<ComplexT>& g_m1 = sector_matrices[1];
        const ComplexT imag_unit = complex_i<RealT>();

        Matrix<RealT> result(n_sites_, n_sites_, RealT(0));
        for (int i = 0; i < n_sites_; ++i) {
            for (int j = 0; j < n_sites_; ++j) {
                ComplexT value;
                if (replica_alpha == replica_beta) {
                    value = (g_m0(i, j) + g_m1(i, j)) / ComplexT(RealT(2));
                } else if (replica_alpha == 0 && replica_beta == 1) {
                    value = imag_unit * (g_m0(i, j) - g_m1(i, j)) / ComplexT(RealT(2));
                } else {
                    value = -imag_unit * (g_m0(i, j) - g_m1(i, j)) / ComplexT(RealT(2));
                }
                result(i, j) = real_part_if_nearly_real(value);
            }
        }
        return result;
    }

    Matrix<RealT> ordinary_lower_equal_time_green_matrix() const {
        Matrix<RealT> kernel(n_sites_, n_sites_, RealT(0));
        for (int p = 0; p < n_sites_; ++p) {
            const RealT x = beta_ * eigenvalues_[static_cast<size_t>(p)];
            RealT occupation;
            if (x >= RealT(0)) {
                const RealT exp_minus_x = real_exp_value(RealT(-x));
                occupation = exp_minus_x / (RealT(1) + exp_minus_x);
            } else {
                const RealT exp_x = real_exp_value(x);
                occupation = RealT(1) / (RealT(1) + exp_x);
            }
            kernel(p, p) = -occupation;
        }
        return transform_real_eigen_kernel_to_site(kernel);
    }

private:
    void validate_precomputed_dimensions() const {
        if (static_cast<int>(eigenvalues_.size()) != n_sites_) {
            throw std::runtime_error("Cached eigenvalue count does not match lattice size");
        }
        if (eigenvectors_.rows() != n_sites_ || eigenvectors_.cols() != n_sites_) {
            throw std::runtime_error("Cached eigenvector matrix shape does not match lattice size");
        }
        for (int m = 0; m < 2; ++m) {
            const Matrix<ComplexT>& mixing = sector_mixing_[static_cast<size_t>(m)];
            if (mixing.rows() != n_sites_ || mixing.cols() != n_sites_) {
                throw std::runtime_error("Cached sector-mixing matrix shape does not match lattice size");
            }
        }
    }

    Matrix<ComplexT> build_sector_mixing_matrix(int m) const {
        if (m != 0 && m != 1) {
            throw std::runtime_error("Replica sector index must be 0 or 1");
        }

#ifdef CDET_USE_LAPACK
        if constexpr (std::is_same_v<RealT, double>) {
            Matrix<ComplexT> mixing = identity_matrix<ComplexT>(n_sites_, ComplexT(1));
            const ComplexT selected_phase = omega_inverse_phase(m, true);
            const ComplexT phase_delta = selected_phase - ComplexT(1);
            if (phase_delta == ComplexT(0)) {
                return mixing;
            }

            std::vector<int> selected_rows;
            selected_rows.reserve(static_cast<size_t>(n_sites_));
            for (int site = 0; site < n_sites_; ++site) {
                if (subregion_mask_[static_cast<size_t>(site)] == 1) {
                    selected_rows.push_back(site);
                }
            }
            if (selected_rows.empty()) {
                return mixing;
            }

            const Matrix<double> gram = blas_selected_row_gram(eigenvectors_, selected_rows);
            for (int p = 0; p < n_sites_; ++p) {
                for (int q = 0; q < n_sites_; ++q) {
                    mixing(p, q) += phase_delta * ComplexT(gram(p, q));
                }
            }
            return mixing;
        }
#endif

        std::vector<ComplexT> phases(static_cast<size_t>(n_sites_));
        for (int site = 0; site < n_sites_; ++site) {
            phases[static_cast<size_t>(site)] =
                omega_inverse_phase(m, subregion_mask_[static_cast<size_t>(site)] == 1);
        }

        Matrix<ComplexT> mixing(n_sites_, n_sites_, ComplexT(0));
        for (int p = 0; p < n_sites_; ++p) {
            for (int site = 0; site < n_sites_; ++site) {
                const RealT vip = eigenvectors_(site, p);
                if (vip == RealT(0)) {
                    continue;
                }
                const ComplexT weighted_left = ComplexT(vip) * phases[static_cast<size_t>(site)];
                for (int q = 0; q < n_sites_; ++q) {
                    mixing(p, q) += weighted_left * ComplexT(eigenvectors_(site, q));
                }
            }
        }
        return mixing;
    }

    Matrix<ComplexT> stable_sector_kernel_eigenbasis(int m,
                                                     const RealT& tau,
                                                     const RealT& tau_prime,
                                                     TriangleBranch branch) const {
        if (m != 0 && m != 1) {
            throw std::runtime_error("Replica sector index must be 0 or 1");
        }

        const RealT tau_fraction = tau / beta_;
        const RealT tau_prime_fraction = tau_prime / beta_;
        const bool upper = branch == TriangleBranch::Upper;
        const RealT theta = upper ? (tau_fraction + tau_prime_fraction) / RealT(2)
                                  : RealT(1) - (tau_fraction + tau_prime_fraction) / RealT(2);
        const Matrix<ComplexT>& mixing = sector_mixing_[static_cast<size_t>(m)];

        Matrix<ComplexT> middle(n_sites_, n_sites_, ComplexT(0));
        Matrix<ComplexT> rhs(n_sites_, n_sites_, ComplexT(0));
        std::vector<RealT> left(static_cast<size_t>(n_sites_), RealT(0));
        std::vector<RealT> row_middle_scale(static_cast<size_t>(n_sites_), RealT(0));
        std::vector<RealT> col_middle_scale(static_cast<size_t>(n_sites_), RealT(0));
        std::vector<RealT> diag_middle(static_cast<size_t>(n_sites_), RealT(0));

        for (int a = 0; a < n_sites_; ++a) {
            const RealT x = beta_ * eigenvalues_[static_cast<size_t>(a)];
            const RealT signed_x = upper ? x : -x;
            const RealT c = std::max(RealT(0), signed_x);

            RealT left_exponent;
            RealT right_exponent;
            if (upper) {
                left_exponent = (RealT(1) - tau_fraction) * x - (RealT(1) - theta) * c;
                right_exponent = tau_prime_fraction * x - theta * c;
            } else {
                left_exponent = tau_fraction * signed_x - (RealT(1) - theta) * c;
                right_exponent = (RealT(1) - tau_prime_fraction) * signed_x - theta * c;
            }

            left[static_cast<size_t>(a)] = real_exp_value(left_exponent);
            rhs(a, a) = ComplexT(real_exp_value(right_exponent));
            row_middle_scale[static_cast<size_t>(a)] = real_exp_value(RealT(-theta * c));
            col_middle_scale[static_cast<size_t>(a)] = real_exp_value(RealT(-(RealT(1) - theta) * c));
            diag_middle[static_cast<size_t>(a)] = real_exp_value(RealT(signed_x - c));
        }

        for (int a = 0; a < n_sites_; ++a) {
            const RealT row_scale = row_middle_scale[static_cast<size_t>(a)];
            for (int b = 0; b < n_sites_; ++b) {
                const RealT col_scale = col_middle_scale[static_cast<size_t>(b)];
                const ComplexT mixing_value = upper ? mixing(a, b) : complex_conj(mixing(b, a));
                middle(a, b) = ComplexT(row_scale) * mixing_value * ComplexT(col_scale);
            }

            middle(a, a) += ComplexT(diag_middle[static_cast<size_t>(a)]);
        }

        LinearSolveDiagnosticContext solve_context;
        solve_context.label = "stable_replica_green_kernel";
        solve_context.branch = upper ? "upper" : "lower";
        solve_context.sector = m;
        solve_context.tau_fraction = static_cast<double>(tau_fraction);
        solve_context.tau_prime_fraction = static_cast<double>(tau_prime_fraction);
        solve_context.rhs_is_diagonal = true;
        Matrix<ComplexT> solved = solve_matrix_system(middle, rhs, solve_context);
        Matrix<ComplexT> kernel(n_sites_, n_sites_, ComplexT(0));
        for (int a = 0; a < n_sites_; ++a) {
            const ComplexT row_scale = upper ? ComplexT(left[static_cast<size_t>(a)])
                                             : ComplexT(-left[static_cast<size_t>(a)]);
            for (int b = 0; b < n_sites_; ++b) {
                kernel(a, b) = row_scale * solved(a, b);
            }
        }

        return kernel;
    }

    Matrix<ComplexT> transform_eigen_kernel_to_site(const Matrix<ComplexT>& kernel) const {
#ifdef CDET_USE_LAPACK
        if constexpr (std::is_same_v<RealT, double>) {
            return transform_eigen_kernel_to_site_blas(kernel);
        }
#endif

        Matrix<ComplexT> temp(n_sites_, n_sites_, ComplexT(0));
        Matrix<ComplexT> result(n_sites_, n_sites_, ComplexT(0));

        for (int i = 0; i < n_sites_; ++i) {
            for (int p = 0; p < n_sites_; ++p) {
                const RealT vip = eigenvectors_(i, p);
                if (vip == RealT(0)) {
                    continue;
                }
                const ComplexT left = ComplexT(vip);
                for (int q = 0; q < n_sites_; ++q) {
                    temp(i, q) += left * kernel(p, q);
                }
            }
        }

        for (int i = 0; i < n_sites_; ++i) {
            for (int q = 0; q < n_sites_; ++q) {
                const ComplexT tiq = temp(i, q);
                if (tiq == ComplexT(0)) {
                    continue;
                }
                for (int j = 0; j < n_sites_; ++j) {
                    result(i, j) += tiq * ComplexT(eigenvectors_(j, q));
                }
            }
        }

        return result;
    }

#ifdef CDET_USE_LAPACK
    Matrix<ComplexT> transform_eigen_kernel_to_site_blas(const Matrix<ComplexT>& kernel) const {
        const int n = n_sites_;
        const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        if (eigenvectors_col_major_.size() != matrix_size) {
            throw std::runtime_error("Missing BLAS eigenvector cache for replica Green transform");
        }

        std::vector<double> kernel_real_col(matrix_size);
        std::vector<double> kernel_imag_col(matrix_size);
        std::vector<double> temp_real(matrix_size);
        std::vector<double> temp_imag(matrix_size);
        std::vector<double> result_real(matrix_size);
        std::vector<double> result_imag(matrix_size);

        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const size_t offset = static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n);
                kernel_real_col[offset] = kernel(row, col).real;
                kernel_imag_col[offset] = kernel(row, col).imag;
            }
        }

        char no_transpose = 'N';
        char transpose = 'T';
        int dim = n;
        double one = 1.0;
        double zero = 0.0;
        dgemm_(&no_transpose,
               &no_transpose,
               &dim,
               &dim,
               &dim,
               &one,
               eigenvectors_col_major_.data(),
               &dim,
               kernel_real_col.data(),
               &dim,
               &zero,
               temp_real.data(),
               &dim);
        dgemm_(&no_transpose,
               &no_transpose,
               &dim,
               &dim,
               &dim,
               &one,
               eigenvectors_col_major_.data(),
               &dim,
               kernel_imag_col.data(),
               &dim,
               &zero,
               temp_imag.data(),
               &dim);
        dgemm_(&no_transpose,
               &transpose,
               &dim,
               &dim,
               &dim,
               &one,
               temp_real.data(),
               &dim,
               eigenvectors_col_major_.data(),
               &dim,
               &zero,
               result_real.data(),
               &dim);
        dgemm_(&no_transpose,
               &transpose,
               &dim,
               &dim,
               &dim,
               &one,
               temp_imag.data(),
               &dim,
               eigenvectors_col_major_.data(),
               &dim,
               &zero,
               result_imag.data(),
               &dim);

        Matrix<ComplexT> result(n, n, ComplexT(0));
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const size_t offset = static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n);
                result(row, col) = ComplexT(result_real[offset], result_imag[offset]);
            }
        }
        return result;
    }
#endif

    void initialize_blas_cache() {
#ifdef CDET_USE_LAPACK
        if constexpr (std::is_same_v<RealT, double>) {
            eigenvectors_col_major_.resize(
                static_cast<size_t>(n_sites_) * static_cast<size_t>(n_sites_));
            for (int row = 0; row < n_sites_; ++row) {
                for (int col = 0; col < n_sites_; ++col) {
                    eigenvectors_col_major_[static_cast<size_t>(row) +
                                            static_cast<size_t>(col) * static_cast<size_t>(n_sites_)] =
                        eigenvectors_(row, col);
                }
            }
        }
#endif
    }

    Matrix<RealT> transform_real_eigen_kernel_to_site(const Matrix<RealT>& kernel) const {
        Matrix<RealT> temp(n_sites_, n_sites_, RealT(0));
        Matrix<RealT> result(n_sites_, n_sites_, RealT(0));

        for (int i = 0; i < n_sites_; ++i) {
            for (int p = 0; p < n_sites_; ++p) {
                const RealT vip = eigenvectors_(i, p);
                if (vip == RealT(0)) {
                    continue;
                }
                for (int q = 0; q < n_sites_; ++q) {
                    temp(i, q) += vip * kernel(p, q);
                }
            }
        }

        for (int i = 0; i < n_sites_; ++i) {
            for (int q = 0; q < n_sites_; ++q) {
                const RealT tiq = temp(i, q);
                if (tiq == RealT(0)) {
                    continue;
                }
                for (int j = 0; j < n_sites_; ++j) {
                    result(i, j) += tiq * eigenvectors_(j, q);
                }
            }
        }

        return result;
    }

    ComplexT omega_inverse_phase(int m, bool in_subregion) const {
        if (!in_subregion) {
            return ComplexT(1);
        }

        if (m == 0) {
            return ComplexT(RealT(0), RealT(-1));
        }
        return ComplexT(RealT(0), RealT(1));
    }

    int N1_;
    int N2_;
    int n_sites_;
    RealT beta_;
    RealT mu_;
    RealT hopping_t_;
    bool pbc_;
    ReplicaSubregionMask subregion_mask_type_;
    int origin_square_radius_;
    std::vector<int> subregion_mask_;
    Matrix<RealT> h_;
    RealVector<RealT> eigenvalues_;
    Matrix<RealT> eigenvectors_;
    std::vector<double> eigenvectors_col_major_;
    std::array<Matrix<ComplexT>, 2> sector_mixing_;
};

using ReplicaGreenCalculatorMP = ReplicaGreenCalculator<MP_Real>;
using ReplicaGreenCalculatorDD = ReplicaGreenCalculator<DoubleDouble>;

#endif
