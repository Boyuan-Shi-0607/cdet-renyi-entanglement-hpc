#ifndef CDET_EXACT_DIAGONALIZATION_BENCHMARK_H
#define CDET_EXACT_DIAGONALIZATION_BENCHMARK_H

#include <Eigen/Dense>

#include "cdet/parameters.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cdet_ed {

using Complex = std::complex<double>;

struct Parameters {
    int n1 = 2;
    int n2 = 2;
    double beta = 2.1;
    double hopping_t = 1.0;
    double interaction_u = 3.0;
    double mu = 1.0;
    double alpha_up = 0.7;
    double alpha_down = 0.7;
    bool pbc = true;
    std::string subregion_mask = "first_row";
    int subregion_square_radius = 0;
};

struct Options {
    std::string params_path = "../CDet_double_double_entanglement_entropy/params.txt";
    bool params_path_explicit = false;
    std::vector<double> steps{1e-2, 5e-3, 2e-3, 1e-3};
    std::vector<int> orders{1, 2, 3};
    int radius = 4;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    int dense_threshold = 8000;
    std::string symmetry = "auto";
    bool block_summary = false;
    bool progress = false;
};

struct State {
    std::uint32_t up = 0;
    std::uint32_t down = 0;
};

struct SplitState {
    int a_index = 0;
    std::uint64_t b_key = 0;
};

struct HoppingTerm {
    int to = 0;
    int from = 0;
    double amplitude = -1.0;
};

struct BlockResult {
    double z = 0.0;
    Eigen::MatrixXcd rho_a;
    int dimension = 0;
};

struct MomentumBasisVector {
    std::vector<std::pair<int, Complex>> coeffs;
};

struct MomentumBlockBasis {
    std::vector<MomentumBasisVector> vectors;
    std::vector<std::vector<std::pair<int, Complex>>> containing;
};

struct LogZResult {
    double z1 = 0.0;
    double tilde_z2 = 0.0;
    double ln_z1 = 0.0;
    double ln_tilde_z2 = 0.0;
    double ln_normalized_z2 = 0.0;
    int max_block_dimension = 0;
};

struct MomentumTask {
    int n_up = 0;
    int n_down = 0;
    int kx = 0;
    int ky = 0;
};

std::string trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

std::string lower_string(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool parse_bool(std::string value) {
    value = lower_string(trim(value));
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Cannot parse boolean value: " + value);
}

std::map<std::string, std::string> read_key_value_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open params file: " + path);
    }

    std::map<std::string, std::string> params;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            continue;
        }
        params[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
    }
    return params;
}

void load_parameters_from_file(const std::string& path, Parameters& params) {
    const auto values = read_key_value_file(path);
    auto get = [&](const std::string& key) -> const std::string* {
        const auto it = values.find(key);
        return it == values.end() ? nullptr : &it->second;
    };

    if (const auto* value = get("N1")) params.n1 = std::stoi(*value);
    if (const auto* value = get("N2")) params.n2 = std::stoi(*value);
    if (const auto* value = get("BETA")) params.beta = std::stod(*value);
    if (const auto* value = get("U")) params.interaction_u = std::stod(*value);
    if (const auto* value = get("MU")) params.mu = std::stod(*value);
    if (const auto* value = get("ALPHA_UP")) params.alpha_up = std::stod(*value);
    if (const auto* value = get("ALPHA_DOWN")) params.alpha_down = std::stod(*value);
    if (const auto* value = get("PBC")) params.pbc = parse_bool(*value);
    if (const auto* value = get("SUBREGION_MASK")) params.subregion_mask = *value;
    if (const auto* value = get("SUBREGION_SQUARE_RADIUS")) params.subregion_square_radius = std::stoi(*value);
}

std::vector<double> parse_double_list(const std::string& raw) {
    std::vector<double> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.push_back(std::stod(item));
        }
    }
    if (result.empty()) {
        throw std::runtime_error("Expected a non-empty double list");
    }
    return result;
}

std::vector<int> parse_int_list(const std::string& raw) {
    std::vector<int> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    if (result.empty()) {
        throw std::runtime_error("Expected a non-empty integer list");
    }
    return result;
}

void parse_command_line(int argc, char** argv, Options& options, Parameters& params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + name);
            }
            return argv[++i];
        };

        if (arg == "--params") {
            options.params_path = require_value(arg);
            options.params_path_explicit = true;
        } else if (arg == "--steps") {
            options.steps = parse_double_list(require_value(arg));
        } else if (arg == "--orders") {
            options.orders = parse_int_list(require_value(arg));
        } else if (arg == "--radius") {
            options.radius = std::stoi(require_value(arg));
        } else if (arg == "--threads") {
            options.threads = std::max(1, std::stoi(require_value(arg)));
        } else if (arg == "--dense-threshold") {
            options.dense_threshold = std::stoi(require_value(arg));
        } else if (arg == "--symmetry") {
            options.symmetry = lower_string(require_value(arg));
            if (options.symmetry != "auto" && options.symmetry != "number" && options.symmetry != "momentum") {
                throw std::runtime_error("--symmetry must be auto, number, or momentum");
            }
        } else if (arg == "--block-summary") {
            options.block_summary = true;
        } else if (arg == "--progress") {
            options.progress = true;
        } else if (arg == "--n1") {
            params.n1 = std::stoi(require_value(arg));
        } else if (arg == "--n2") {
            params.n2 = std::stoi(require_value(arg));
        } else if (arg == "--beta") {
            params.beta = std::stod(require_value(arg));
        } else if (arg == "--U") {
            params.interaction_u = std::stod(require_value(arg));
        } else if (arg == "--mu") {
            params.mu = std::stod(require_value(arg));
        } else if (arg == "--alpha-up") {
            params.alpha_up = std::stod(require_value(arg));
        } else if (arg == "--alpha-down") {
            params.alpha_down = std::stod(require_value(arg));
        } else if (arg == "--subregion") {
            params.subregion_mask = require_value(arg);
        } else if (arg == "--subregion-square-radius") {
            params.subregion_square_radius = std::stoi(require_value(arg));
        } else if (arg == "--pbc") {
            params.pbc = true;
        } else if (arg == "--open-boundary") {
            params.pbc = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: handwritten_normalized_ed [options]\n"
                << "  --params PATH\n"
                << "  --n1 N --n2 N --beta B --U U --mu MU\n"
                << "  --alpha-up A --alpha-down A --subregion first_row|first_column|checkerboard|origin_square\n"
                << "  --subregion-square-radius R\n"
                << "  --steps 1e-2,5e-3 --orders 1,2,3 --radius 4\n"
                << "  --threads N --dense-threshold N --symmetry auto|number|momentum\n"
                << "  --block-summary --progress\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown command-line option: " + arg);
        }
    }
}

bool file_exists(const std::string& path) {
    std::ifstream input(path);
    return static_cast<bool>(input);
}

void resolve_default_params_path(Options& options) {
    if (options.params_path_explicit && !file_exists(options.params_path)) {
        throw std::runtime_error("Could not open explicit --params path: " + options.params_path);
    }
    if (options.params_path_explicit || file_exists(options.params_path)) {
        return;
    }

    const std::vector<std::string> candidates{
        "../../CDet_double_double_entanglement_entropy/params.txt",
        "params.txt",
        "../params.txt",
    };
    for (const std::string& candidate : candidates) {
        if (file_exists(candidate)) {
            options.params_path = candidate;
            return;
        }
    }

    throw std::runtime_error(
        "Could not find params.txt. Pass --params /path/to/CDet_double_double_entanglement_entropy/params.txt");
}

int site_index(int x, int y, int n1, int n2) {
    if (x < 0 || x >= n1 || y < 0 || y >= n2) {
        throw std::runtime_error("Site coordinate out of range");
    }
    return y * n1 + x;
}

bool coordinate_in_origin_square_interval(int coordinate, int extent, int radius) {
    if (radius <= 0) {
        return false;
    }
    if (2 * radius >= extent) {
        return true;
    }
    return coordinate < radius || coordinate >= extent - radius;
}

bool site_in_origin_square_subregion(int x, int y, int n1, int n2, int radius) {
    return coordinate_in_origin_square_interval(x, n1, radius) &&
           coordinate_in_origin_square_interval(y, n2, radius);
}

std::vector<int> build_subregion_mask(int n1,
                                      int n2,
                                      const std::string& raw_name,
                                      int origin_square_radius = 0) {
    const std::string name = lower_string(trim(raw_name));
    std::vector<int> mask(static_cast<std::size_t>(n1 * n2), 0);
    if (name == "checkerboard") {
        for (int y = 0; y < n2; ++y) {
            for (int x = 0; x < n1; ++x) {
                mask[static_cast<std::size_t>(site_index(x, y, n1, n2))] = ((x + y) % 2 == 0) ? 1 : 0;
            }
        }
        return mask;
    }
    if (name == "first_row") {
        for (int x = 0; x < n1; ++x) {
            mask[static_cast<std::size_t>(site_index(x, 0, n1, n2))] = 1;
        }
        return mask;
    }
    if (name == "first_column") {
        for (int y = 0; y < n2; ++y) {
            mask[static_cast<std::size_t>(site_index(0, y, n1, n2))] = 1;
        }
        return mask;
    }
    if (name == "origin_square" || name == "center_square" || name == "centered_square") {
        if (origin_square_radius <= 0) {
            throw std::runtime_error("SUBREGION_SQUARE_RADIUS must be positive for origin_square");
        }
        if (2 * origin_square_radius > n1 || 2 * origin_square_radius > n2) {
            throw std::runtime_error(
                "SUBREGION_SQUARE_RADIUS is too large: require 2 * radius <= min(N1, N2)");
        }
        for (int y = 0; y < n2; ++y) {
            for (int x = 0; x < n1; ++x) {
                mask[static_cast<std::size_t>(site_index(x, y, n1, n2))] =
                    site_in_origin_square_subregion(x, y, n1, n2, origin_square_radius) ? 1 : 0;
            }
        }
        return mask;
    }
    throw std::runtime_error("Unknown SUBREGION_MASK: " + raw_name);
}

std::vector<HoppingTerm> build_hopping_terms(int n1, int n2, bool pbc, double hopping_t) {
    std::vector<HoppingTerm> terms;
    auto add_bond = [&](int i, int j) {
        if (i == j) {
            terms.push_back({i, i, -2.0 * hopping_t});
            return;
        }
        terms.push_back({j, i, -hopping_t});
        terms.push_back({i, j, -hopping_t});
    };

    for (int y = 0; y < n2; ++y) {
        for (int x = 0; x < n1; ++x) {
            const int i = site_index(x, y, n1, n2);
            if (x + 1 < n1) {
                add_bond(i, site_index(x + 1, y, n1, n2));
            } else if (pbc) {
                add_bond(i, site_index(0, y, n1, n2));
            }
            if (y + 1 < n2) {
                add_bond(i, site_index(x, y + 1, n1, n2));
            } else if (pbc) {
                add_bond(i, site_index(x, 0, n1, n2));
            }
        }
    }
    return terms;
}

std::vector<std::uint32_t> generate_fixed_popcount_masks(int n_sites, int n_particles) {
    if (n_particles < 0 || n_particles > n_sites || n_sites > 30) {
        throw std::runtime_error("Unsupported particle sector");
    }
    if (n_particles == 0) {
        return {0};
    }

    std::vector<std::uint32_t> masks;
    std::uint32_t value = (std::uint32_t(1) << n_particles) - 1u;
    const std::uint32_t limit = std::uint32_t(1) << n_sites;
    while (value < limit) {
        masks.push_back(value);
        const std::uint32_t c = value & (~value + 1u);
        const std::uint32_t r = value + c;
        if (r == 0) {
            break;
        }
        value = (((r ^ value) >> 2u) / c) | r;
    }
    return masks;
}

std::uint64_t state_key(std::uint32_t up, std::uint32_t down) {
    return (std::uint64_t(up) << 32u) | std::uint64_t(down);
}

int fermion_hop_sign(std::uint32_t bits, int from, int to) {
    if (((bits >> from) & 1u) == 0u || ((bits >> to) & 1u) != 0u) {
        return 0;
    }

    const std::uint32_t below_from = (from == 0) ? 0u : ((std::uint32_t(1) << from) - 1u);
    const int annihilate_parity = __builtin_popcount(bits & below_from) & 1;
    bits ^= (std::uint32_t(1) << from);

    const std::uint32_t below_to = (to == 0) ? 0u : ((std::uint32_t(1) << to) - 1u);
    const int create_parity = __builtin_popcount(bits & below_to) & 1;
    return ((annihilate_parity ^ create_parity) == 0) ? 1 : -1;
}

std::vector<State> build_sector_basis(int n_sites, int n_up, int n_down) {
    const auto up_masks = generate_fixed_popcount_masks(n_sites, n_up);
    const auto down_masks = generate_fixed_popcount_masks(n_sites, n_down);
    std::vector<State> basis;
    basis.reserve(up_masks.size() * down_masks.size());
    for (const auto up : up_masks) {
        for (const auto down : down_masks) {
            basis.push_back({up, down});
        }
    }
    return basis;
}

std::unordered_map<std::uint64_t, int> build_state_index(const std::vector<State>& basis) {
    std::unordered_map<std::uint64_t, int> index_of;
    index_of.reserve(basis.size() * 2u);
    for (int idx = 0; idx < static_cast<int>(basis.size()); ++idx) {
        index_of.emplace(state_key(basis[static_cast<std::size_t>(idx)].up,
                                   basis[static_cast<std::size_t>(idx)].down),
                         idx);
    }
    return index_of;
}

double diagonal_energy(const Parameters& params, double xi, const State& state) {
    const double up_potential = -(params.mu - params.alpha_up) - xi * params.alpha_up;
    const double down_potential = -(params.mu - params.alpha_down) - xi * params.alpha_down;
    const double interaction = xi * params.interaction_u;
    return up_potential * __builtin_popcount(state.up) +
           down_potential * __builtin_popcount(state.down) +
           interaction * __builtin_popcount(state.up & state.down);
}

template <typename Callback>
void for_each_hamiltonian_image(const Parameters& params,
                                double xi,
                                const std::vector<HoppingTerm>& hopping_terms,
                                const std::unordered_map<std::uint64_t, int>& index_of,
                                const State& state,
                                int source_index,
                                Callback&& callback) {
    callback(source_index, diagonal_energy(params, xi, state));

    for (const HoppingTerm& term : hopping_terms) {
        if (term.to == term.from) {
            double diagonal = 0.0;
            if (((state.up >> term.from) & 1u) != 0u) {
                diagonal += term.amplitude;
            }
            if (((state.down >> term.from) & 1u) != 0u) {
                diagonal += term.amplitude;
            }
            if (diagonal != 0.0) {
                callback(source_index, diagonal);
            }
            continue;
        }

        const int up_sign = fermion_hop_sign(state.up, term.from, term.to);
        if (up_sign != 0) {
            const std::uint32_t new_up =
                (state.up ^ (std::uint32_t(1) << term.from)) | (std::uint32_t(1) << term.to);
            const auto it = index_of.find(state_key(new_up, state.down));
            if (it == index_of.end()) {
                throw std::runtime_error("Internal error: hopped up state left the sector");
            }
            callback(it->second, term.amplitude * static_cast<double>(up_sign));
        }

        const int down_sign = fermion_hop_sign(state.down, term.from, term.to);
        if (down_sign != 0) {
            const std::uint32_t new_down =
                (state.down ^ (std::uint32_t(1) << term.from)) | (std::uint32_t(1) << term.to);
            const auto it = index_of.find(state_key(state.up, new_down));
            if (it == index_of.end()) {
                throw std::runtime_error("Internal error: hopped down state left the sector");
            }
            callback(it->second, term.amplitude * static_cast<double>(down_sign));
        }
    }
}

Eigen::MatrixXd build_hamiltonian_block(const Parameters& params,
                                        double xi,
                                        const std::vector<HoppingTerm>& hopping_terms,
                                        const std::vector<State>& basis) {
    const int dim = static_cast<int>(basis.size());
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dim, dim);
    const auto index_of = build_state_index(basis);

    for (int col = 0; col < dim; ++col) {
        const State state = basis[static_cast<std::size_t>(col)];
        for_each_hamiltonian_image(params, xi, hopping_terms, index_of, state, col,
                                   [&](int row, double amplitude) { h(row, col) += amplitude; });
    }

    return 0.5 * (h + h.transpose());
}

std::vector<SplitState> build_split_states(const Parameters& params,
                                           const std::vector<int>& subregion_mask,
                                           const std::vector<State>& basis) {
    const int n_sites = params.n1 * params.n2;
    std::vector<int> a_position(static_cast<std::size_t>(n_sites), -1);
    std::vector<int> b_position(static_cast<std::size_t>(n_sites), -1);
    int a_count = 0;
    int b_count = 0;
    for (int site = 0; site < n_sites; ++site) {
        if (subregion_mask[static_cast<std::size_t>(site)] != 0) {
            a_position[static_cast<std::size_t>(site)] = a_count++;
        } else {
            b_position[static_cast<std::size_t>(site)] = b_count++;
        }
    }

    std::vector<SplitState> split(basis.size());
    for (std::size_t idx = 0; idx < basis.size(); ++idx) {
        int a_index = 0;
        std::uint64_t b_key = 0;
        for (int site = 0; site < n_sites; ++site) {
            const int local_digit =
                int((basis[idx].up >> site) & 1u) + 2 * int((basis[idx].down >> site) & 1u);
            if (a_position[static_cast<std::size_t>(site)] >= 0) {
                a_index |= local_digit << (2 * a_position[static_cast<std::size_t>(site)]);
            } else {
                b_key |= std::uint64_t(local_digit) << (2 * b_position[static_cast<std::size_t>(site)]);
            }
        }
        split[idx] = {a_index, b_key};
    }
    return split;
}

std::vector<int> build_translation_site_map(const Parameters& params, int shift_x, int shift_y) {
    std::vector<int> site_map(static_cast<std::size_t>(params.n1 * params.n2));
    for (int y = 0; y < params.n2; ++y) {
        for (int x = 0; x < params.n1; ++x) {
            const int source = site_index(x, y, params.n1, params.n2);
            const int target_x = (x + shift_x) % params.n1;
            const int target_y = (y + shift_y) % params.n2;
            site_map[static_cast<std::size_t>(source)] = site_index(target_x, target_y, params.n1, params.n2);
        }
    }
    return site_map;
}

std::uint32_t translate_bits_with_sign(std::uint32_t bits, const std::vector<int>& site_map, int& sign) {
    std::vector<int> translated_sites;
    translated_sites.reserve(static_cast<std::size_t>(__builtin_popcount(bits)));
    std::uint32_t translated_bits = 0;
    for (int site = 0; site < static_cast<int>(site_map.size()); ++site) {
        if (((bits >> site) & 1u) != 0u) {
            const int target = site_map[static_cast<std::size_t>(site)];
            translated_sites.push_back(target);
            translated_bits |= (std::uint32_t(1) << target);
        }
    }

    int inversions = 0;
    for (std::size_t i = 0; i < translated_sites.size(); ++i) {
        for (std::size_t j = i + 1; j < translated_sites.size(); ++j) {
            if (translated_sites[i] > translated_sites[j]) {
                ++inversions;
            }
        }
    }
    sign = (inversions % 2 == 0) ? 1 : -1;
    return translated_bits;
}

State translate_state_with_sign(const State& state, const std::vector<int>& site_map, int& sign) {
    int up_sign = 1;
    int down_sign = 1;
    const std::uint32_t up = translate_bits_with_sign(state.up, site_map, up_sign);
    const std::uint32_t down = translate_bits_with_sign(state.down, site_map, down_sign);
    sign = up_sign * down_sign;
    return {up, down};
}

MomentumBlockBasis build_momentum_block_basis(const Parameters& params,
                                              const std::vector<State>& basis,
                                              const std::unordered_map<std::uint64_t, int>& index_of,
                                              int kx,
                                              int ky) {
    const int dim = static_cast<int>(basis.size());
    const int group_size = params.n1 * params.n2;
    const double pi = std::acos(-1.0);

    std::vector<std::vector<int>> translation_maps;
    translation_maps.reserve(static_cast<std::size_t>(group_size));
    std::vector<std::pair<int, int>> shifts;
    shifts.reserve(static_cast<std::size_t>(group_size));
    for (int dy = 0; dy < params.n2; ++dy) {
        for (int dx = 0; dx < params.n1; ++dx) {
            shifts.push_back({dx, dy});
            translation_maps.push_back(build_translation_site_map(params, dx, dy));
        }
    }

    MomentumBlockBasis momentum_basis;
    momentum_basis.containing.resize(static_cast<std::size_t>(dim));
    std::vector<unsigned char> visited(static_cast<std::size_t>(dim), 0);

    for (int rep_index = 0; rep_index < dim; ++rep_index) {
        if (visited[static_cast<std::size_t>(rep_index)] != 0) {
            continue;
        }

        std::vector<int> orbit_indices;
        orbit_indices.reserve(static_cast<std::size_t>(group_size));
        for (const auto& site_map : translation_maps) {
            int sign = 1;
            const State translated = translate_state_with_sign(basis[static_cast<std::size_t>(rep_index)],
                                                               site_map, sign);
            const auto it = index_of.find(state_key(translated.up, translated.down));
            if (it == index_of.end()) {
                throw std::runtime_error("Internal error: translated state left the sector");
            }
            if (std::find(orbit_indices.begin(), orbit_indices.end(), it->second) == orbit_indices.end()) {
                orbit_indices.push_back(it->second);
            }
        }
        for (const int orbit_index : orbit_indices) {
            visited[static_cast<std::size_t>(orbit_index)] = 1;
        }

        std::unordered_map<int, Complex> coeff_by_state;
        coeff_by_state.reserve(static_cast<std::size_t>(group_size));
        for (std::size_t g = 0; g < translation_maps.size(); ++g) {
            int sign = 1;
            const State translated = translate_state_with_sign(basis[static_cast<std::size_t>(rep_index)],
                                                               translation_maps[g], sign);
            const auto it = index_of.find(state_key(translated.up, translated.down));
            if (it == index_of.end()) {
                throw std::runtime_error("Internal error: translated state left the sector");
            }

            const auto [dx, dy] = shifts[g];
            const double angle =
                -2.0 * pi * (static_cast<double>(kx * dx) / static_cast<double>(params.n1) +
                             static_cast<double>(ky * dy) / static_cast<double>(params.n2));
            coeff_by_state[it->second] += static_cast<double>(sign) * Complex(std::cos(angle), std::sin(angle));
        }

        double norm2 = 0.0;
        for (const auto& [_, coeff] : coeff_by_state) {
            norm2 += std::norm(coeff);
        }
        if (norm2 < 1e-20) {
            continue;
        }

        MomentumBasisVector vector;
        vector.coeffs.reserve(coeff_by_state.size());
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (const auto& [state_index, coeff] : coeff_by_state) {
            const Complex normalized_coeff = coeff * inv_norm;
            vector.coeffs.push_back({state_index, normalized_coeff});
        }
        const int momentum_index = static_cast<int>(momentum_basis.vectors.size());
        for (const auto& [state_index, coeff] : vector.coeffs) {
            momentum_basis.containing[static_cast<std::size_t>(state_index)].push_back({momentum_index, coeff});
        }
        momentum_basis.vectors.push_back(std::move(vector));
    }

    return momentum_basis;
}

BlockResult solve_block(const Parameters& params,
                        double xi,
                        int n_up,
                        int n_down,
                        int d_a,
                        const std::vector<int>& subregion_mask,
                        const std::vector<HoppingTerm>& hopping_terms,
                        int dense_threshold) {
    const int n_sites = params.n1 * params.n2;
    const auto basis = build_sector_basis(n_sites, n_up, n_down);
    const int dim = static_cast<int>(basis.size());
    if (dim == 0) {
        return {0.0, Eigen::MatrixXcd::Zero(d_a, d_a), 0};
    }
    if (dim > dense_threshold) {
        std::ostringstream message;
        message << "Block (N_up=" << n_up << ", N_down=" << n_down << ") has dimension "
                << dim << ", above --dense-threshold=" << dense_threshold;
        throw std::runtime_error(message.str());
    }

    const Eigen::MatrixXd h = build_hamiltonian_block(params, xi, hopping_terms, basis);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen dense diagonalization failed");
    }

    const auto& energies = solver.eigenvalues();
    const auto& vectors = solver.eigenvectors();
    const auto split = build_split_states(params, subregion_mask, basis);

    std::vector<int> order(static_cast<std::size_t>(dim));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const auto lhs_key = split[static_cast<std::size_t>(lhs)].b_key;
        const auto rhs_key = split[static_cast<std::size_t>(rhs)].b_key;
        if (lhs_key != rhs_key) {
            return lhs_key < rhs_key;
        }
        return split[static_cast<std::size_t>(lhs)].a_index < split[static_cast<std::size_t>(rhs)].a_index;
    });

    std::vector<std::pair<int, int>> b_groups;
    for (int begin = 0; begin < dim;) {
        int end = begin + 1;
        const auto key = split[static_cast<std::size_t>(order[static_cast<std::size_t>(begin)])].b_key;
        while (end < dim &&
               split[static_cast<std::size_t>(order[static_cast<std::size_t>(end)])].b_key == key) {
            ++end;
        }
        b_groups.push_back({begin, end});
        begin = end;
    }

    double z_block = 0.0;
    Eigen::MatrixXcd rho_a = Eigen::MatrixXcd::Zero(d_a, d_a);
    for (int eig = 0; eig < dim; ++eig) {
        const double weight = std::exp(-params.beta * energies(eig));
        z_block += weight;
        for (const auto& group : b_groups) {
            for (int ii = group.first; ii < group.second; ++ii) {
                const int basis_i = order[static_cast<std::size_t>(ii)];
                const int a_i = split[static_cast<std::size_t>(basis_i)].a_index;
                const double coeff_i = vectors(basis_i, eig);
                if (coeff_i == 0.0) {
                    continue;
                }
                for (int jj = group.first; jj < group.second; ++jj) {
                    const int basis_j = order[static_cast<std::size_t>(jj)];
                    const int a_j = split[static_cast<std::size_t>(basis_j)].a_index;
                    rho_a(a_i, a_j) += weight * coeff_i * vectors(basis_j, eig);
                }
            }
        }
    }

    return {z_block, 0.5 * (rho_a + rho_a.adjoint()), dim};
}

BlockResult solve_momentum_block(const Parameters& params,
                                 double xi,
                                 int n_up,
                                 int n_down,
                                 int kx,
                                 int ky,
                                 int d_a,
                                 const std::vector<int>& subregion_mask,
                                 const std::vector<HoppingTerm>& hopping_terms,
                                 int dense_threshold) {
    const int n_sites = params.n1 * params.n2;
    const auto basis = build_sector_basis(n_sites, n_up, n_down);
    const int real_dim = static_cast<int>(basis.size());
    if (real_dim == 0) {
        return {0.0, Eigen::MatrixXcd::Zero(d_a, d_a), 0};
    }

    const auto index_of = build_state_index(basis);
    const MomentumBlockBasis momentum_basis = build_momentum_block_basis(params, basis, index_of, kx, ky);
    const int block_dim = static_cast<int>(momentum_basis.vectors.size());
    if (block_dim == 0) {
        return {0.0, Eigen::MatrixXcd::Zero(d_a, d_a), 0};
    }
    if (block_dim > dense_threshold) {
        std::ostringstream message;
        message << "Momentum block (N_up=" << n_up << ", N_down=" << n_down
                << ", kx=" << kx << ", ky=" << ky << ") has dimension "
                << block_dim << ", above --dense-threshold=" << dense_threshold;
        throw std::runtime_error(message.str());
    }

    Eigen::MatrixXcd h = Eigen::MatrixXcd::Zero(block_dim, block_dim);
    for (int col = 0; col < block_dim; ++col) {
        std::unordered_map<int, Complex> image;
        image.reserve(momentum_basis.vectors[static_cast<std::size_t>(col)].coeffs.size() *
                      static_cast<std::size_t>(2 * n_sites + 1));

        for (const auto& state_coeff : momentum_basis.vectors[static_cast<std::size_t>(col)].coeffs) {
            const int state_index = state_coeff.first;
            const Complex coeff = state_coeff.second;
            const State state = basis[static_cast<std::size_t>(state_index)];
            for_each_hamiltonian_image(params, xi, hopping_terms, index_of, state, state_index,
                                       [&](int target_index, double amplitude) {
                                           image[target_index] += coeff * amplitude;
                                       });
        }

        for (const auto& [state_index, value] : image) {
            for (const auto& [row, row_coeff] :
                 momentum_basis.containing[static_cast<std::size_t>(state_index)]) {
                h(row, col) += std::conj(row_coeff) * value;
            }
        }
    }
    h = 0.5 * (h + h.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(h);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen complex dense diagonalization failed");
    }

    const auto& energies = solver.eigenvalues();
    const auto& vectors = solver.eigenvectors();
    const auto split = build_split_states(params, subregion_mask, basis);

    std::vector<int> order(static_cast<std::size_t>(real_dim));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const auto lhs_key = split[static_cast<std::size_t>(lhs)].b_key;
        const auto rhs_key = split[static_cast<std::size_t>(rhs)].b_key;
        if (lhs_key != rhs_key) {
            return lhs_key < rhs_key;
        }
        return split[static_cast<std::size_t>(lhs)].a_index < split[static_cast<std::size_t>(rhs)].a_index;
    });

    std::vector<std::pair<int, int>> b_groups;
    for (int begin = 0; begin < real_dim;) {
        int end = begin + 1;
        const auto key = split[static_cast<std::size_t>(order[static_cast<std::size_t>(begin)])].b_key;
        while (end < real_dim &&
               split[static_cast<std::size_t>(order[static_cast<std::size_t>(end)])].b_key == key) {
            ++end;
        }
        b_groups.push_back({begin, end});
        begin = end;
    }

    Eigen::VectorXd weights(block_dim);
    double z_block = 0.0;
    for (int eig = 0; eig < block_dim; ++eig) {
        weights(eig) = std::exp(-params.beta * energies(eig));
        z_block += weights(eig);
    }

    const Eigen::MatrixXcd thermal_density =
        vectors * weights.asDiagonal() * vectors.adjoint();

    Eigen::MatrixXcd rho_a = Eigen::MatrixXcd::Zero(d_a, d_a);
    for (const auto& group : b_groups) {
        for (int ii = group.first; ii < group.second; ++ii) {
            const int basis_i = order[static_cast<std::size_t>(ii)];
            const int a_i = split[static_cast<std::size_t>(basis_i)].a_index;
            const auto& row_entries = momentum_basis.containing[static_cast<std::size_t>(basis_i)];
            if (row_entries.empty()) {
                continue;
            }
            for (int jj = group.first; jj < group.second; ++jj) {
                const int basis_j = order[static_cast<std::size_t>(jj)];
                const int a_j = split[static_cast<std::size_t>(basis_j)].a_index;
                const auto& col_entries = momentum_basis.containing[static_cast<std::size_t>(basis_j)];
                if (col_entries.empty()) {
                    continue;
                }

                Complex value = 0.0;
                for (const auto& [row, row_coeff] : row_entries) {
                    for (const auto& [col, col_coeff] : col_entries) {
                        value += row_coeff * thermal_density(row, col) * std::conj(col_coeff);
                    }
                }
                rho_a(a_i, a_j) += value;
            }
        }
    }

    return {z_block, 0.5 * (rho_a + rho_a.adjoint()), block_dim};
}

int subregion_site_count(const std::vector<int>& mask) {
    return std::accumulate(mask.begin(), mask.end(), 0);
}

long long binomial_int(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    k = std::min(k, n - k);
    long long value = 1;
    for (int i = 1; i <= k; ++i) {
        value = (value * static_cast<long long>(n - k + i)) / static_cast<long long>(i);
    }
    return value;
}

LogZResult normalized_log_z2_for_xi(const Parameters& params, double xi, const Options& options) {
    const int n_sites = params.n1 * params.n2;
    const auto mask = build_subregion_mask(
        params.n1, params.n2, params.subregion_mask, params.subregion_square_radius);
    const int a_sites = subregion_site_count(mask);
    const int d_a = 1 << (2 * a_sites);
    const auto hopping_terms = build_hopping_terms(params.n1, params.n2, params.pbc, params.hopping_t);

    std::vector<std::pair<int, int>> blocks;
    blocks.reserve(static_cast<std::size_t>((n_sites + 1) * (n_sites + 1)));
    int largest_requested_block = 0;
    std::pair<int, int> largest_block_sector{0, 0};
    for (int n_up = 0; n_up <= n_sites; ++n_up) {
        for (int n_down = 0; n_down <= n_sites; ++n_down) {
            const long long block_dim = binomial_int(n_sites, n_up) * binomial_int(n_sites, n_down);
            if (block_dim > largest_requested_block) {
                largest_requested_block = static_cast<int>(block_dim);
                largest_block_sector = {n_up, n_down};
            }
            blocks.push_back({n_up, n_down});
        }
    }
    if (options.symmetry == "momentum" && !params.pbc) {
        throw std::runtime_error("Momentum block diagonalization requires PBC=true");
    }
    const bool use_momentum =
        options.symmetry == "momentum" ||
        (options.symmetry == "auto" && params.pbc && largest_requested_block > options.dense_threshold);
    if (!use_momentum && largest_requested_block > options.dense_threshold) {
        std::ostringstream message;
        message << "Largest (N_up,N_down) block is ("
                << largest_block_sector.first << ", " << largest_block_sector.second
                << ") with dimension " << largest_requested_block
                << ", above --dense-threshold=" << options.dense_threshold
                << ". Use --symmetry momentum, a smaller lattice, or a larger threshold.";
        throw std::runtime_error(message.str());
    }

    std::vector<MomentumTask> momentum_blocks;
    if (use_momentum) {
        momentum_blocks.reserve(blocks.size() * static_cast<std::size_t>(params.n1 * params.n2));
        for (const auto& [n_up, n_down] : blocks) {
            for (int kx = 0; kx < params.n1; ++kx) {
                for (int ky = 0; ky < params.n2; ++ky) {
                    momentum_blocks.push_back({n_up, n_down, kx, ky});
                }
            }
        }
    }

    double z1 = 0.0;
    Eigen::MatrixXcd rho_a_total = Eigen::MatrixXcd::Zero(d_a, d_a);
    int max_block_dimension = 0;
    std::mutex accumulation_mutex;
    std::mutex error_mutex;
    std::mutex progress_mutex;
    std::exception_ptr first_error = nullptr;
    std::atomic<int> next_block{0};
    std::atomic<int> completed_blocks{0};

    const int task_count = use_momentum ? static_cast<int>(momentum_blocks.size()) : static_cast<int>(blocks.size());
    const int worker_count = std::max(1, std::min(options.threads, task_count));
    auto worker = [&]() {
        while (true) {
            const int block_index = next_block.fetch_add(1);
            if (block_index >= task_count) {
                return;
            }
            try {
                BlockResult result;
                if (use_momentum) {
                    const MomentumTask task = momentum_blocks[static_cast<std::size_t>(block_index)];
                    result = solve_momentum_block(params, xi, task.n_up, task.n_down, task.kx, task.ky,
                                                  d_a, mask, hopping_terms, options.dense_threshold);
                } else {
                    const auto [n_up, n_down] = blocks[static_cast<std::size_t>(block_index)];
                    result = solve_block(params, xi, n_up, n_down, d_a, mask, hopping_terms,
                                         options.dense_threshold);
                }
                std::lock_guard<std::mutex> lock(accumulation_mutex);
                z1 += result.z;
                rho_a_total += result.rho_a;
                max_block_dimension = std::max(max_block_dimension, result.dimension);
                const int done = completed_blocks.fetch_add(1) + 1;
                if (options.progress && (done == task_count || done % 25 == 0)) {
                    std::lock_guard<std::mutex> progress_lock(progress_mutex);
                    std::cerr << "xi=" << xi << " completed " << done << "/" << task_count
                              << " blocks, current max block dim=" << max_block_dimension << '\n';
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count));
    for (int i = 0; i < worker_count; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers) {
        thread.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }

    rho_a_total = 0.5 * (rho_a_total + rho_a_total.adjoint());
    const Complex tilde_z2_complex = (rho_a_total * rho_a_total).trace();
    const double tilde_z2 = std::real(tilde_z2_complex);
    if (!(z1 > 0.0) || !(tilde_z2 > 0.0)) {
        throw std::runtime_error("Encountered non-positive Z1 or tilde Z2");
    }
    const double ln_z1 = std::log(z1);
    const double ln_tilde_z2 = std::log(tilde_z2);
    return {z1, tilde_z2, ln_z1, ln_tilde_z2, ln_tilde_z2 - 2.0 * ln_z1, max_block_dimension};
}

double factorial(int n) {
    double value = 1.0;
    for (int i = 2; i <= n; ++i) {
        value *= static_cast<double>(i);
    }
    return value;
}

std::vector<double> finite_difference_weights(const std::vector<int>& offsets, int derivative_order) {
    const int n = static_cast<int>(offsets.size());
    if (derivative_order >= n) {
        throw std::runtime_error("Finite-difference stencil has too few points");
    }
    Eigen::MatrixXd vandermonde(n, n);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
    rhs(derivative_order) = factorial(derivative_order);
    for (int power = 0; power < n; ++power) {
        for (int col = 0; col < n; ++col) {
            vandermonde(power, col) = std::pow(static_cast<double>(offsets[static_cast<std::size_t>(col)]), power);
        }
    }
    const Eigen::VectorXd weights = vandermonde.partialPivLu().solve(rhs);
    return std::vector<double>(weights.data(), weights.data() + weights.size());
}

std::vector<int> build_offsets(int radius) {
    if (radius < 1) {
        throw std::runtime_error("--radius must be positive");
    }
    std::vector<int> offsets;
    offsets.reserve(static_cast<std::size_t>(2 * radius + 1));
    for (int offset = -radius; offset <= radius; ++offset) {
        offsets.push_back(offset);
    }
    return offsets;
}

void print_parameters(const Parameters& params, const Options& options) {
    const int n_sites = params.n1 * params.n2;
    const auto mask = build_subregion_mask(
        params.n1, params.n2, params.subregion_mask, params.subregion_square_radius);
    const int a_sites = subregion_site_count(mask);
    std::cout << "=== Handwritten normalized ED benchmark ===\n";
    std::cout << "params = " << options.params_path << '\n';
    std::cout << "N1 x N2 = " << params.n1 << " x " << params.n2 << ", N = " << n_sites << '\n';
    std::cout << "beta = " << params.beta << ", U = " << params.interaction_u << ", mu = " << params.mu << '\n';
    std::cout << "alpha_up = " << params.alpha_up << ", alpha_down = " << params.alpha_down << '\n';
    std::cout << "PBC = " << (params.pbc ? "true" : "false") << '\n';
    std::cout << "SUBREGION_MASK = " << params.subregion_mask;
    if (lower_string(trim(params.subregion_mask)) == "origin_square") {
        std::cout << ", SUBREGION_SQUARE_RADIUS = " << params.subregion_square_radius;
    }
    std::cout << ", |A| = " << a_sites
              << ", dim(rho_A) = " << (1 << (2 * a_sites)) << '\n';
    std::cout << "symmetry mode = " << options.symmetry
              << " [number=(N_up,N_down), momentum=(N_up,N_down,kx,ky)]\n";
    std::cout << "threads = " << options.threads << ", dense threshold = " << options.dense_threshold << '\n';
    std::cout << "object = ln tilde Z2[A] - 2 ln Z1\n";
    std::cout << "coefficient density = derivative / (k! * N)\n";
}

void print_block_summary(const Parameters& params, const Options& options) {
    (void)options;
    const int n_sites = params.n1 * params.n2;
    int max_number_dim = 0;
    std::pair<int, int> max_number_sector{0, 0};
    for (int n_up = 0; n_up <= n_sites; ++n_up) {
        for (int n_down = 0; n_down <= n_sites; ++n_down) {
            const int dim = static_cast<int>(binomial_int(n_sites, n_up) * binomial_int(n_sites, n_down));
            if (dim > max_number_dim) {
                max_number_dim = dim;
                max_number_sector = {n_up, n_down};
            }
        }
    }

    std::cout << "\n=== Block Summary ===\n";
    std::cout << "largest number block: (N_up,N_down)=("
              << max_number_sector.first << ", " << max_number_sector.second
              << "), dim=" << max_number_dim << '\n';

    if (!params.pbc) {
        std::cout << "momentum summary skipped because PBC=false\n";
        return;
    }

    int max_momentum_dim = 0;
    int nonempty_momentum_blocks = 0;
    int total_momentum_dim = 0;
    MomentumTask max_task;
    for (int n_up = 0; n_up <= n_sites; ++n_up) {
        for (int n_down = 0; n_down <= n_sites; ++n_down) {
            const auto basis = build_sector_basis(n_sites, n_up, n_down);
            const auto index_of = build_state_index(basis);
            for (int kx = 0; kx < params.n1; ++kx) {
                for (int ky = 0; ky < params.n2; ++ky) {
                    const MomentumBlockBasis momentum_basis =
                        build_momentum_block_basis(params, basis, index_of, kx, ky);
                    const int dim = static_cast<int>(momentum_basis.vectors.size());
                    if (dim > 0) {
                        ++nonempty_momentum_blocks;
                        total_momentum_dim += dim;
                    }
                    if (dim > max_momentum_dim) {
                        max_momentum_dim = dim;
                        max_task = {n_up, n_down, kx, ky};
                    }
                }
            }
        }
    }

    std::cout << "largest momentum block: (N_up,N_down,kx,ky)=("
              << max_task.n_up << ", " << max_task.n_down << ", "
              << max_task.kx << ", " << max_task.ky << "), dim="
              << max_momentum_dim << '\n';
    std::cout << "nonempty momentum blocks = " << nonempty_momentum_blocks
              << ", summed momentum dimensions = " << total_momentum_dim << '\n';
}

inline void run_exact_diagonalization_benchmark(const Parameters& params, const Options& options) {
    const int n_sites = params.n1 * params.n2;
    const auto offsets = build_offsets(options.radius);
    for (const int order : options.orders) {
        if (order < 0 || order >= static_cast<int>(offsets.size())) {
            throw std::runtime_error("ED derivative order requires a larger ED_STENCIL_RADIUS");
        }
    }

    std::map<int, std::vector<double>> weights_by_order;
    for (const int order : options.orders) {
        weights_by_order.emplace(order, finite_difference_weights(offsets, order));
    }

    print_parameters(params, options);
    std::cout << "orders = ";
    for (std::size_t i = 0; i < options.orders.size(); ++i) {
        std::cout << (i == 0 ? "[" : ", ") << options.orders[i];
    }
    std::cout << "]\n";
    std::cout << "radius = " << options.radius << ", offsets = ";
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        std::cout << (i == 0 ? "[" : ", ") << offsets[i];
    }
    std::cout << "]\n";
    std::cout << "steps = ";
    for (std::size_t i = 0; i < options.steps.size(); ++i) {
        std::cout << (i == 0 ? "[" : ", ") << options.steps[i];
    }
    std::cout << "]\n";

    if (options.block_summary) {
        print_block_summary(params, options);
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    std::cout << std::setprecision(15);

    const LogZResult xi_one = normalized_log_z2_for_xi(params, 1.0, options);
    std::cout << "\nED xi=1 ln Z2 norm / N = "
              << xi_one.ln_normalized_z2 / static_cast<double>(n_sites) << '\n';

    for (const double step : options.steps) {
        std::map<int, double> values_by_offset;
        std::map<int, LogZResult> details_by_offset;
        for (const int offset : offsets) {
            const double xi = static_cast<double>(offset) * step;
            LogZResult result = normalized_log_z2_for_xi(params, xi, options);
            values_by_offset[offset] = result.ln_normalized_z2;
            details_by_offset[offset] = result;
        }

        std::cout << "\n=== ED XI_STEP = " << std::scientific << step << std::defaultfloat << " ===\n";
        const auto& zero = details_by_offset.at(0);
        std::cout << "xi=0 ln tilde Z2 = " << std::scientific << zero.ln_tilde_z2 << '\n';
        std::cout << "xi=0 ln Z1       = " << zero.ln_z1 << '\n';
        std::cout << "xi=0 ln Z2 norm  = " << zero.ln_normalized_z2 << '\n';
        std::cout << "max block dimension = " << zero.max_block_dimension << '\n';

        for (const int order : options.orders) {
            const auto& weights = weights_by_order.at(order);
            double derivative = 0.0;
            for (std::size_t i = 0; i < offsets.size(); ++i) {
                derivative += weights[i] * values_by_offset.at(offsets[i]);
            }
            derivative /= std::pow(step, order);
            const double coefficient_density = derivative / (factorial(order) * static_cast<double>(n_sites));
            std::cout << "ED order " << order << ": derivative = " << derivative
                      << ", coefficient density = " << coefficient_density << '\n';
        }
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "\nED elapsed time = " << std::fixed << std::setprecision(6) << elapsed << " s\n";
}

inline Parameters make_parameters_from_cdet_globals() {
    Parameters params;
    params.n1 = N1;
    params.n2 = N2;
    params.beta = beta.convert_to<double>();
    params.hopping_t = 1.0;
    params.interaction_u = U.convert_to<double>();
    params.mu = mu.convert_to<double>();
    params.alpha_up = alpha_shift_up.convert_to<double>();
    params.alpha_down = alpha_shift_down.convert_to<double>();
    params.pbc = pbc;
    params.subregion_mask = subregion_mask_name;
    params.subregion_square_radius = subregion_square_radius;
    return params;
}

inline Options make_options_from_cdet_globals() {
    Options options;
    options.params_path = "CDet params.txt";
    options.params_path_explicit = true;
    options.steps = ed_xi_steps;
    options.orders.clear();
    for (int order = 1; order <= ed_order_max; ++order) {
        options.orders.push_back(order);
    }
    options.radius = ed_stencil_radius;
    options.threads = ed_threads;
    options.dense_threshold = ed_dense_threshold;
    options.symmetry = ed_symmetry;
    options.block_summary = ed_block_summary;
    options.progress = ed_progress;
    return options;
}

inline void run_exact_diagonalization_benchmark_from_cdet_globals() {
    run_exact_diagonalization_benchmark(make_parameters_from_cdet_globals(),
                                        make_options_from_cdet_globals());
}

}  // namespace cdet_ed

#endif
