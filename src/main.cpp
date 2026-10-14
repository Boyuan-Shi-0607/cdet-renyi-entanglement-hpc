#include "cdet/diagonal_green_observable.h"
#include "cdet/exact_diagonalization_benchmark.h"
#include "cdet/free_fermion_renyi.h"
#include "cdet/load_data.h"
#include "cdet/mcmcmc_sampler.h"
#include "cdet/runtime_counters.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#ifndef CDET_ENABLE_PLAIN_MC_BENCHMARK
#define CDET_ENABLE_PLAIN_MC_BENCHMARK 0
#endif

namespace {

using ParamMap = std::unordered_map<std::string, std::vector<std::string>>;

struct DeterministicOrderBenchmarks {
    DoubleDouble order1_value;
    bool has_order1_value = false;
};

std::array<unsigned char, 256> createMSBTable() {
    std::array<unsigned char, 256> table{};
    for (int i = 0; i < 256; ++i) {
        unsigned char msb = 0;
        if (i != 0) {
            unsigned char value = static_cast<unsigned char>(i);
            msb = 128;
            while ((value & msb) == 0) {
                msb >>= 1;
            }
        }
        table[i] = msb;
    }
    return table;
}

std::string resident_green_table_real_type_name(const ResidentGreenFunctionTable& table) {
    return std::visit(
        [](const auto& typed_table) -> std::string {
            using TableT = std::decay_t<decltype(typed_table)>;
            return real_type_name<typename TableT::value_type>();
        },
        table);
}

void trim(std::string& s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
            s.end());
}

bool string_to_bool(std::string value) {
    trim(value);
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (value == "true" || value == "1" || value == "yes") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no") {
        return false;
    }
    throw std::runtime_error("Invalid boolean value: " + value);
}

MP_Real string_to_mp_real(const std::string& value) {
    return MP_Real(value);
}

const std::vector<std::string>& require_param(const ParamMap& params, const std::string& key) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        throw std::runtime_error("Missing parameter: " + key);
    }
    return it->second;
}

std::string require_single_param(const ParamMap& params, const std::string& key) {
    const auto& values = require_param(params, key);
    if (values.size() != 1) {
        throw std::runtime_error("Parameter " + key + " must contain exactly one value");
    }
    return values.front();
}

std::string optional_single_param(const ParamMap& params,
                                  const std::string& key,
                                  const std::string& default_value) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return default_value;
    }
    if (it->second.size() != 1) {
        throw std::runtime_error("Parameter " + key + " must contain exactly one value");
    }
    return it->second.front();
}

const std::vector<std::string>* find_optional_param_values(const ParamMap& params,
                                                           const std::string& key) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<double> optional_double_list_param(const ParamMap& params,
                                               const std::vector<std::string>& keys,
                                               std::size_t expected_size,
                                               double default_value) {
    const std::vector<std::string>* values = nullptr;
    std::string chosen_key;
    for (const auto& key : keys) {
        values = find_optional_param_values(params, key);
        if (values != nullptr) {
            chosen_key = key;
            break;
        }
    }

    if (values == nullptr) {
        return std::vector<double>(expected_size, default_value);
    }
    if (values->size() != expected_size) {
        std::ostringstream msg;
        msg << "Parameter " << chosen_key << " must contain exactly " << expected_size << " values";
        throw std::runtime_error(msg.str());
    }

    std::vector<double> result;
    result.reserve(values->size());
    for (const auto& value : *values) {
        const double parsed = std::stod(value);
        if (!std::isfinite(parsed) || parsed <= 0.0) {
            throw std::runtime_error(chosen_key + " entries must be positive finite numbers");
        }
        result.push_back(parsed);
    }
    return result;
}

std::vector<double> optional_positive_double_list_param(const ParamMap& params,
                                                        const std::vector<std::string>& keys,
                                                        std::vector<double> default_values) {
    const std::vector<std::string>* values = nullptr;
    std::string chosen_key;
    for (const auto& key : keys) {
        values = find_optional_param_values(params, key);
        if (values != nullptr) {
            chosen_key = key;
            break;
        }
    }

    if (values == nullptr) {
        return default_values;
    }

    std::vector<double> result;
    result.reserve(values->size());
    for (const auto& value : *values) {
        const double parsed = std::stod(value);
        if (!std::isfinite(parsed) || parsed <= 0.0) {
            throw std::runtime_error(chosen_key + " entries must be positive finite numbers");
        }
        result.push_back(parsed);
    }
    if (result.empty()) {
        throw std::runtime_error(chosen_key + " must contain at least one value");
    }
    return result;
}

int parse_positive_int_or_auto(const std::string& raw_value, const std::string& name) {
    if (normalize_real_type_name(raw_value) == "auto") {
        return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    const int value = std::stoi(raw_value);
    if (value <= 0) {
        throw std::runtime_error(name + " must be positive or auto");
    }
    return value;
}

std::size_t read_env_size_t(const char* name, std::size_t default_value) {
    if (const char* raw_value = std::getenv(name); raw_value != nullptr && raw_value[0] != '\0') {
        try {
            return static_cast<std::size_t>(std::stoull(raw_value));
        } catch (...) {
        }
    }
    return default_value;
}

ParamMap read_params_file(std::string& params_path) {
    constexpr std::array<const char*, 2> candidate_paths = {"../params.txt", "params.txt"};

    for (const char* candidate_path : candidate_paths) {
        std::ifstream infile(candidate_path);
        if (!infile.is_open()) {
            continue;
        }

        params_path = candidate_path;
        ParamMap params;
        std::string line;
        while (std::getline(infile, line)) {
            trim(line);
            if (line.empty() || line.front() == '#') {
                continue;
            }

            const std::size_t pos = line.find('=');
            if (pos == std::string::npos) {
                continue;
            }

            std::string key = line.substr(0, pos);
            std::string value_str = line.substr(pos + 1);
            trim(key);
            trim(value_str);

            std::vector<std::string> tokens;
            std::stringstream ss(value_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                trim(token);
                if (!token.empty()) {
                    tokens.push_back(token);
                }
            }
            params[key] = tokens;
        }
        return params;
    }

    throw std::runtime_error("Could not open params.txt from ../params.txt or params.txt");
}

void load_scalar_parameters(const ParamMap& params) {
    N1 = std::stoi(require_single_param(params, "N1"));
    N2 = std::stoi(require_single_param(params, "N2"));
    beta = string_to_mp_real(require_single_param(params, "BETA"));

    cheb_nu = std::stoi(require_single_param(params, "CHEB_NU"));
    cheb_nv = std::stoi(require_single_param(params, "CHEB_NV"));
    tol_fpm = string_to_mp_real(require_single_param(params, "TOL_FPM"));
    const std::string fpm_shift_raw = optional_single_param(params, "FPM_GLOBAL_SHIFT", "auto");
    if (normalize_real_type_name(fpm_shift_raw) == "auto") {
        fpm_global_shift = MP_Real(-1);
    } else {
        fpm_global_shift = string_to_mp_real(fpm_shift_raw);
        if (fpm_global_shift < MP_Real(0)) {
            throw std::runtime_error("FPM_GLOBAL_SHIFT must be nonnegative or auto");
        }
    }
    branch_epsilon = string_to_mp_real(require_single_param(params, "BRANCH_EPSILON"));
    alpha_shift_up = string_to_mp_real(require_single_param(params, "ALPHA_UP"));
    alpha_shift_down = string_to_mp_real(require_single_param(params, "ALPHA_DOWN"));
    mu = string_to_mp_real(require_single_param(params, "MU"));
    U = string_to_mp_real(require_single_param(params, "U"));
    pbc = string_to_bool(require_single_param(params, "PBC"));
    internal_integrand_threads =
        std::max(1, std::stoi(optional_single_param(params, "INTERNAL_INTEGRAND_THREADS", "1")));
    table_build_threads =
        std::max(1, std::stoi(optional_single_param(params, "TABLE_BUILD_THREADS", "1")));
    green_table_drop_bits =
        std::stoi(optional_single_param(params, "GREEN_TABLE_DROP_BITS", "0"));
    if (green_table_drop_bits < 0) {
        throw std::runtime_error("GREEN_TABLE_DROP_BITS must be nonnegative");
    }
    const ReplicaSubregionMask parsed_subregion_mask =
        parse_subregion_mask(require_single_param(params, "SUBREGION_MASK"));
    subregion_mask_name = subregion_mask_to_string(parsed_subregion_mask);
    subregion_square_radius =
        std::stoi(optional_single_param(params, "SUBREGION_SQUARE_RADIUS", "0"));
    if (subregion_square_radius < 0) {
        throw std::runtime_error("SUBREGION_SQUARE_RADIUS must be nonnegative");
    }
    if (parsed_subregion_mask == ReplicaSubregionMask::OriginSquare) {
        if (subregion_square_radius <= 0) {
            throw std::runtime_error("SUBREGION_SQUARE_RADIUS must be positive when SUBREGION_MASK=origin_square");
        }
        if (2 * subregion_square_radius > N1 || 2 * subregion_square_radius > N2) {
            throw std::runtime_error(
                "SUBREGION_SQUARE_RADIUS is too large: require 2 * radius <= min(N1, N2)");
        }
    }

    const LoadDataRealType load_data_real_type =
        parse_load_data_real_type(optional_single_param(params, "LOAD_DATA_REAL", "MP"));
    load_data_real_name = load_data_real_type_to_string(load_data_real_type);
    const IntegrandRealType integrand_real_type =
        parse_integrand_real_type(optional_single_param(params, "INTEGRAND_REAL", "double_double"));
    integrand_real_name = integrand_real_type_to_string(integrand_real_type);
    const IntegrandRealType vector_real_type =
        parse_vector_real_type(optional_single_param(params, "VECTOR_REAL", integrand_real_name));
    vector_real_name = vector_real_type_to_string(vector_real_type);
    load_data_workflow =
        normalize_real_type_name(optional_single_param(params, "LOAD_DATA_WORKFLOW", "resident"));
    if (const char* workflow_env = std::getenv("CDET_LOAD_DATA_WORKFLOW");
        workflow_env != nullptr && workflow_env[0] != '\0') {
        load_data_workflow = normalize_real_type_name(workflow_env);
    }
    if (load_data_workflow == "build_resident") {
        load_data_workflow = "resident";
    }
    if (load_data_workflow == "load_prebuilt" || load_data_workflow == "load_prebuilt_coeffs") {
        load_data_workflow = "load_chunks";
    }
    if (load_data_workflow == "sample") {
        load_data_workflow = "sample_nodes";
    }
    if (load_data_workflow == "precompute" || load_data_workflow == "calculator_cache") {
        load_data_workflow = "precompute_calculators";
    }
    if (load_data_workflow == "transform") {
        load_data_workflow = "transform_sitepairs";
    }
    if (load_data_workflow == "diagonal_observable" || load_data_workflow == "diagonal_lines") {
        load_data_workflow = "diagonal_reduce";
    }
    load_data_only =
        string_to_bool(optional_single_param(params, "LOAD_DATA_ONLY", load_data_only ? "true" : "false"));
    if (const char* load_data_only_env = std::getenv("CDET_LOAD_DATA_ONLY");
        load_data_only_env != nullptr && load_data_only_env[0] != '\0') {
        load_data_only = string_to_bool(load_data_only_env);
    }
    load_data_chunk_dir =
        optional_single_param(params, "LOAD_DATA_CHUNK_DIR", load_data_chunk_dir);
    if (const char* chunk_dir_env = std::getenv("CDET_LOAD_DATA_CHUNK_DIR");
        chunk_dir_env != nullptr && chunk_dir_env[0] != '\0') {
        load_data_chunk_dir = chunk_dir_env;
    }
    load_data_node_chunk_size =
        std::stoi(optional_single_param(params, "LOAD_DATA_NODE_CHUNK_SIZE", "0"));
    if (const char* node_chunk_env = std::getenv("CDET_LOAD_DATA_NODE_CHUNK_SIZE");
        node_chunk_env != nullptr && node_chunk_env[0] != '\0') {
        load_data_node_chunk_size = std::stoi(node_chunk_env);
    }
    load_data_sitepair_chunk_size =
        std::stoi(optional_single_param(params, "LOAD_DATA_SITEPAIR_CHUNK_SIZE", "0"));
    if (const char* sitepair_chunk_env = std::getenv("CDET_LOAD_DATA_SITEPAIR_CHUNK_SIZE");
        sitepair_chunk_env != nullptr && sitepair_chunk_env[0] != '\0') {
        load_data_sitepair_chunk_size = std::stoi(sitepair_chunk_env);
    }
    if (load_data_node_chunk_size < 0 || load_data_sitepair_chunk_size < 0) {
        throw std::runtime_error("LOAD_DATA_*_CHUNK_SIZE must be nonnegative");
    }
    load_data_sample_target_file_gb =
        std::stod(optional_single_param(params, "LOAD_DATA_SAMPLE_TARGET_FILE_GB", "4.0"));
    if (const char* sample_target_env = std::getenv("CDET_LOAD_DATA_SAMPLE_TARGET_FILE_GB");
        sample_target_env != nullptr && sample_target_env[0] != '\0') {
        load_data_sample_target_file_gb = std::stod(sample_target_env);
    }
    load_data_transform_target_mem_gb =
        std::stod(optional_single_param(params, "LOAD_DATA_TRANSFORM_TARGET_MEM_GB", "4.0"));
    if (const char* transform_target_env = std::getenv("CDET_LOAD_DATA_TRANSFORM_TARGET_MEM_GB");
        transform_target_env != nullptr && transform_target_env[0] != '\0') {
        load_data_transform_target_mem_gb = std::stod(transform_target_env);
    }
    if (load_data_sample_target_file_gb <= 0.0 || load_data_transform_target_mem_gb <= 0.0) {
        throw std::runtime_error("LOAD_DATA sample/transform target GB values must be positive");
    }
    load_data_calculator_cache =
        normalize_real_type_name(optional_single_param(params, "LOAD_DATA_CALCULATOR_CACHE", load_data_calculator_cache));
    if (const char* cache_env = std::getenv("CDET_LOAD_DATA_CALCULATOR_CACHE");
        cache_env != nullptr && cache_env[0] != '\0') {
        load_data_calculator_cache = normalize_real_type_name(cache_env);
    }
    if (load_data_calculator_cache != "auto" && load_data_calculator_cache != "off" &&
        load_data_calculator_cache != "require") {
        throw std::runtime_error("LOAD_DATA_CALCULATOR_CACHE must be auto, off, or require");
    }
    load_data_condition_check_mode = parse_linear_solve_condition_check_mode(
        optional_single_param(params, "LOAD_DATA_CONDITION_CHECK",
                              linear_solve_condition_check_mode_to_string(load_data_condition_check_mode)));
    if (const char* condition_env = std::getenv("CDET_LOAD_DATA_CONDITION_CHECK");
        condition_env != nullptr && condition_env[0] != '\0') {
        load_data_condition_check_mode =
            parse_linear_solve_condition_check_mode(condition_env);
    }
    load_data_condition_rcond_warn =
        std::stod(optional_single_param(params, "LOAD_DATA_CONDITION_RCOND_WARN",
                                        "1e-12"));
    if (const char* rcond_env = std::getenv("CDET_LOAD_DATA_CONDITION_RCOND_WARN");
        rcond_env != nullptr && rcond_env[0] != '\0') {
        load_data_condition_rcond_warn = std::stod(rcond_env);
    }
    if (!(load_data_condition_rcond_warn > 0.0) ||
        load_data_condition_rcond_warn > 1.0) {
        throw std::runtime_error("LOAD_DATA_CONDITION_RCOND_WARN must be in (0, 1]");
    }
    load_data_condition_max_warnings =
        std::stoi(optional_single_param(params, "LOAD_DATA_CONDITION_MAX_WARNINGS",
                                        std::to_string(load_data_condition_max_warnings)));
    if (const char* max_warning_env = std::getenv("CDET_LOAD_DATA_CONDITION_MAX_WARNINGS");
        max_warning_env != nullptr && max_warning_env[0] != '\0') {
        load_data_condition_max_warnings = std::stoi(max_warning_env);
    }
    if (load_data_condition_max_warnings < 0) {
        throw std::runtime_error("LOAD_DATA_CONDITION_MAX_WARNINGS must be nonnegative");
    }
    load_data_condition_warning_count.store(0, std::memory_order_relaxed);
    mcmcmc_table_backend =
        normalize_real_type_name(optional_single_param(params, "MCMCMC_TABLE_BACKEND", "resident"));
    if (const char* backend_env = std::getenv("CDET_MCMCMC_TABLE_BACKEND");
        backend_env != nullptr && backend_env[0] != '\0') {
        mcmcmc_table_backend = normalize_real_type_name(backend_env);
    }
    if (mcmcmc_table_backend == "lru" || mcmcmc_table_backend == "on_demand" ||
        mcmcmc_table_backend == "disk" || mcmcmc_table_backend == "disk_cache") {
        mcmcmc_table_backend = "disk_lru";
    }
    if (mcmcmc_table_backend != "resident" && mcmcmc_table_backend != "disk_lru") {
        throw std::runtime_error("MCMCMC_TABLE_BACKEND must be resident or disk_lru");
    }
    mcmcmc_lru_cache_gb =
        std::stod(optional_single_param(params, "MCMCMC_LRU_CACHE_GB", "64.0"));
    if (const char* cache_env = std::getenv("CDET_MCMCMC_LRU_CACHE_GB");
        cache_env != nullptr && cache_env[0] != '\0') {
        mcmcmc_lru_cache_gb = std::stod(cache_env);
    }
    if (!(mcmcmc_lru_cache_gb > 0.0)) {
        throw std::runtime_error("MCMCMC_LRU_CACHE_GB must be positive");
    }

    const std::string epsilon0_raw = optional_single_param(params, "ZTRANSFORM_EPSILON0", "auto");
    auto& roundoff_config = ztransform_roundoff_config();
    if (normalize_real_type_name(epsilon0_raw) == "auto") {
        roundoff_config.set_epsilon0_auto();
    } else {
        roundoff_config.set_epsilon0_literal(epsilon0_raw);
    }
    roundoff_config.set_safety_factor_literal(
        optional_single_param(params, "ZTRANSFORM_SAFETY_FACTOR", "1e4"));

    mcmcmc_order_min =
        std::stoi(optional_single_param(params, "MCMCMC_ORDER_MIN", "2"));
    mcmcmc_order_max =
        std::stoi(optional_single_param(params, "MCMCMC_ORDER_MAX", "4"));
    mcmcmc_envelope_order =
        std::stoi(optional_single_param(params,
                                        "MCMCMC_ENVELOPE_ORDER",
                                        std::to_string(mcmcmc_order_max + 1)));
    mcmcmc_normalization_order =
        std::stoi(optional_single_param(params,
                                        "MCMCMC_NORMALIZATION_ORDER",
                                        std::to_string(mcmcmc_order_min)));
    const int measured_order_count = mcmcmc_order_max - mcmcmc_order_min + 1;
    if (measured_order_count <= 0) {
        throw std::runtime_error("MCMCMC_ORDER_MAX must be >= MCMCMC_ORDER_MIN");
    }
    mcmcmc_order_lambdas = optional_double_list_param(
        params,
        {"MCMCMC_LAMBDAS"},
        static_cast<std::size_t>(measured_order_count),
        1.0);
    mcmcmc_chain_count =
        std::stoi(optional_single_param(params, "MCMCMC_CHAIN_COUNT", "8"));
    mcmcmc_base_seed = static_cast<std::uint64_t>(
        std::stoull(optional_single_param(params, "MCMCMC_BASE_SEED", "998875432210")));
    mcmcmc_lambda_warmup_steps =
        std::stoi(optional_single_param(params, "MCMCMC_LAMBDA_WARMUP_STEPS", "10000"));
    mcmcmc_lambda_adapt_interval =
        std::stoi(optional_single_param(params, "MCMCMC_LAMBDA_ADAPT_INTERVAL", "1000"));
    mcmcmc_lambda_adapt_rate =
        std::stod(optional_single_param(params, "MCMCMC_LAMBDA_ADAPT_RATE", "0.25"));
    mcmcmc_production_steps =
        std::stoi(optional_single_param(params, "MCMCMC_PRODUCTION_STEPS", "100000"));
    mcmcmc_seed_mode =
        normalize_real_type_name(optional_single_param(params, "MCMCMC_SEED_MODE", "uniform"));
    mcmcmc_seed_histogram_learning_steps =
        std::stoi(optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_LEARNING_STEPS", "0"));
    mcmcmc_seed_histogram_tau_bins =
        std::stoi(optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_TAU_BINS", "16"));
    mcmcmc_seed_histogram_uniform_mix =
        std::stod(optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_UNIFORM_MIX", "0.05"));
    mcmcmc_seed_histogram_pseudo_count =
        std::stod(optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_PSEUDO_COUNT", "0"));
    mcmcmc_seed_histogram_learning_only =
        string_to_bool(optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_LEARNING_ONLY", "false"));
    mcmcmc_seed_histogram_output_prefix =
        optional_single_param(params, "MCMCMC_SEED_HISTOGRAM_OUTPUT_PREFIX", "");
    mcmcmc_compare_uniform_seed =
        string_to_bool(optional_single_param(params, "MCMCMC_COMPARE_UNIFORM_SEED", "false"));
    mcmcmc_checkpoint_enabled =
        string_to_bool(optional_single_param(params, "MCMCMC_CHECKPOINT_ENABLED", "false"));
    if (const char* checkpoint_env = std::getenv("CDET_MCMCMC_CHECKPOINT_ENABLED");
        checkpoint_env != nullptr && checkpoint_env[0] != '\0') {
        mcmcmc_checkpoint_enabled = string_to_bool(checkpoint_env);
    }
    mcmcmc_checkpoint_dir =
        optional_single_param(params, "MCMCMC_CHECKPOINT_DIR", "mcmcmc_checkpoints");
    if (const char* checkpoint_dir_env = std::getenv("CDET_MCMCMC_CHECKPOINT_DIR");
        checkpoint_dir_env != nullptr && checkpoint_dir_env[0] != '\0') {
        mcmcmc_checkpoint_dir = checkpoint_dir_env;
    }
    mcmcmc_checkpoint_interval_steps = static_cast<std::uint64_t>(
        std::stoull(optional_single_param(params, "MCMCMC_CHECKPOINT_INTERVAL_STEPS", "10000000")));
    if (const char* checkpoint_steps_env = std::getenv("CDET_MCMCMC_CHECKPOINT_INTERVAL_STEPS");
        checkpoint_steps_env != nullptr && checkpoint_steps_env[0] != '\0') {
        mcmcmc_checkpoint_interval_steps = static_cast<std::uint64_t>(std::stoull(checkpoint_steps_env));
    }
    mcmcmc_resume_mode =
        normalize_real_type_name(optional_single_param(params, "MCMCMC_RESUME", "off"));
    if (const char* resume_env = std::getenv("CDET_MCMCMC_RESUME");
        resume_env != nullptr && resume_env[0] != '\0') {
        mcmcmc_resume_mode = normalize_real_type_name(resume_env);
    }
    if (mcmcmc_resume_mode == "required") {
        mcmcmc_resume_mode = "require";
    }
#if CDET_ENABLE_PLAIN_MC_BENCHMARK
    plain_mc_benchmark_samples =
        std::stoi(optional_single_param(params, "PLAIN_MC_BENCHMARK_SAMPLES", "100000"));
#else
    plain_mc_benchmark_samples = 0;
#endif

    ed_benchmark_enabled =
        string_to_bool(optional_single_param(params, "ED_BENCHMARK_ENABLED", "false"));
    ed_benchmark_only =
        string_to_bool(optional_single_param(params, "ED_BENCHMARK_ONLY", "false"));
    ed_threads = parse_positive_int_or_auto(optional_single_param(params, "ED_THREADS", "auto"),
                                            "ED_THREADS");
    ed_xi_steps = optional_positive_double_list_param(params,
                                                      {"ED_XI_STEPS", "ED_STEP_SIZE"},
                                                      {1.0e-2});
    ed_stencil_radius =
        std::stoi(optional_single_param(params, "ED_STENCIL_RADIUS", "4"));
    if (ed_stencil_radius <= 0) {
        throw std::runtime_error("ED_STENCIL_RADIUS must be positive");
    }
    ed_order_max =
        std::stoi(optional_single_param(params, "ED_ORDER_MAX", "3"));
    if (ed_order_max <= 0) {
        throw std::runtime_error("ED_ORDER_MAX must be positive");
    }
    ed_dense_threshold =
        std::stoi(optional_single_param(params, "ED_DENSE_THRESHOLD", "8000"));
    if (ed_dense_threshold <= 0) {
        throw std::runtime_error("ED_DENSE_THRESHOLD must be positive");
    }
    ed_symmetry = normalize_real_type_name(optional_single_param(params, "ED_SYMMETRY", "auto"));
    if (ed_symmetry != "auto" && ed_symmetry != "number" && ed_symmetry != "momentum") {
        throw std::runtime_error("ED_SYMMETRY must be auto, number, or momentum");
    }
    ed_progress =
        string_to_bool(optional_single_param(params, "ED_PROGRESS", "false"));
    ed_block_summary =
        string_to_bool(optional_single_param(params, "ED_BLOCK_SUMMARY", "false"));
}

void print_loaded_parameters(const std::string& params_path) {
    std::cout << "\n--- Loaded Parameters ---\n";
    std::cout << "params file: " << params_path << '\n';
    std::cout << "N1 = " << N1 << ", N2 = " << N2 << ", beta = " << beta << '\n';
    std::cout << "CHEB_NU = " << cheb_nu << ", CHEB_NV = " << cheb_nv << '\n';
    std::cout << "TOL_FPM = " << tol_fpm
              << ", FPM_GLOBAL_SHIFT = "
              << (fpm_global_shift < MP_Real(0) ? std::string("auto") : fpm_global_shift.str())
              << '\n';
    std::cout << "MU = " << mu << ", U = " << U << '\n';
    std::cout << "ALPHA_UP = " << alpha_shift_up
              << ", ALPHA_DOWN = " << alpha_shift_down << '\n';
    std::cout << "PBC = " << (pbc ? "true" : "false")
              << ", SUBREGION_MASK = " << subregion_mask_name << '\n';
    if (parse_subregion_mask(subregion_mask_name) == ReplicaSubregionMask::OriginSquare) {
        std::cout << "SUBREGION_SQUARE_RADIUS = " << subregion_square_radius
                  << ", square sites = " << (2 * subregion_square_radius) << " x "
                  << (2 * subregion_square_radius) << '\n';
    }
    std::cout << "INTERNAL_INTEGRAND_THREADS = " << internal_integrand_threads << '\n';
    std::cout << "TABLE_BUILD_THREADS = " << table_build_threads << '\n';
    std::cout << "GREEN_TABLE_DROP_BITS = " << green_table_drop_bits << '\n';
    std::cout << "LOAD_DATA_REAL = " << load_data_real_name << '\n';
    std::cout << "VECTOR_REAL = " << vector_real_name << '\n';
    std::cout << "INTEGRAND_REAL = " << integrand_real_name << '\n';
    std::cout << "LOAD_DATA_WORKFLOW = " << load_data_workflow << '\n';
    std::cout << "LOAD_DATA_ONLY = " << (load_data_only ? "true" : "false") << '\n';
    if (load_data_workflow != "resident") {
        std::cout << "LOAD_DATA_CHUNK_DIR = " << load_data_chunk_dir << '\n';
        std::cout << "LOAD_DATA_NODE_CHUNK_SIZE = "
                  << (load_data_node_chunk_size > 0 ? std::to_string(load_data_node_chunk_size) : std::string("auto"))
                  << ", LOAD_DATA_SITEPAIR_CHUNK_SIZE = "
                  << (load_data_sitepair_chunk_size > 0 ? std::to_string(load_data_sitepair_chunk_size) : std::string("auto"))
                  << '\n';
        std::cout << "LOAD_DATA_SAMPLE_TARGET_FILE_GB = " << load_data_sample_target_file_gb
                  << ", LOAD_DATA_TRANSFORM_TARGET_MEM_GB = "
                  << load_data_transform_target_mem_gb << '\n';
        std::cout << "LOAD_DATA_CALCULATOR_CACHE = " << load_data_calculator_cache << '\n';
    }
    std::cout << "LOAD_DATA_CONDITION_CHECK = "
              << linear_solve_condition_check_mode_to_string(load_data_condition_check_mode)
              << ", LOAD_DATA_CONDITION_RCOND_WARN = "
              << load_data_condition_rcond_warn
              << ", LOAD_DATA_CONDITION_MAX_WARNINGS = "
              << load_data_condition_max_warnings << '\n';
    const IntegrandRealType integrand_real_type = parse_integrand_real_type(integrand_real_name);
    std::cout << "ZTRANSFORM_EPSILON0 = " << ztransform_epsilon0_display_string(integrand_real_type)
              << ", ZTRANSFORM_SAFETY_FACTOR = "
              << ztransform_safety_factor_display_string(integrand_real_type)
              << ", filter_threshold = " << ztransform_threshold_display_string(integrand_real_type)
              << '\n';
    std::cout << "MCMCMC_ORDER_MIN = " << mcmcmc_order_min
              << ", MCMCMC_ORDER_MAX = " << mcmcmc_order_max
              << ", MCMCMC_ENVELOPE_ORDER = " << mcmcmc_envelope_order << '\n';
    std::cout << "MCMCMC_NORMALIZATION_ORDER = " << mcmcmc_normalization_order << '\n';
    std::cout << "MCMCMC lambdas = {";
    for (std::size_t i = 0; i < mcmcmc_order_lambdas.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << "order " << (mcmcmc_order_min + static_cast<int>(i))
                  << ": " << mcmcmc_order_lambdas[i];
    }
    std::cout << "}\n";
    std::cout << "MCMCMC_CHAIN_COUNT = " << mcmcmc_chain_count
              << ", MCMCMC_BASE_SEED = " << mcmcmc_base_seed << '\n';
    std::cout << "MCMCMC_TABLE_BACKEND = " << mcmcmc_table_backend;
    if (mcmcmc_table_backend == "disk_lru") {
        std::cout << ", MCMCMC_LRU_CACHE_GB = " << mcmcmc_lru_cache_gb;
    }
    std::cout << '\n';
    std::cout << "MCMCMC lambda warmup / production = "
              << mcmcmc_lambda_warmup_steps << " / " << mcmcmc_production_steps << '\n';
    std::cout << "MCMCMC_LAMBDA_ADAPT_INTERVAL = " << mcmcmc_lambda_adapt_interval
              << ", MCMCMC_LAMBDA_ADAPT_RATE = " << mcmcmc_lambda_adapt_rate << '\n';
    std::cout << "MCMCMC_SEED_MODE = " << mcmcmc_seed_mode
              << ", MCMCMC_COMPARE_UNIFORM_SEED = "
              << (mcmcmc_compare_uniform_seed ? "true" : "false") << '\n';
    std::cout << "MCMCMC_CHECKPOINT_ENABLED = "
              << (mcmcmc_checkpoint_enabled ? "true" : "false")
              << ", MCMCMC_RESUME = " << mcmcmc_resume_mode
              << ", CHECKPOINT_DIR = " << mcmcmc_checkpoint_dir << '\n';
    if (mcmcmc_checkpoint_enabled) {
        std::cout << "MCMCMC_CHECKPOINT_INTERVAL_STEPS = "
                  << mcmcmc_checkpoint_interval_steps << '\n';
    }
    if (mcmcmc_seed_mode == "histogram") {
        std::cout << "MCMCMC_SEED_HISTOGRAM_LEARNING_STEPS = "
                  << mcmcmc_seed_histogram_learning_steps
                  << ", TAU_BINS = " << mcmcmc_seed_histogram_tau_bins
                  << ", UNIFORM_MIX = " << mcmcmc_seed_histogram_uniform_mix
                  << ", PSEUDO_COUNT = " << mcmcmc_seed_histogram_pseudo_count << '\n';
        std::cout << "MCMCMC_SEED_HISTOGRAM_LEARNING_ONLY = "
                  << (mcmcmc_seed_histogram_learning_only ? "true" : "false")
                  << ", OUTPUT_PREFIX = "
                  << (mcmcmc_seed_histogram_output_prefix.empty()
                          ? "(none)"
                          : mcmcmc_seed_histogram_output_prefix)
                  << '\n';
    }
    std::cout << "ED_BENCHMARK_ENABLED = " << (ed_benchmark_enabled ? "true" : "false")
              << ", ED_BENCHMARK_ONLY = " << (ed_benchmark_only ? "true" : "false") << '\n';
    if (ed_benchmark_enabled || ed_benchmark_only || ed_block_summary) {
        std::cout << "ED_THREADS = " << ed_threads
                  << ", ED_ORDER_MAX = " << ed_order_max
                  << ", ED_STENCIL_RADIUS = " << ed_stencil_radius
                  << ", ED_DENSE_THRESHOLD = " << ed_dense_threshold
                  << ", ED_SYMMETRY = " << ed_symmetry << '\n';
        std::cout << "ED_XI_STEPS = {";
        for (std::size_t i = 0; i < ed_xi_steps.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << ed_xi_steps[i];
        }
        std::cout << "}, ED_PROGRESS = " << (ed_progress ? "true" : "false")
                  << ", ED_BLOCK_SUMMARY = " << (ed_block_summary ? "true" : "false") << '\n';
    }
#if CDET_ENABLE_PLAIN_MC_BENCHMARK
    std::cout << "PLAIN_MC_BENCHMARK_SAMPLES = " << plain_mc_benchmark_samples << '\n';
#else
    std::cout << "Plain MC benchmark disabled at compile time "
              << "(CDET_ENABLE_PLAIN_MC_BENCHMARK=0)\n";
#endif
}

void write_matrix_csv(const std::filesystem::path& path,
                      const std::vector<double>& values,
                      int rows,
                      int cols) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open histogram output file: " + path.string());
    }
    out << std::setprecision(17);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col > 0) {
                out << ',';
            }
            out << values[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                          static_cast<std::size_t>(col)];
        }
        out << '\n';
    }
}

void export_factorized_histogram_seed(const FactorizedHistogramSeed& histogram_seed,
                                      const std::string& output_prefix) {
    if (output_prefix.empty()) {
        return;
    }

    const std::filesystem::path prefix(output_prefix);
    const std::filesystem::path parent = prefix.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    const auto& settings = histogram_seed.settings();
    write_matrix_csv(prefix.string() + "_tau.csv",
                     histogram_seed.tau_probabilities(),
                     settings.tau_bins,
                     settings.tau_bins);
    write_matrix_csv(prefix.string() + "_space.csv",
                     histogram_seed.space_probabilities(),
                     histogram_seed.site_count(),
                     histogram_seed.site_count());
    write_matrix_csv(prefix.string() + "_replica.csv",
                     histogram_seed.replica_probabilities(),
                     2,
                     2);

    std::ofstream meta(prefix.string() + "_metadata.txt");
    if (!meta.is_open()) {
        throw std::runtime_error("Could not open histogram metadata file");
    }
    meta << "N1=" << settings.N1 << '\n'
         << "N2=" << settings.N2 << '\n'
         << "BETA=" << beta << '\n'
         << "TAU_BINS=" << settings.tau_bins << '\n'
         << "UNIFORM_MIX=" << settings.uniform_mixing << '\n'
         << "PSEUDO_COUNT=" << settings.pseudo_count << '\n';

    std::cout << "Saved histogram seed matrices to:\n"
              << "  " << prefix.string() << "_tau.csv\n"
              << "  " << prefix.string() << "_space.csv\n"
              << "  " << prefix.string() << "_replica.csv\n"
              << "  " << prefix.string() << "_metadata.txt\n";
}

std::string dd_display_string(const DoubleDouble& value) {
    return format_mp_real_compact(value.to_mp_real());
}

void print_mcmcmc_summary(const MCMCMCSummary& summary,
                          const std::string& seed_label,
                          const DoubleDouble* deterministic_order1 = nullptr) {
    const auto& config = summary.config;
    auto safe_ratio = [](std::uint64_t numerator, std::uint64_t denominator) {
        if (denominator == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return static_cast<double>(numerator) / static_cast<double>(denominator);
    };

    std::cout << "\nMCMCMC chain summaries (" << seed_label << " seed)\n";
    if (!summary.completed) {
        std::cout << "  status: checkpointed/incomplete run; summaries below use completed steps only\n";
    }
    for (const auto& chain : summary.chains) {
        std::cout << "  chain " << chain.chain_index << " (seed " << chain.seed << "): lambdas = {";
        bool first = true;
        for (int order = config.order_min; order <= config.order_max; ++order) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << order << ": " << chain.final_lambdas[static_cast<std::size_t>(order)];
        }
        std::cout << "}, chosen production counts = {";
        first = true;
        for (int order = config.order_min; order <= config.order_max; ++order) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << order << ": " << chain.production_chosen_counts[static_cast<std::size_t>(order)];
        }
        std::cout << "}, warmup acceptance = "
                  << safe_ratio(chain.warmup_accepted_steps, chain.warmup_proposed_steps)
                  << ", production acceptance = "
                  << safe_ratio(chain.production_accepted_steps, chain.production_proposed_steps)
                  << ", mean retained fraction = "
                  << static_cast<double>(chain.production_retained_fraction_stats.mean)
                  << ", warmup steps = " << chain.warmup_proposed_steps
                  << ", production steps = " << chain.production_proposed_steps
                  << "\n";
    }

    std::cout << "\nMCMCMC ratios relative to order " << config.normalization_order
              << " (" << seed_label << " seed)\n";
    for (int order = config.order_min; order <= config.order_max; ++order) {
        const auto& stats = summary.cross_chain_order_signal_stats[static_cast<std::size_t>(order)];
        std::cout << "  order " << order
                  << ": de-lambda mean = " << static_cast<double>(stats.mean)
                  << " +/- " << static_cast<double>(stats.stderr())
                  << ", ratio = " << static_cast<double>(summary.ratios_to_normalization.at(order))
                  << '\n';
    }

    if (deterministic_order1 != nullptr && config.order_min <= 1 && config.order_max >= 1) {
        const auto& ref_stats = summary.cross_chain_order_signal_stats[1];
        const long double ref_mean = ref_stats.mean;
        const long double ref_stderr = ref_stats.stderr();
        const long double known_order1 = deterministic_order1->to_mp_real().convert_to<long double>();

        std::cout << "\nMCMCMC absolute estimates normalized by deterministic order 1"
                  << " (" << seed_label << " seed)\n";
        if (ref_mean == 0.0L || !std::isfinite(static_cast<double>(ref_mean))) {
            std::cout << "  skipped: MCMCMC order-1 de-lambda mean is zero or non-finite\n";
            return;
        }

        const long double normalization_constant = known_order1 / ref_mean;
        const long double normalization_stderr =
            std::abs(known_order1 * ref_stderr / (ref_mean * ref_mean));
        std::cout << "  normalization constant = "
                  << static_cast<double>(normalization_constant)
                  << " +/- " << static_cast<double>(normalization_stderr)
                  << " from deterministic order 1 = "
                  << dd_display_string(*deterministic_order1) << '\n';

        for (int order = config.order_min; order <= config.order_max; ++order) {
            const auto& stats = summary.cross_chain_order_signal_stats[static_cast<std::size_t>(order)];
            if (order == 1) {
                std::cout << "  order 1: integral = "
                          << dd_display_string(*deterministic_order1)
                          << " (deterministic), MCMCMC de-lambda mean = "
                          << static_cast<double>(stats.mean)
                          << " +/- " << static_cast<double>(stats.stderr()) << '\n';
                continue;
            }

            const long double mean = stats.mean;
            const long double stderr = stats.stderr();
            const long double estimate = normalization_constant * mean;
            const long double variance =
                normalization_constant * normalization_constant * stderr * stderr +
                (mean * normalization_stderr) * (mean * normalization_stderr);
            const long double propagated_stderr =
                variance >= 0.0L ? std::sqrt(variance) : std::numeric_limits<long double>::quiet_NaN();

            std::cout << "  order " << order
                      << ": integral = " << static_cast<double>(estimate)
                      << " +/- " << static_cast<double>(propagated_stderr)
                      << '\n';
        }
    }
}

#if CDET_ENABLE_PLAIN_MC_BENCHMARK
void print_plain_mc_summary(const PlainMCBenchmarkSummary& summary) {
    if (summary.samples_per_order <= 0) {
        std::cout << "\nPlain MC benchmark skipped.\n";
        return;
    }
    std::cout << "\nPlain uniform MC benchmark ratios relative to order "
              << summary.normalization_order << '\n';
    for (int order = summary.order_min; order <= summary.order_max; ++order) {
        const auto& stats = summary.order_integral_stats[static_cast<std::size_t>(order)];
        std::cout << "  order " << order
                  << ": mean = " << static_cast<double>(stats.mean)
                  << " +/- " << static_cast<double>(stats.stderr())
                  << ", ratio = " << static_cast<double>(summary.ratios_to_normalization.at(order))
                  << '\n';
    }
}
#endif

DeterministicOrderBenchmarks run_deterministic_order_benchmarks() {
    DeterministicOrderBenchmarks result;
    std::cout << "\nComputing deterministic order-1 normalized diagonal observable...\n";
    if (mcmcmc_table_backend == "disk_lru") {
        if (g_disk_diagonal_order1_value_available) {
            std::cout << "Diagonal observable uses precomputed chunked lower-branch line reduction.\n";
        } else {
            std::cout << "Diagonal observable uses disk-backed LRU Green tables"
                      << " with exact lower-branch Chebyshev line integration.\n";
        }
    } else {
        std::cout << "Diagonal observable uses resident Green-table precision = "
                  << resident_green_table_real_type_name(g_table_up)
                  << " with exact lower-branch Chebyshev line integration.\n";
    }
    const auto order1_start = std::chrono::steady_clock::now();
    const DoubleDouble order1_value = diagonal_observable_density_dd();
    result.order1_value = order1_value;
    result.has_order1_value = true;
    const std::chrono::duration<double> order1_elapsed =
        std::chrono::steady_clock::now() - order1_start;
    std::cout << "order 1 normalized diagonal = " << dd_display_string(order1_value)
              << ", elapsed time = " << order1_elapsed.count() << " s\n";

    return result;
}

void print_runtime_counters() {
    const CDetRuntimeCountersSnapshot counters = snapshot_cdet_runtime_counters();
    std::cout << "\nRuntime integrand/FPM counters\n";
    std::cout << "  integrand calls = " << counters.integrand_calls << '\n';
    std::cout << "  principal-minor fallbacks = " << counters.fpm_fallbacks << '\n';
    std::cout << "  retained Green-pair cache hits = "
              << counters.green_pair_cache_hits
              << ", misses = " << counters.green_pair_cache_misses << '\n';
    const std::uint64_t green_pair_cache_total =
        counters.green_pair_cache_hits + counters.green_pair_cache_misses;
    if (green_pair_cache_total > 0) {
        const long double hit_fraction =
            static_cast<long double>(counters.green_pair_cache_hits) /
            static_cast<long double>(green_pair_cache_total);
        std::cout << "  retained Green-pair cache hit fraction = "
                  << static_cast<double>(hit_fraction) << '\n';
    }
    if (counters.integrand_calls > 0) {
        const long double fallback_per_call =
            static_cast<long double>(counters.fpm_fallbacks) /
            static_cast<long double>(counters.integrand_calls);
        std::cout << "  fallbacks per integrand call = "
                  << static_cast<double>(fallback_per_call) << '\n';
    }
}

void print_disk_lru_cache_stats() {
    if (mcmcmc_table_backend != "disk_lru" || !g_disk_tables_initialized) {
        return;
    }
    const DiskBackedGreenCacheStats stats = std::visit(
        [](const auto& table) -> DiskBackedGreenCacheStats {
            if (!table.coeff_cache) {
                return {};
            }
            return table.coeff_cache->stats();
        },
        g_disk_table_up);
    const auto gib = [](std::uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    };
    std::cout << "\nDisk-backed Green-table LRU cache stats\n";
    std::cout << "  hits = " << stats.hits
              << ", misses = " << stats.misses
              << ", evictions = " << stats.evictions << '\n';
    std::cout << "  bytes read = " << gib(stats.bytes_read) << " GiB"
              << ", cached = " << gib(stats.cached_bytes) << " / "
              << gib(stats.max_bytes) << " GiB"
              << ", entries = " << stats.entry_count << '\n';
}

}  // namespace

int main() {
    try {
        std::cout << std::setprecision(15);

        std::string params_path;
        const ParamMap params = read_params_file(params_path);
        load_scalar_parameters(params);

        internal_integrand_threads = static_cast<int>(
            read_env_size_t("CDET_INTERNAL_THREADS", static_cast<std::size_t>(internal_integrand_threads)));
        internal_integrand_threads = std::max(1, internal_integrand_threads);
        table_build_threads =
            static_cast<int>(read_env_size_t("CDET_TABLE_THREADS", static_cast<std::size_t>(table_build_threads)));
        table_build_threads = std::max(1, table_build_threads);
        set_integrand_internal_thread_count(internal_integrand_threads);

        if (mcmcmc_chain_count > 1 && internal_integrand_threads > 1) {
            std::cout << "Warning: running multiple outer MCMCMC chains with internal integrand threads > 1"
                      << " can oversubscribe the machine.\n";
        }

        const MCMCMCConfig mcmcmc_config = make_mcmcmc_config_from_globals();
        validate_mcmcmc_config(mcmcmc_config);
        MSB_table = createMSBTable();
        print_loaded_parameters(params_path);

        if (load_data_workflow == "plan") {
            std::cout << "\nPlanning chunked load-data workflow...\n";
            run_chunked_load_data_planner();
            return 0;
        }
        if (load_data_workflow == "sample_nodes") {
            std::cout << "\nRunning one chunked load-data node-sample job...\n";
            run_chunked_load_data_sample_node_job();
            return 0;
        }
        if (load_data_workflow == "precompute_calculators") {
            std::cout << "\nPrecomputing chunked load-data calculator cache...\n";
            run_chunked_load_data_precompute_calculators();
            return 0;
        }
        if (load_data_workflow == "corners") {
            std::cout << "\nRunning one chunked load-data corner job...\n";
            run_chunked_load_data_corner_job();
            return 0;
        }
        if (load_data_workflow == "transform_sitepairs") {
            std::cout << "\nRunning one chunked load-data sitepair-transform job...\n";
            run_chunked_load_data_transform_sitepair_job();
            return 0;
        }
        if (load_data_workflow == "diagonal_reduce" || load_data_workflow == "diagonal") {
            std::cout << "\nReducing chunked diagonal observable...\n";
            run_chunked_load_data_diagonal_reduce();
            return 0;
        }
        if (load_data_workflow == "validate") {
            std::cout << "\nValidating chunked load-data outputs...\n";
            validate_chunked_load_data_outputs();
            return 0;
        }
        if (load_data_workflow != "resident" && load_data_workflow != "load_chunks") {
            throw std::runtime_error(
                "Unknown LOAD_DATA_WORKFLOW: " + load_data_workflow +
                ". Supported values are resident, plan, precompute_calculators, sample_nodes, corners, "
                "transform_sitepairs, diagonal_reduce, validate, load_chunks.");
        }

        if (ed_benchmark_only) {
            std::cout << "\nRunning exact-diagonalization benchmark only "
                      << "(ED_BENCHMARK_ONLY=true)...\n";
            cdet_ed::run_exact_diagonalization_benchmark_from_cdet_globals();
            return 0;
        }

        std::cout << "\nLoading replica Green-function tables...\n";
        load_data();
        if (load_data_only) {
            std::cout << "\nStopping after load_data because LOAD_DATA_ONLY=true.\n";
            print_disk_lru_cache_stats();
            print_runtime_counters();
            return 0;
        }
        cdet_free::print_free_spinful_renyi_entropy_from_cdet_globals();

        const DeterministicOrderBenchmarks deterministic_benchmarks =
            run_deterministic_order_benchmarks();
        const DoubleDouble* deterministic_order1 =
            deterministic_benchmarks.has_order1_value
                ? &deterministic_benchmarks.order1_value
                : nullptr;

        std::shared_ptr<const MCMCMCSeedKernel> seed_kernel = make_uniform_mcmcmc_seed();
        if (mcmcmc_config.seed_mode == "histogram") {
            std::cout << "\nLearning factorized histogram MCMCMC seed from uniform envelopes...\n";
            long double total_pair_weight = 0.0L;
            const auto histogram_learning_start = std::chrono::steady_clock::now();
            seed_kernel = learn_factorized_histogram_seed(mcmcmc_config, &total_pair_weight);
            const std::chrono::duration<double> histogram_learning_elapsed =
                std::chrono::steady_clock::now() - histogram_learning_start;
            std::cout << "histogram seed learning elapsed time = "
                      << histogram_learning_elapsed.count() << " s\n";
            std::cout << "Histogram seed learned with total directed-pair weight = "
                      << static_cast<double>(total_pair_weight) << '\n';
            std::cout << "Seed density convention: q_tau is stored in tau_unit = tau / beta, "
                      << "so q_unit = beta * q_physical_tau for the time factor.\n";
            const auto* histogram_seed = dynamic_cast<const FactorizedHistogramSeed*>(seed_kernel.get());
            if (histogram_seed == nullptr) {
                throw std::runtime_error("Internal error: learned histogram seed has unexpected type");
            }
            export_factorized_histogram_seed(*histogram_seed,
                                             mcmcmc_seed_histogram_output_prefix);
            if (mcmcmc_config.seed_mode == "histogram" && mcmcmc_seed_histogram_learning_only) {
                std::cout << "\nStopping after histogram learning/export because "
                          << "MCMCMC_SEED_HISTOGRAM_LEARNING_ONLY=true.\n";
                print_disk_lru_cache_stats();
                print_runtime_counters();
                return 0;
            }
        }

        std::cout << "\nRunning set heat-bath MCMCMC with "
                  << seed_kernel->name() << " seed...\n";
        const auto mcmcmc_start = std::chrono::steady_clock::now();
        const MCMCMCSummary mcmcmc_summary = run_mcmcmc_chains_parallel(mcmcmc_config, seed_kernel);
        const std::chrono::duration<double> mcmcmc_elapsed =
            std::chrono::steady_clock::now() - mcmcmc_start;
        std::cout << "set heat-bath MCMCMC elapsed time = "
                  << mcmcmc_elapsed.count() << " s\n";
        print_mcmcmc_summary(mcmcmc_summary, seed_kernel->name(), deterministic_order1);

        if (mcmcmc_config.compare_uniform_seed && seed_kernel->name() != "uniform") {
            std::cout << "\nRunning uniform-seed comparison with the same MCMCMC settings...\n";
            const std::shared_ptr<const MCMCMCSeedKernel> uniform_seed = make_uniform_mcmcmc_seed();
            const auto uniform_mcmcmc_start = std::chrono::steady_clock::now();
            const MCMCMCSummary uniform_summary =
                run_mcmcmc_chains_parallel(mcmcmc_config, uniform_seed);
            const std::chrono::duration<double> uniform_mcmcmc_elapsed =
                std::chrono::steady_clock::now() - uniform_mcmcmc_start;
            std::cout << "uniform-seed comparison MCMCMC elapsed time = "
                      << uniform_mcmcmc_elapsed.count() << " s\n";
            print_mcmcmc_summary(uniform_summary, uniform_seed->name(), deterministic_order1);
        }

#if CDET_ENABLE_PLAIN_MC_BENCHMARK
        const PlainMCBenchmarkSummary plain_summary =
            run_plain_mc_benchmark(mcmcmc_order_min,
                                   mcmcmc_order_max,
                                   mcmcmc_normalization_order,
                                   plain_mc_benchmark_samples,
                                   derive_mcmcmc_seed(mcmcmc_base_seed, 0x504C41494E4D4355ULL));
        print_plain_mc_summary(plain_summary);
#endif

        if (ed_benchmark_enabled) {
            std::cout << "\nRunning exact-diagonalization benchmark from the same params...\n";
            cdet_ed::run_exact_diagonalization_benchmark_from_cdet_globals();
        }

        print_disk_lru_cache_stats();
        print_runtime_counters();
        std::cout << "\nFinished set heat-bath MCMCMC benchmark.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
