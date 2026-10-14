#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_PARAMETERS_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_PARAMETERS_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cdet/replica_chebyshev_tables.h"

inline int N1 = 0;
inline int N2 = 0;
inline MP_Real beta = MP_Real(0);

inline int cheb_nu = 0;
inline int cheb_nv = 0;
inline MP_Real tol_fpm = MP_Real(0);
inline MP_Real fpm_global_shift = MP_Real(-1);

inline MP_Real alpha_shift_up = MP_Real(0);
inline MP_Real alpha_shift_down = MP_Real(0);
inline MP_Real mu = MP_Real(0);
inline MP_Real U = MP_Real(0);
inline MP_Real branch_epsilon = MP_Real(0);
inline bool pbc = false;
inline std::string subregion_mask_name = "checkerboard";
inline int subregion_square_radius = 0;
inline int internal_integrand_threads = 1;
inline int table_build_threads = 1;
inline int green_table_drop_bits = 0;
inline std::string load_data_real_name = "MP";
inline std::string vector_real_name = "double_double";
inline std::string integrand_real_name = "double_double";
inline std::string load_data_workflow = "resident";
inline bool load_data_only = false;
inline std::string load_data_chunk_dir = "load_data_chunks";
inline int load_data_node_chunk_size = 0;
inline int load_data_sitepair_chunk_size = 0;
inline double load_data_sample_target_file_gb = 4.0;
inline double load_data_transform_target_mem_gb = 4.0;
inline std::string load_data_calculator_cache = "auto";
inline std::string mcmcmc_table_backend = "resident";
inline double mcmcmc_lru_cache_gb = 64.0;

inline int mcmcmc_order_min = 2;
inline int mcmcmc_order_max = 4;
inline int mcmcmc_envelope_order = 5;
inline int mcmcmc_normalization_order = 2;
inline std::vector<double> mcmcmc_order_lambdas{1.0, 1.0, 1.0};
inline int mcmcmc_chain_count = 8;
inline std::uint64_t mcmcmc_base_seed = 998875432210ULL;
inline int mcmcmc_lambda_warmup_steps = 10'000;
inline int mcmcmc_lambda_adapt_interval = 1'000;
inline double mcmcmc_lambda_adapt_rate = 0.25;
inline int mcmcmc_production_steps = 100'000;
inline std::string mcmcmc_seed_mode = "uniform";
inline int mcmcmc_seed_histogram_learning_steps = 0;
inline int mcmcmc_seed_histogram_tau_bins = 16;
inline double mcmcmc_seed_histogram_uniform_mix = 0.05;
inline double mcmcmc_seed_histogram_pseudo_count = 0.0;
inline bool mcmcmc_seed_histogram_learning_only = false;
inline std::string mcmcmc_seed_histogram_output_prefix;
inline bool mcmcmc_compare_uniform_seed = false;
inline bool mcmcmc_checkpoint_enabled = false;
inline std::string mcmcmc_checkpoint_dir = "mcmcmc_checkpoints";
inline std::uint64_t mcmcmc_checkpoint_interval_steps = 10'000'000ULL;
inline std::string mcmcmc_resume_mode = "off";
inline int plain_mc_benchmark_samples = 100'000;

inline bool ed_benchmark_enabled = false;
inline bool ed_benchmark_only = false;
inline int ed_threads = 1;
inline std::vector<double> ed_xi_steps{1.0e-2};
inline int ed_stencil_radius = 4;
inline int ed_order_max = 3;
inline int ed_dense_threshold = 8000;
inline std::string ed_symmetry = "auto";
inline bool ed_progress = false;
inline bool ed_block_summary = false;

inline ResidentGreenFunctionTable g_table_up;
inline ResidentGreenFunctionTable g_table_down;
inline ResidentGreenFunctionTable g_table_ordinary_up;
inline ResidentGreenFunctionTable g_table_ordinary_down;
inline DiskBackedGreenFunctionTable g_disk_table_up;
inline DiskBackedGreenFunctionTable g_disk_table_down;
inline DiskBackedGreenFunctionTable g_disk_table_ordinary_up;
inline DiskBackedGreenFunctionTable g_disk_table_ordinary_down;
inline bool g_disk_tables_initialized = false;
inline DoubleDouble g_disk_diagonal_order1_value;
inline bool g_disk_diagonal_order1_value_available = false;

inline std::array<unsigned char, 256> MSB_table{};

#endif
