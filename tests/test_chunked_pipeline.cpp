#include "audit_test_utils.h"

#include "cdet/chunked_load_data.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

template <typename JobT, typename BeginFn, typename EndFn, typename MatchFn>
void require_exact_interval_cover(const std::vector<JobT>& jobs,
                                  std::size_t expected_end,
                                  BeginFn begin_of,
                                  EndFn end_of,
                                  MatchFn matches,
                                  const std::string& label) {
    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    for (const JobT& job : jobs) {
        if (matches(job)) {
            intervals.emplace_back(begin_of(job), end_of(job));
        }
    }
    std::sort(intervals.begin(), intervals.end());
    audit_require(!intervals.empty(), label + " has no intervals");
    std::size_t cursor = 0;
    for (const auto& [begin, end] : intervals) {
        audit_require(begin == cursor,
                      label + " contains a gap, overlap, or duplicate interval");
        audit_require(end > begin && end <= expected_end,
                      label + " contains an out-of-range or empty interval");
        cursor = end;
    }
    audit_require(cursor == expected_end, label + " does not reach the final element");
}

void require_generated_job_coverage() {
    const auto node_jobs = read_chunked_node_jobs();
    const auto transform_jobs = read_chunked_transform_jobs();
    const auto corner_jobs = read_chunked_corner_jobs();
    const std::size_t node_count = static_cast<std::size_t>(cheb_nu * cheb_nv);
    const std::size_t sitepair_count = replica_sitepair_count(N1, N2);

    std::set<int> node_indices;
    std::set<int> transform_indices;
    std::set<int> corner_indices;
    for (const auto& job : node_jobs) {
        audit_require(node_indices.insert(job.index).second, "duplicate node job index");
    }
    for (const auto& job : transform_jobs) {
        audit_require(transform_indices.insert(job.index).second, "duplicate transform job index");
    }
    for (const auto& job : corner_jobs) {
        audit_require(corner_indices.insert(job.index).second, "duplicate corner job index");
    }

    for (ChunkedGreenTableKind kind : chunked_table_kinds()) {
        for (TriangleBranch branch : chunked_branches()) {
            require_exact_interval_cover(
                node_jobs,
                node_count,
                [](const ChunkedNodeJob& job) { return static_cast<std::size_t>(job.node_begin); },
                [](const ChunkedNodeJob& job) { return static_cast<std::size_t>(job.node_end); },
                [=](const ChunkedNodeJob& job) {
                    return job.kind == kind && job.branch == branch;
                },
                std::string("node coverage ") + chunked_table_kind_name(kind) + "/" +
                    chunked_branch_name(branch));
            require_exact_interval_cover(
                transform_jobs,
                sitepair_count,
                [](const ChunkedSitepairJob& job) { return job.sitepair_begin; },
                [](const ChunkedSitepairJob& job) { return job.sitepair_end; },
                [=](const ChunkedSitepairJob& job) {
                    return job.kind == kind && job.branch == branch;
                },
                std::string("sitepair coverage ") + chunked_table_kind_name(kind) + "/" +
                    chunked_branch_name(branch));

            int corner_count = 0;
            for (const auto& job : corner_jobs) {
                if (job.kind == kind && job.branch == branch) {
                    ++corner_count;
                }
            }
            audit_require(corner_count == 1,
                          std::string("corner coverage is not exactly one for ") +
                              chunked_table_kind_name(kind) + "/" +
                              chunked_branch_name(branch));
        }
    }
}

void compare_table_vectors(const std::vector<double>& chunked,
                           const std::vector<double>& resident,
                           const std::string& label) {
    audit_require(chunked.size() == resident.size(), label + " size mismatch");
    for (std::size_t index = 0; index < chunked.size(); ++index) {
        audit_require_close(chunked[index], resident[index], 3.0e-12, 3.0e-12,
                            label + " differs at index " + std::to_string(index));
    }
}

void compare_tables(const PiecewiseChebyshevTable<double>& chunked,
                    const PiecewiseChebyshevTable<double>& resident,
                    const std::string& label) {
    compare_table_vectors(chunked.upper_coeffs, resident.upper_coeffs,
                          label + " upper coefficients");
    compare_table_vectors(chunked.lower_coeffs, resident.lower_coeffs,
                          label + " lower coefficients");
    compare_table_vectors(chunked.upper_corners, resident.upper_corners,
                          label + " upper corners");
    compare_table_vectors(chunked.lower_corners, resident.lower_corners,
                          label + " lower corners");
}

PiecewiseChebyshevTable<double> build_direct_table(ChunkedGreenTableKind kind) {
    const auto calculator = make_chunked_calculator<double>(kind);
    return build_green_function_table(calculator, cheb_nu, cheb_nv, 1);
}

void set_job_index(int index) {
    const std::string value = std::to_string(index);
    audit_require(setenv("CDET_LOAD_DATA_JOB_INDEX", value.c_str(), 1) == 0,
                  "could not set chunk test job index");
}

void run_chunked_pipeline_tests() {
    load_data_condition_check_mode = LinearSolveConditionCheckMode::Strict;
    load_data_condition_rcond_warn = 1.0e-13;
    load_data_condition_max_warnings = 16;
    N1 = 3;
    N2 = 2;
    beta = MP_Real("2.25");
    mu = MP_Real("0.35");
    U = MP_Real("2.7");
    alpha_shift_up = MP_Real("0.45");
    alpha_shift_down = MP_Real("0.2");
    pbc = true;
    subregion_mask_name = "first_row";
    subregion_square_radius = 0;
    cheb_nu = 4;
    cheb_nv = 5;
    green_table_drop_bits = 0;
    load_data_real_name = "double";
    vector_real_name = "double";
    integrand_real_name = "double";
    load_data_node_chunk_size = 7;
    load_data_sitepair_chunk_size = 17;
    load_data_calculator_cache = "off";
    table_build_threads = 1;

    const std::filesystem::path output_dir =
        std::filesystem::path("tests") / "build" / "chunked_pipeline";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    load_data_chunk_dir = output_dir.string();

    run_chunked_plan_for_real<double>();
    require_generated_job_coverage();

    for (const auto& job : read_chunked_node_jobs()) {
        set_job_index(job.index);
        run_chunked_sample_node_job_for_real<double>();
    }
    for (const auto& job : read_chunked_corner_jobs()) {
        set_job_index(job.index);
        run_chunked_corner_job_for_real<double>();
    }
    for (const auto& job : read_chunked_transform_jobs()) {
        set_job_index(job.index);
        run_chunked_transform_sitepair_job_for_real<double>();
    }
    run_chunked_diagonal_reduce_for_real<double>();
    validate_chunked_outputs_for_real<double>();

    const std::array<ChunkedGreenTableKind, 4> kinds = chunked_table_kinds();
    for (ChunkedGreenTableKind kind : kinds) {
        const auto chunked = load_chunked_table_for_kind<double>(kind);
        const auto resident = build_direct_table(kind);
        compare_tables(chunked, resident, chunked_table_kind_name(kind));
    }

    g_table_up = load_chunked_table_for_kind<double>(ChunkedGreenTableKind::TwistedUp);
    g_table_down = load_chunked_table_for_kind<double>(ChunkedGreenTableKind::TwistedDown);
    g_table_ordinary_up = load_chunked_table_for_kind<double>(ChunkedGreenTableKind::OrdinaryUp);
    g_table_ordinary_down = load_chunked_table_for_kind<double>(ChunkedGreenTableKind::OrdinaryDown);
    mcmcmc_table_backend = "resident";
    const DoubleDouble resident_order1 = normalized_log_partition_first_order_density_dd();
    const DoubleDouble chunked_order1 = read_chunked_diagonal_observable_value();
    audit_require_close(chunked_order1.to_double(), resident_order1.to_double(),
                        3.0e-12, 3.0e-12,
                        "chunked diagonal reduction disagrees with resident integration");

    unsetenv("CDET_LOAD_DATA_JOB_INDEX");
}

int main() {
    return audit_test_main("chunked_pipeline", run_chunked_pipeline_tests);
}
