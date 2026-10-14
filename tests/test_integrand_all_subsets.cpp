#include "audit_test_utils.h"

#include "cdet/cdet_integrand_evaluator.h"
#include "cdet/load_data.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

void configure_small_physical_integrand() {
    N1 = 2;
    N2 = 2;
    beta = MP_Real("2.5");
    mu = MP_Real("0.3");
    U = MP_Real("1.7");
    alpha_shift_up = MP_Real("0.55");
    alpha_shift_down = MP_Real("0.4");
    pbc = true;
    subregion_mask_name = "first_row";
    subregion_square_radius = 0;
    cheb_nu = 18;
    cheb_nv = 18;
    branch_epsilon = MP_Real("1e-13");
    tol_fpm = MP_Real("1e-7");
    fpm_global_shift = MP_Real(-1);
    green_table_drop_bits = 0;
    load_data_real_name = "double";
    vector_real_name = "double";
    integrand_real_name = "double";
    load_data_workflow = "resident";
    load_data_calculator_cache = "off";
    mcmcmc_table_backend = "resident";
    table_build_threads = 1;
    load_data_condition_check_mode = LinearSolveConditionCheckMode::Off;
    ztransform_roundoff_config().set_epsilon0_auto();
    ztransform_roundoff_config().set_safety_factor_literal("1e4");
    load_data_into_resident_tables<double, double>();
}

void run_integrand_all_subset_tests() {
    configure_small_physical_integrand();
    std::mt19937_64 rng(0x4dcd173bULL);
    std::uniform_real_distribution<double> unit(0.01, 0.99);

    for (int envelope_trial = 0; envelope_trial < 5; ++envelope_trial) {
        std::vector<UnitVertex> envelope(6);
        for (auto& vertex : envelope) {
            vertex = UnitVertex{unit(rng), unit(rng), unit(rng), unit(rng)};
        }
        std::vector<double> all_subset;
        evaluate_cdet_density_all_subset_integrands_double_into(envelope, all_subset);
        audit_require(all_subset.size() == 64, "all-subset integrand returned the wrong size");
        for (std::uint64_t mask = 1; mask < 64; ++mask) {
            const std::vector<UnitVertex> selected = subset_vertices(envelope, mask);
            const double direct = evaluate_cdet_density_integrand_double(selected);
            const double combined = all_subset[static_cast<std::size_t>(mask)];
            audit_require_close(combined, direct, 2.0e-9, 2.0e-9,
                                "all-subset hot path differs from independent variable-order evaluation");
        }
    }
}

int main() {
    return audit_test_main("integrand_all_subsets", run_integrand_all_subset_tests);
}
