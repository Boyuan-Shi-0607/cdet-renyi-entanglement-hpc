#include "audit_test_utils.h"

#include "cdet/g_funcs.h"
#include "cdet/replica_chebyshev_tables.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

struct ChebyshevErrorSummary {
    double max_abs = 0.0;
    long double sum_squared = 0.0L;
    std::uint64_t count = 0;

    void add(double error) {
        max_abs = std::max(max_abs, std::abs(error));
        sum_squared += static_cast<long double>(error) * error;
        ++count;
    }

    double rms() const {
        return std::sqrt(static_cast<double>(sum_squared / count));
    }
};

ChebyshevErrorSummary measure_table_error(const ReplicaGreenCalculator<double>& calculator,
                                          int degree,
                                          const std::vector<std::array<double, 8>>& probes) {
    const auto table = build_green_function_table(calculator, degree, degree, 1);
    ChebyshevErrorSummary summary;
    for (const auto& probe : probes) {
        const double tau1 = probe[0] * calculator.beta();
        const double tau2 = probe[1] * calculator.beta();
        const int replica1 = static_cast<int>(probe[2]);
        const int replica2 = static_cast<int>(probe[3]);
        const int x1 = static_cast<int>(probe[4]);
        const int y1 = static_cast<int>(probe[5]);
        const int x2 = static_cast<int>(probe[6]);
        const int y2 = static_cast<int>(probe[7]);
        const bool upper = tau1 > tau2;
        const TriangleBranch branch = upper ? TriangleBranch::Upper : TriangleBranch::Lower;
        const auto exact_matrix =
            calculator.green_replica_matrix(replica1, replica2, tau1, tau2, branch);
        const int site1 = site_index(x1, y1, calculator.N1(), calculator.N2());
        const int site2 = site_index(x2, y2, calculator.N1(), calculator.N2());
        const auto descriptor = prepare_green_lookup_descriptor(
            replica1, replica2, tau1, tau2, x1, y1, x2, y2,
            calculator.beta(), 1.0e-13, degree, degree,
            calculator.N1(), calculator.N2(), upper);
        const auto value_pair = evaluate_green_pair_descriptor(
            descriptor, table, table, degree, degree);
        summary.add(value_pair.first - exact_matrix(site1, site2));
    }
    return summary;
}

void run_chebyshev_accuracy_tests() {
    load_data_condition_check_mode = LinearSolveConditionCheckMode::Off;
    ReplicaGreenCalculator<double> calculator(
        3, 3, 8.0, -0.4, true, ReplicaSubregionMask::FirstRow, 0);

    std::mt19937_64 rng(0x5e918adbULL);
    std::uniform_real_distribution<double> unit(0.02, 0.98);
    std::uniform_int_distribution<int> replica(0, 1);
    std::uniform_int_distribution<int> coordinate(0, 2);
    std::vector<std::array<double, 8>> probes;
    probes.reserve(240);
    for (int sample = 0; sample < 240; ++sample) {
        double tau1 = unit(rng);
        double tau2 = unit(rng);
        if (std::abs(tau1 - tau2) < 0.03) {
            tau2 = std::fmod(tau2 + 0.17, 0.96) + 0.02;
        }
        probes.push_back({tau1,
                          tau2,
                          static_cast<double>(replica(rng)),
                          static_cast<double>(replica(rng)),
                          static_cast<double>(coordinate(rng)),
                          static_cast<double>(coordinate(rng)),
                          static_cast<double>(coordinate(rng)),
                          static_cast<double>(coordinate(rng))});
    }

    const std::array<int, 4> degrees = {8, 12, 18, 25};
    std::array<ChebyshevErrorSummary, 4> errors;
    for (std::size_t index = 0; index < degrees.size(); ++index) {
        errors[index] = measure_table_error(calculator, degrees[index], probes);
        std::cout << "degree " << degrees[index]
                  << ": max_abs=" << errors[index].max_abs
                  << ", rms=" << errors[index].rms() << '\n';
    }

    audit_require(errors.back().max_abs < 2.0e-7,
                  "degree-25 Chebyshev table is not accurate to the expected scale");
    audit_require(errors.back().rms() < 2.0e-8,
                  "degree-25 Chebyshev RMS error is too large");
    audit_require(errors.back().rms() < errors.front().rms() * 1.0e-3,
                  "Chebyshev error did not decrease substantially with degree");

    const auto table = build_green_function_table(calculator, 25, 25, 1);
    for (TriangleBranch branch : {TriangleBranch::Upper, TriangleBranch::Lower}) {
        const auto exact = calculator.green_replica_matrix(0, 1, 0.0, 0.0, branch);
        const auto descriptor = prepare_green_lookup_descriptor(
            0, 1, 0.0, 0.0, 1, 1, 2, 0,
            calculator.beta(), 1.0e-13, 25, 25, 3, 3,
            branch == TriangleBranch::Upper);
        audit_require(descriptor.use_corner, "zero-time point did not use the corner table");
        const auto evaluated = evaluate_green_pair_descriptor(descriptor, table, table, 25, 25);
        audit_require_close(evaluated.first,
                            exact(site_index(1, 1, 3, 3), site_index(2, 0, 3, 3)),
                            2.0e-13, 2.0e-13,
                            "corner table does not equal the exact zero-time kernel");
    }
}

int main() {
    return audit_test_main("chebyshev_accuracy", run_chebyshev_accuracy_tests);
}
