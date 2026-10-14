#ifndef CDET_LINEAR_SOLVE_DIAGNOSTICS_H
#define CDET_LINEAR_SOLVE_DIAGNOSTICS_H

#include <atomic>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

enum class LinearSolveConditionCheckMode {
    Off,
    Auto,
    Strict,
};

struct LinearSolveDiagnosticContext {
    const char* label = "complex_matrix_solve";
    const char* branch = "";
    int sector = -1;
    double tau_fraction = -1.0;
    double tau_prime_fraction = -1.0;
    bool rhs_is_diagonal = false;
};

inline LinearSolveConditionCheckMode load_data_condition_check_mode =
    LinearSolveConditionCheckMode::Auto;
inline double load_data_condition_rcond_warn = 1.0e-12;
inline int load_data_condition_max_warnings = 8;
inline std::atomic<int> load_data_condition_warning_count{0};

inline const char* linear_solve_condition_check_mode_to_string(
    LinearSolveConditionCheckMode mode) {
    switch (mode) {
        case LinearSolveConditionCheckMode::Off:
            return "off";
        case LinearSolveConditionCheckMode::Auto:
            return "auto";
        case LinearSolveConditionCheckMode::Strict:
            return "strict";
    }
    throw std::runtime_error("Unhandled LinearSolveConditionCheckMode enum value");
}

inline LinearSolveConditionCheckMode parse_linear_solve_condition_check_mode(
    std::string name) {
    for (char& ch : name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-') {
            ch = '_';
        }
    }

    if (name == "off" || name == "false" || name == "0" || name == "none") {
        return LinearSolveConditionCheckMode::Off;
    }
    if (name == "auto" || name == "warn" || name == "warning" ||
        name == "on" || name == "true" || name == "1") {
        return LinearSolveConditionCheckMode::Auto;
    }
    if (name == "strict" || name == "error" || name == "throw") {
        return LinearSolveConditionCheckMode::Strict;
    }

    throw std::runtime_error(
        "LOAD_DATA_CONDITION_CHECK must be off, auto, or strict");
}

inline bool linear_solve_condition_check_enabled() {
    return load_data_condition_check_mode != LinearSolveConditionCheckMode::Off;
}

inline bool linear_solve_condition_check_strict() {
    return load_data_condition_check_mode == LinearSolveConditionCheckMode::Strict;
}

inline std::mutex& linear_solve_condition_warning_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::string format_linear_solve_condition_context(
    const LinearSolveDiagnosticContext& context,
    int n) {
    std::ostringstream out;
    out << context.label << " n=" << n;
    if (context.branch != nullptr && context.branch[0] != '\0') {
        out << " branch=" << context.branch;
    }
    if (context.sector >= 0) {
        out << " sector=" << context.sector;
    }
    if (context.tau_fraction >= 0.0) {
        out << " tau/beta=" << context.tau_fraction;
    }
    if (context.tau_prime_fraction >= 0.0) {
        out << " tau_prime/beta=" << context.tau_prime_fraction;
    }
    return out.str();
}

inline void handle_linear_solve_condition_estimate(
    const LinearSolveDiagnosticContext& context,
    int n,
    double rcond,
    const char* estimator) {
    if (!linear_solve_condition_check_enabled()) {
        return;
    }

    const bool dangerous =
        !std::isfinite(rcond) || rcond <= load_data_condition_rcond_warn;
    if (!dangerous) {
        return;
    }

    const double condition_number =
        (std::isfinite(rcond) && rcond > 0.0)
            ? (1.0 / rcond)
            : std::numeric_limits<double>::infinity();

    std::ostringstream message;
    message << "Load-data linear solve is ill-conditioned: "
            << format_linear_solve_condition_context(context, n)
            << ", estimator=" << estimator
            << ", rcond=" << rcond
            << ", cond~=" << condition_number
            << ", threshold=" << load_data_condition_rcond_warn;

    if (linear_solve_condition_check_strict()) {
        throw std::runtime_error(message.str());
    }

    const int warning_index =
        load_data_condition_warning_count.fetch_add(1, std::memory_order_relaxed);
    if (warning_index < load_data_condition_max_warnings) {
        std::lock_guard<std::mutex> lock(linear_solve_condition_warning_mutex());
        std::cerr << "Warning: " << message.str() << '\n';
        if (warning_index + 1 == load_data_condition_max_warnings) {
            std::cerr << "Warning: further load-data condition warnings are suppressed\n";
        }
    }
}

#endif
