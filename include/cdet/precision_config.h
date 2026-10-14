#ifndef CDET_PRECISION_CONFIG_H
#define CDET_PRECISION_CONFIG_H

#include "cdet/utility/types.h"

#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

enum class IntegrandRealType {
    Double,
    LongDouble,
    DoubleDouble,
};

enum class LoadDataRealType {
    MP,
    Double,
    LongDouble,
    DoubleDouble,
};

inline std::string normalize_real_type_name(std::string name) {
    for (char& ch : name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-') {
            ch = '_';
        }
    }
    return name;
}

inline LoadDataRealType parse_load_data_real_type(const std::string& raw_name) {
    const std::string name = normalize_real_type_name(raw_name);
    if (name == "mp" || name == "multiprecision" || name == "boost" || name == "cpp_dec_float") {
        return LoadDataRealType::MP;
    }
    if (name == "double") {
        return LoadDataRealType::Double;
    }
    if (name == "long_double" || name == "longdouble") {
        return LoadDataRealType::LongDouble;
    }
    if (name == "double_double" || name == "doubledouble" || name == "dd") {
        return LoadDataRealType::DoubleDouble;
    }
    throw std::runtime_error(
        "Unknown LOAD_DATA_REAL: " + raw_name + ". Supported values are MP, double, long_double, double_double.");
}

inline const char* load_data_real_type_to_string(LoadDataRealType type) {
    switch (type) {
        case LoadDataRealType::MP:
            return "MP";
        case LoadDataRealType::Double:
            return "double";
        case LoadDataRealType::LongDouble:
            return "long_double";
        case LoadDataRealType::DoubleDouble:
            return "double_double";
    }
    throw std::runtime_error("Unhandled LoadDataRealType enum value");
}

inline IntegrandRealType parse_integrand_real_type(const std::string& raw_name) {
    const std::string name = normalize_real_type_name(raw_name);
    if (name == "double") {
        return IntegrandRealType::Double;
    }
    if (name == "long_double" || name == "longdouble") {
        return IntegrandRealType::LongDouble;
    }
    if (name == "double_double" || name == "doubledouble" || name == "dd") {
        return IntegrandRealType::DoubleDouble;
    }
    throw std::runtime_error(
        "Unknown INTEGRAND_REAL: " + raw_name + ". Supported values are double, long_double, double_double.");
}

inline IntegrandRealType parse_vector_real_type(const std::string& raw_name) {
    const std::string name = normalize_real_type_name(raw_name);
    if (name == "double") {
        return IntegrandRealType::Double;
    }
    if (name == "long_double" || name == "longdouble") {
        return IntegrandRealType::LongDouble;
    }
    if (name == "double_double" || name == "doubledouble" || name == "dd") {
        return IntegrandRealType::DoubleDouble;
    }
    throw std::runtime_error(
        "Unknown VECTOR_REAL: " + raw_name + ". Supported values are double, long_double, double_double.");
}

inline const char* integrand_real_type_to_string(IntegrandRealType type) {
    switch (type) {
        case IntegrandRealType::Double:
            return "double";
        case IntegrandRealType::LongDouble:
            return "long_double";
        case IntegrandRealType::DoubleDouble:
            return "double_double";
    }
    throw std::runtime_error("Unhandled IntegrandRealType enum value");
}

inline const char* vector_real_type_to_string(IntegrandRealType type) {
    return integrand_real_type_to_string(type);
}

inline DoubleDouble double_double_from_long_double_value(long double value) {
    const double hi = static_cast<double>(value);
    const long double residual = value - static_cast<long double>(hi);
    return DoubleDouble(hi, static_cast<double>(residual));
}

template <typename TargetT, typename SourceT>
inline TargetT real_precision_cast(const SourceT& value) {
    if constexpr (std::is_same_v<TargetT, SourceT>) {
        return value;
    } else if constexpr (std::is_same_v<TargetT, DoubleDouble>) {
        if constexpr (std::is_same_v<SourceT, MP_Real>) {
            return DoubleDouble(value);
        } else if constexpr (std::is_same_v<SourceT, long double>) {
            return double_double_from_long_double_value(value);
        } else {
            return DoubleDouble(static_cast<double>(value));
        }
    } else if constexpr (std::is_same_v<TargetT, long double>) {
        if constexpr (std::is_same_v<SourceT, DoubleDouble>) {
            return static_cast<long double>(value.hi) + static_cast<long double>(value.lo);
        } else {
            return static_cast<long double>(value);
        }
    } else if constexpr (std::is_same_v<TargetT, double>) {
        if constexpr (std::is_same_v<SourceT, DoubleDouble>) {
            return value.to_double();
        } else {
            return static_cast<double>(value);
        }
    } else if constexpr (std::is_same_v<TargetT, MP_Real>) {
        if constexpr (std::is_same_v<SourceT, DoubleDouble>) {
            return value.to_mp_real();
        } else {
            return MP_Real(value);
        }
    } else {
        return static_cast<TargetT>(value);
    }
}

template <typename RealT>
inline RealT parse_real_literal_for_precision(const std::string& literal) {
    if constexpr (std::is_same_v<RealT, double>) {
        return std::stod(literal);
    } else if constexpr (std::is_same_v<RealT, long double>) {
        return std::stold(literal);
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DoubleDouble(MP_Real(literal));
    } else if constexpr (std::is_same_v<RealT, MP_Real>) {
        return MP_Real(literal);
    } else {
        return RealT(literal);
    }
}

template <typename RealT>
inline RealT default_ztransform_epsilon0_for_precision() {
    if constexpr (std::is_same_v<RealT, double>) {
        return 2.0e-16;
    } else if constexpr (std::is_same_v<RealT, long double>) {
        return std::numeric_limits<long double>::epsilon();
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return DoubleDouble(MP_Real("1e-31"));
    } else {
        static_assert(std::is_same_v<RealT, double> ||
                      std::is_same_v<RealT, long double> ||
                      std::is_same_v<RealT, DoubleDouble>,
                      "Unsupported integrand real type");
    }
}

template <typename RealT>
inline MP_Real real_to_mp_real_for_display(const RealT& value) {
    if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return value.to_mp_real();
    } else {
        return MP_Real(value);
    }
}

inline std::string format_mp_real_compact(const MP_Real& value) {
    std::ostringstream out;
    out << std::setprecision(18) << value;
    return out.str();
}

template <typename RealT>
inline std::string real_value_to_string(const RealT& value) {
    return format_mp_real_compact(real_to_mp_real_for_display(value));
}

class ZTransformRoundoffConfig {
public:
    void set_epsilon0_auto() {
        epsilon0_is_auto_ = true;
        epsilon0_literal_ = "auto";
    }

    void set_epsilon0_literal(std::string literal) {
        validate_positive_literal(literal, "ZTRANSFORM_EPSILON0");
        epsilon0_is_auto_ = false;
        epsilon0_literal_ = std::move(literal);
    }

    void set_safety_factor_literal(std::string literal) {
        validate_nonnegative_literal(literal, "ZTRANSFORM_SAFETY_FACTOR");
        safety_factor_literal_ = std::move(literal);
    }

    bool epsilon0_is_auto() const {
        return epsilon0_is_auto_;
    }

    const std::string& epsilon0_literal() const {
        return epsilon0_literal_;
    }

    const std::string& safety_factor_literal() const {
        return safety_factor_literal_;
    }

    template <typename RealT>
    RealT epsilon0() const {
        RealT value;
        if (epsilon0_is_auto_) {
            value = default_ztransform_epsilon0_for_precision<RealT>();
        } else {
            value = parse_real_literal_for_precision<RealT>(epsilon0_literal_);
        }
        if (!(value > RealT(0))) {
            throw std::runtime_error("ZTRANSFORM_EPSILON0 is not positive in the selected integrand precision");
        }
        return value;
    }

    template <typename RealT>
    RealT safety_factor() const {
        const RealT value = parse_real_literal_for_precision<RealT>(safety_factor_literal_);
        if (!(value >= RealT(0))) {
            throw std::runtime_error("ZTRANSFORM_SAFETY_FACTOR is negative in the selected integrand precision");
        }
        return value;
    }

    template <typename RealT>
    RealT threshold() const {
        return epsilon0<RealT>() * safety_factor<RealT>();
    }

private:
    static void validate_positive_literal(const std::string& literal, const char* name) {
        if (!(MP_Real(literal) > MP_Real(0))) {
            throw std::runtime_error(std::string(name) + " must be positive or auto");
        }
    }

    static void validate_nonnegative_literal(const std::string& literal, const char* name) {
        if (!(MP_Real(literal) >= MP_Real(0))) {
            throw std::runtime_error(std::string(name) + " must be nonnegative");
        }
    }

    bool epsilon0_is_auto_ = true;
    std::string epsilon0_literal_ = "auto";
    std::string safety_factor_literal_ = "1e4";
};

inline ZTransformRoundoffConfig& ztransform_roundoff_config() {
    static ZTransformRoundoffConfig config;
    return config;
}

template <typename RealT>
inline std::string ztransform_epsilon0_display_string() {
    const auto& config = ztransform_roundoff_config();
    const std::string value = real_value_to_string(config.epsilon0<RealT>());
    if (config.epsilon0_is_auto()) {
        return "auto -> " + value;
    }
    return value;
}

template <typename RealT>
inline std::string ztransform_safety_factor_display_string() {
    return real_value_to_string(ztransform_roundoff_config().safety_factor<RealT>());
}

template <typename RealT>
inline std::string ztransform_threshold_display_string() {
    return real_value_to_string(ztransform_roundoff_config().threshold<RealT>());
}

inline std::string ztransform_epsilon0_display_string(IntegrandRealType type) {
    switch (type) {
        case IntegrandRealType::Double:
            return ztransform_epsilon0_display_string<double>();
        case IntegrandRealType::LongDouble:
            return ztransform_epsilon0_display_string<long double>();
        case IntegrandRealType::DoubleDouble:
            return ztransform_epsilon0_display_string<DoubleDouble>();
    }
    throw std::runtime_error("Unhandled IntegrandRealType enum value");
}

inline std::string ztransform_safety_factor_display_string(IntegrandRealType type) {
    switch (type) {
        case IntegrandRealType::Double:
            return ztransform_safety_factor_display_string<double>();
        case IntegrandRealType::LongDouble:
            return ztransform_safety_factor_display_string<long double>();
        case IntegrandRealType::DoubleDouble:
            return ztransform_safety_factor_display_string<DoubleDouble>();
    }
    throw std::runtime_error("Unhandled IntegrandRealType enum value");
}

inline std::string ztransform_threshold_display_string(IntegrandRealType type) {
    switch (type) {
        case IntegrandRealType::Double:
            return ztransform_threshold_display_string<double>();
        case IntegrandRealType::LongDouble:
            return ztransform_threshold_display_string<long double>();
        case IntegrandRealType::DoubleDouble:
            return ztransform_threshold_display_string<DoubleDouble>();
    }
    throw std::runtime_error("Unhandled IntegrandRealType enum value");
}

template <typename RealT>
inline const char* real_type_name() {
    if constexpr (std::is_same_v<RealT, double>) {
        return "double";
    } else if constexpr (std::is_same_v<RealT, long double>) {
        return "long double";
    } else if constexpr (std::is_same_v<RealT, DoubleDouble>) {
        return "double-double";
    } else if constexpr (std::is_same_v<RealT, MP_Real>) {
        return "boost::multiprecision::cpp_dec_float";
    } else {
        return "unknown";
    }
}

#endif
