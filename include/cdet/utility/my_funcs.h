#ifndef MY_FUNCS_H
#define MY_FUNCS_H

#include "cdet/utility/types.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

inline Real MY_PI_REAL() {
    static const Real pi = static_cast<Real>(get_pi());
    return pi;
}

inline Real MY_COS(Real x) {
    return std::cos(x);
}

inline Real MY_SIN(Real x) {
    return std::sin(x);
}

inline Real MY_POW(Real x, Real y) {
    return std::pow(x, y);
}

inline Real MY_SQRT(Real x) {
    return std::sqrt(x);
}

inline Real MY_ATAN2(Real y, Real x) {
    return std::atan2(y, x);
}

inline Real MY_EXP(Real x) {
    return std::exp(x);
}

inline Real MY_ROUND(Real x) {
    return std::round(x);
}

inline Real MY_LOG2(Real x) {
    return std::log2(x);
}

inline Real MY_LOG(Real x) {
    return std::log(x);
}

inline Real MY_FLOOR(Real x) {
    return std::floor(x);
}

inline Real MY_ABS(Real x) {
    return std::abs(x);
}

inline MP_Real MP_COS(const MP_Real& x) {
    return boost::multiprecision::cos(x);
}

inline MP_Real MP_SIN(const MP_Real& x) {
    return boost::multiprecision::sin(x);
}

inline MP_Real MP_POW(const MP_Real& x, const MP_Real& y) {
    return boost::multiprecision::pow(x, y);
}

inline MP_Real MP_SQRT(const MP_Real& x) {
    return boost::multiprecision::sqrt(x);
}

inline MP_Real MP_ATAN2(const MP_Real& y, const MP_Real& x) {
    return boost::multiprecision::atan2(y, x);
}

inline MP_Real MP_EXP(const MP_Real& x) {
    return boost::multiprecision::exp(x);
}

inline MP_Real MP_ROUND(const MP_Real& x) {
    return boost::multiprecision::round(x);
}

inline MP_Real MP_LOG2(const MP_Real& x) {
    return boost::multiprecision::log(x) / boost::multiprecision::log(MP_Real(2));
}

inline MP_Real MP_LOG(const MP_Real& x) {
    return boost::multiprecision::log(x);
}

inline MP_Real MP_FLOOR(const MP_Real& x) {
    return boost::multiprecision::floor(x);
}

inline MP_Real MP_ABS(const MP_Real& x) {
    return boost::multiprecision::abs(x);
}

inline MP_Real MP_CEIL(const MP_Real& x) {
    return boost::multiprecision::ceil(x);
}

inline MP_Real MP_ACOS(const MP_Real& x) {
    return boost::multiprecision::acos(x);
}

inline MP_Real MP_ASIN(const MP_Real& x) {
    return boost::multiprecision::asin(x);
}

inline MP_Real MP_ATAN(const MP_Real& x) {
    return boost::multiprecision::atan(x);
}

inline MP_Real MP_TAN(const MP_Real& x) {
    return boost::multiprecision::tan(x);
}

#define DD_ABS(x) dd_abs(x)
#define DD_FLOOR(x) dd_floor(x)
#define DD_SQRT(x) dd_sqrt(x)
#define DD_EXP(x) dd_exp(x)
#define DD_COS(x) dd_cos(x)
#define DD_SIN(x) dd_sin(x)
#define DD_POW(x, y) dd_pow(x, y)
#define DD_ATAN2(y, x) dd_atan2(y, x)
#define DD_LOG(x) dd_log(x)

template <typename T>
inline bool is_nan(const T& x) {
    if constexpr (std::is_same_v<T, DoubleDouble>) {
        return std::isnan(x.hi) || std::isnan(x.lo);
    } else if constexpr (std::is_same_v<T, MP_Real>) {
        return boost::multiprecision::isnan(x);
    } else {
        return std::isnan(x);
    }
}

template <typename T>
inline bool is_inf(const T& x) {
    if constexpr (std::is_same_v<T, DoubleDouble>) {
        return std::isinf(x.hi) || std::isinf(x.lo);
    } else if constexpr (std::is_same_v<T, MP_Real>) {
        return boost::multiprecision::isinf(x);
    } else {
        return std::isinf(x);
    }
}

template <typename T>
inline bool is_finite(const T& x) {
    if constexpr (std::is_same_v<T, DoubleDouble>) {
        return std::isfinite(x.hi) && std::isfinite(x.lo);
    } else if constexpr (std::is_same_v<T, MP_Real>) {
        return boost::multiprecision::isfinite(x);
    } else {
        return std::isfinite(x);
    }
}

#endif
