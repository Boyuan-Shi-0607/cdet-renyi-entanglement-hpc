#ifndef TYPES_H
#define TYPES_H

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef MP_PRECISION_DIGITS10
#define MP_PRECISION_DIGITS10 100
#endif

using MP_Real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<MP_PRECISION_DIGITS10>>;

inline MP_Real get_pi() {
    static const MP_Real pi = boost::multiprecision::acos(MP_Real(-1));
    return pi;
}

inline MP_Real get_two_pi() {
    static const MP_Real two_pi = MP_Real(2) * get_pi();
    return two_pi;
}

struct DoubleDouble {
    double hi;
    double lo;

    constexpr DoubleDouble() noexcept : hi(0.0), lo(0.0) {}
    constexpr explicit DoubleDouble(int x) noexcept : hi(static_cast<double>(x)), lo(0.0) {}
    constexpr explicit DoubleDouble(unsigned long long x) noexcept : hi(static_cast<double>(x)), lo(0.0) {}
    constexpr explicit DoubleDouble(long long x) noexcept : hi(static_cast<double>(x)), lo(0.0) {}
    constexpr explicit DoubleDouble(unsigned long x) noexcept : hi(static_cast<double>(x)), lo(0.0) {}
    constexpr explicit DoubleDouble(double x) noexcept : hi(x), lo(0.0) {}

    explicit DoubleDouble(const MP_Real& x) noexcept {
        hi = static_cast<double>(x);
        const MP_Real residual = x - MP_Real(hi);
        lo = static_cast<double>(residual);
    }

    constexpr DoubleDouble(double h, double l) noexcept : hi(h), lo(l) {}

    static inline constexpr double quick_two_sum(double a, double b, double& err) noexcept {
        const double s = a + b;
        err = b - (s - a);
        return s;
    }

    static inline constexpr double two_sum(double a, double b, double& err) noexcept {
        const double s = a + b;
        const double bb = s - a;
        err = (a - (s - bb)) + (b - bb);
        return s;
    }

    static inline constexpr double two_prod(double a, double b, double& err) noexcept {
        const double p = a * b;
        err = std::fma(a, b, -p);
        return p;
    }

    constexpr void renormalize() noexcept {
        const double s = hi + lo;
        lo = lo - (s - hi);
        hi = s;
    }

    friend constexpr DoubleDouble operator+(const DoubleDouble& x, const DoubleDouble& y) noexcept {
        double s1;
        double s2;
        double t1;
        double t2;
        s1 = two_sum(x.hi, y.hi, s2);
        t1 = two_sum(x.lo, y.lo, t2);
        s2 += t1;
        s1 = quick_two_sum(s1, s2, s2);
        s2 += t2;
        DoubleDouble r;
        r.hi = quick_two_sum(s1, s2, r.lo);
        return r;
    }

    friend constexpr DoubleDouble operator-(const DoubleDouble& x, const DoubleDouble& y) noexcept {
        double s1;
        double s2;
        double t1;
        double t2;
        s1 = two_sum(x.hi, -y.hi, s2);
        t1 = two_sum(x.lo, -y.lo, t2);
        s2 += t1;
        s1 = quick_two_sum(s1, s2, s2);
        s2 += t2;
        DoubleDouble r;
        r.hi = quick_two_sum(s1, s2, r.lo);
        return r;
    }

    friend constexpr DoubleDouble operator*(const DoubleDouble& x, const DoubleDouble& y) noexcept {
        double p;
        double e;
        p = two_prod(x.hi, y.hi, e);
        e += x.hi * y.lo + x.lo * y.hi + x.lo * y.lo;
        DoubleDouble r;
        r.hi = quick_two_sum(p, e, r.lo);
        return r;
    }

    friend DoubleDouble operator/(const DoubleDouble& x, const DoubleDouble& y) {
        const double q1 = x.hi / y.hi;
        const DoubleDouble r = x - DoubleDouble(q1) * y;
        const double q2 = (r.hi + r.lo) / y.hi;
        DoubleDouble q = DoubleDouble(q1) + DoubleDouble(q2);
        q.renormalize();
        return q;
    }

    friend DoubleDouble operator/(const DoubleDouble& x, int n) {
        return x / DoubleDouble(n);
    }

    constexpr DoubleDouble operator-() const noexcept {
        return DoubleDouble(-hi, -lo);
    }

    DoubleDouble& operator+=(const DoubleDouble& other) {
        return *this = *this + other;
    }

    DoubleDouble& operator-=(const DoubleDouble& other) {
        return *this = *this - other;
    }

    DoubleDouble& operator*=(const DoubleDouble& other) {
        return *this = *this * other;
    }

    DoubleDouble& operator/=(const DoubleDouble& other) {
        return *this = *this / other;
    }

    constexpr double to_double() const noexcept {
        return hi + lo;
    }

    constexpr operator double() const noexcept {
        return to_double();
    }

    MP_Real to_mp_real() const {
        return MP_Real(hi) + MP_Real(lo);
    }

    friend bool operator<(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return (lhs.hi < rhs.hi) || (lhs.hi == rhs.hi && lhs.lo < rhs.lo);
    }

    friend bool operator>(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return rhs < lhs;
    }

    friend bool operator<=(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return !(lhs > rhs);
    }

    friend bool operator>=(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return !(lhs < rhs);
    }

    friend bool operator==(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
    }

    friend bool operator!=(const DoubleDouble& lhs, const DoubleDouble& rhs) {
        return !(lhs == rhs);
    }
};

inline DoubleDouble DD_PI() {
    static const DoubleDouble pi_dd(get_pi());
    return pi_dd;
}

inline DoubleDouble DD_TWO_PI() {
    static const DoubleDouble two_pi_dd(get_two_pi());
    return two_pi_dd;
}

inline int dd_floor_to_int(const DoubleDouble& x) {
    if (!std::isfinite(x.hi)) {
        throw std::domain_error("dd_floor_to_int: non-finite input");
    }

    double hi_floor = std::floor(x.hi);
    if (hi_floor == x.hi && x.lo < 0.0) {
        hi_floor -= 1.0;
    }

    constexpr auto int_min_double = static_cast<double>(std::numeric_limits<int>::min());
    constexpr auto int_max_double = static_cast<double>(std::numeric_limits<int>::max());
    if (hi_floor < int_min_double || hi_floor > int_max_double) {
        throw std::overflow_error("dd_floor_to_int: result out of int range");
    }

    return static_cast<int>(hi_floor);
}

inline int DD_FLOOR_TO_INT(const DoubleDouble& x) {
    return dd_floor_to_int(x);
}

inline DoubleDouble dd_floor(const DoubleDouble& x) {
    return DoubleDouble(dd_floor_to_int(x));
}

inline DoubleDouble dd_abs(const DoubleDouble& x) {
    return (x.hi < 0.0) ? -x : x;
}

inline DoubleDouble dd_sqrt(const DoubleDouble& x) {
    return DoubleDouble(boost::multiprecision::sqrt(x.to_mp_real()));
}

inline DoubleDouble dd_ldexp(const DoubleDouble& x, int exponent) {
    return DoubleDouble(std::ldexp(x.hi, exponent), std::ldexp(x.lo, exponent));
}

inline DoubleDouble dd_exp(const DoubleDouble& x) {
    if (x.hi == 0.0 && x.lo == 0.0) {
        return DoubleDouble(1);
    }
    if (x.hi <= -745.1332191019411) {
        return DoubleDouble(0);
    }
    if (x.hi >= 709.782712893384) {
        return DoubleDouble(std::numeric_limits<double>::infinity());
    }

    static constexpr int reduction_bits = 9;
    static constexpr int reduction_factor = 1 << reduction_bits;
    static constexpr DoubleDouble log2_dd(6.93147180559945286227e-01,
                                          2.31904681384629955842e-17);
    const int exponent =
        static_cast<int>(std::floor((x.hi + x.lo) / log2_dd.hi + 0.5));
    const DoubleDouble reduced =
        (x - DoubleDouble(exponent) * log2_dd) / DoubleDouble(reduction_factor);

    DoubleDouble term = reduced;
    DoubleDouble sum = DoubleDouble(1) + term;
    for (int order = 2; order <= 12; ++order) {
        term = (term * reduced) / DoubleDouble(order);
        sum += term;
    }

    for (int i = 0; i < reduction_bits; ++i) {
        sum *= sum;
    }

    return dd_ldexp(sum, exponent);
}

inline DoubleDouble dd_cos(const DoubleDouble& x) {
    return DoubleDouble(boost::multiprecision::cos(x.to_mp_real()));
}

inline DoubleDouble dd_sin(const DoubleDouble& x) {
    return DoubleDouble(boost::multiprecision::sin(x.to_mp_real()));
}

inline DoubleDouble dd_pow(const DoubleDouble& x, const DoubleDouble& y) {
    return DoubleDouble(boost::multiprecision::pow(x.to_mp_real(), y.to_mp_real()));
}

inline DoubleDouble dd_atan2(const DoubleDouble& y, const DoubleDouble& x) {
    return DoubleDouble(boost::multiprecision::atan2(y.to_mp_real(), x.to_mp_real()));
}

inline DoubleDouble dd_log(const DoubleDouble& x) {
    return DoubleDouble(boost::multiprecision::log(x.to_mp_real()));
}

using Real = double;

using MP_Real1DVector = std::vector<MP_Real>;
using MP_Real2DVector = std::vector<MP_Real1DVector>;

using DD1DVector = std::vector<DoubleDouble>;
using DD2DVector = std::vector<DD1DVector>;

using Real1DVector = std::vector<Real>;
using Real2DVector = std::vector<Real1DVector>;

using Int1DVector = std::vector<int>;
using Int2DVector = std::vector<std::vector<int>>;

inline std::ostream& operator<<(std::ostream& os, const DoubleDouble& x) {
    os << std::fixed << std::setprecision(32) << x.to_double();
    return os;
}

#endif
