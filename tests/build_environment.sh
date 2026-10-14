#!/usr/bin/env bash
# Shared include discovery for the audit-only test launchers.
if [[ -n "${BOOST_INCLUDE_DIR:-}" ]]; then
    boost_include="$BOOST_INCLUDE_DIR"
else
    boost_include=""
    for candidate in /opt/homebrew/opt/boost/include /opt/homebrew/include /usr/local/include /usr/include; do
        if [[ -f "$candidate/boost/multiprecision/cpp_dec_float.hpp" ]]; then
            boost_include="$candidate"
            break
        fi
    done
fi
if [[ -z "$boost_include" ]]; then
    echo "Boost headers not found; set BOOST_INCLUDE_DIR." >&2
    exit 2
fi
