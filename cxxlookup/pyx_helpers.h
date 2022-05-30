#pragma once

#include <stdint.h>

#include <cstdlib>
#include <utility>

namespace cxxlookup {

inline uint32_t bit_length(uint64_t v) { return 64 - __builtin_clzll(v); }

constexpr uint64_t gcd(uint64_t a, uint64_t b) {
  if (a < b) std::swap(a, b);
  while (b) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

struct Frac {
  int64_t numerator;
  uint64_t denominator;

  double to_double() const { return double(numerator) / denominator; }
};

inline Frac make_frac_fast(int64_t n, uint64_t d) { return Frac{n, d}; }

inline Frac make_frac(int64_t n, uint64_t d) {
  uint64_t g = gcd(std::abs(n), d);
  return make_frac_fast(n / g, d / g);
}

}  // namespace cxxlookup
