/*
Copyright (c) 2014-2022, chys <admin@CHYS.INFO>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of chys <admin@CHYS.INFO> nor the names of other
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <stdint.h>

#include <cstdlib>
#include <utility>

namespace cxxlookup {

inline uint32_t bit_length(uint64_t v) { return 64 - __builtin_clzll(v); }

inline bool is_pow2(int64_t v) { return v && !(v & (v - 1)); }

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
