#pragma once

#include <stdint.h>

namespace cxxlookup {

inline uint32_t bit_length(uint64_t v) { return 64 - __builtin_clzll(v); }

}  // namespace cxxlookup
