cimport cython
from cpython.ref cimport PyObject
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector
from .pyx_helpers cimport Frac, make_frac, make_frac_fast

@cython.profile(False)
cdef inline c_bool is_pow2(int64_t v) nogil:
    return v > 0 and not (v & (v - 1))


cdef float linregress_slope(uint32_t[::1] y)
cdef vector[PyObject*] walk_dedup_fast(node)

# Frac operations
cdef Frac double_as_frac(double)
cdef Frac limit_denominator(Frac, uint64_t max_denominator) nogil

cdef linear_reduce(uint32_t[::1] values, Frac slope)
