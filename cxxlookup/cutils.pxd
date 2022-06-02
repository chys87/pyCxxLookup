cimport cython
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector
from numpy cimport ndarray

from .pyx_helpers cimport Frac, make_frac, make_frac_fast

@cython.profile(False)
cdef inline c_bool is_pow2(int64_t v) nogil:
    return v > 0 and not (v & (v - 1))


cdef float linregress_slope(uint32_t[::1] y)

# Frac operations
cdef Frac double_as_frac(double)
cdef Frac limit_denominator(Frac, uint64_t max_denominator) nogil

cdef linear_reduce(uint32_t[::1] values, Frac slope)


cdef class PrepareStrideResult:
    cdef ndarray base_values
    cdef ndarray delta
    cdef uint32_t delta_max
    cdef uint32_t delta_nonzeros


cdef PrepareStrideResult prepare_strides(uint32_t[::1] values, uint32_t stride)
