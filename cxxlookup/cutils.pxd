cimport cython
from cpython.ref cimport PyObject
from libc.stdint cimport int64_t, uint32_t
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector

@cython.profile(False)
cdef inline c_bool is_pow2(int64_t v) nogil:
    return v > 0 and not (v & (v - 1))


cdef float linregress_slope(uint32_t[::1] y)
cdef vector[PyObject*] walk_dedup_fast(node)
