# distutils: language=c++
# cython: language_level=3
# cython: profile=True
# cython: cdivision=True

# Copyright (c) 2014-2022, chys <admin@CHYS.INFO>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   Neither the name of chys <admin@CHYS.INFO> nor the names of other
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np

cimport cython
from libc.stdint cimport int64_t, uintptr_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as c_bool
from libcpp.utility cimport pair

from .pyx_helpers cimport abs as cpp_abs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(False)
cdef inline float linregress_slope_internal(uint32_t[::1] y, int n) nogil:
    #          sigma (y_i - y_bar)(x_i - x_bar)
    # slope = ---------------------------------
    #           sigma (x_i - x_bar)^2
    #
    #          sigma  y_i (x_i - x_bar)
    #       = ---------------------------
    #          sigma x_i^2 - n * x_bar^2
    #
    # because x_i = i, so denominator = n(n^2-1)/12

    cdef int i
    cdef double nd = n
    cdef double x_bar = (nd - 1) * .5
    cdef double a = 0
    for i in range(n):
        a += y[i] * (i - x_bar)
    return 12 * a / (nd * (nd * nd - 1))


@cython.nonecheck(False)
cdef float linregress_slope(uint32_t[::1] y):
    '''Compute linear regression slope

    Roughly equivalent to
    scipy.stats.linregress(np.arange(len(y), np.float32),
                           y.astype(np.float32)).slope,
    but much faster.
    '''
    cdef int n = len(y)

    if n < 1024:
        if n < 2:
            return 0
        return linregress_slope_internal(y, n)
    else:
        with nogil:
            return linregress_slope_internal(y, n)


def __test_linregress_slope(y):
    '''
    >>> __test_linregress_slope(np.zeros(10, dtype=np.uint32))
    0.0
    >>> __test_linregress_slope(np.arange(1, 100, 3, dtype=np.uint32))
    3.0
    >>> round(__test_linregress_slope(np.arange(100, dtype=np.uint32) // 2), 3)
    0.5
    >>> round(__test_linregress_slope(
    ...     np.array([54025 - (i + (i % 17)) // 7 for i in range(16384)],
    ...     dtype=np.uint32)), 3)
    -0.143
    '''
    return linregress_slope(y)


cdef Frac double_as_frac(double flt):
    n, d = float(flt).as_integer_ratio()
    return make_frac_fast(n, d)


def test_limit_denominator(double num, uint64_t max_denominator):
    '''
    >>> import math
    >>> test_limit_denominator(math.pi, 10)
    (22, 7)
    >>> test_limit_denominator(math.pi, 100)
    (311, 99)
    >>> test_limit_denominator(math.pi, 1024)
    (355, 113)
    >>> test_limit_denominator(-math.pi, 1024)
    (-355, 113)
    '''
    frac = limit_denominator(double_as_frac(num), max_denominator)
    return (frac.numerator, frac.denominator)


cdef Frac limit_denominator(Frac self, uint64_t max_denominator) nogil:
    if max_denominator < 1 or self.denominator == 0:
        return self
    if self.denominator <= max_denominator:
        return self

    cdef uint64_t p0 = 0
    cdef uint64_t q0 = 1
    cdef uint64_t p1 = 1
    cdef uint64_t q1 = 0
    cdef uint64_t d = self.denominator
    cdef int64_t nn = self.numerator
    cdef c_bool negative = nn < 0
    cdef uint64_t int_part = <uint64_t>cpp_abs(nn) // d
    cdef uint64_t n = <uint64_t>cpp_abs(nn) % d

    cdef uint64_t orig_n = n
    cdef uint64_t orig_d = d

    cdef uint64_t a
    cdef uint64_t q2
    cdef uint64_t tmp

    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > max_denominator:
            break

        tmp = p0
        p0 = p1
        q0 = q1
        p1 = tmp + a * p1
        q1 = q2

        tmp = n
        n = d
        d = tmp - a * d

    cdef uint64_t k = (max_denominator - q0) // q1
    cdef Frac bound1 = make_frac(p0+k*p1, q0+k*q1)
    cdef Frac bound2 = make_frac(p1, q1)
    cdef double self_frac_double = <double>orig_n / <double>orig_d
    cdef double bound1_double = bound1.to_double()
    cdef double bound2_double = bound2.to_double()
    cdef Frac frac_part_res
    if cpp_abs(bound2_double - self_frac_double) <= \
            cpp_abs(bound1_double - self_frac_double):
        frac_part_res = bound2
    else:
        frac_part_res = bound1

    cdef uint64_t new_d = frac_part_res.denominator
    cdef int64_t new_n_abs = new_d * int_part + frac_part_res.numerator
    return make_frac_fast(-new_n_abs if negative else new_n_abs, new_d)


cdef class LinearReduceResult:
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t __linear_reduce(
        const uint32_t[::1] values, int64_t numerator,
        uint64_t denominator, uint32_t[::1] reduced_values) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t n = len(values)
    # We're not using PyMem_Malloc/PyMem_Free because they require GIL
    cdef int64_t tmp_buffer[1024]
    cdef int64_t* tmp
    if n > 1024:
        tmp = <int64_t*>malloc(n * sizeof(int64_t))
    else:
        tmp = tmp_buffer

    cdef int64_t v
    cdef int64_t offset = 0x7fffffffffffffffll

    for i in range(n):
        v = values[i] - i * numerator // <int64_t>denominator
        tmp[i] = v
        if v < offset:
            offset = v

    for i in range(n):
        reduced_values[i] = tmp[i] - offset

    if tmp != tmp_buffer:
        free(tmp)
    return offset


cdef LinearReduceResult linear_reduce(uint32_t[::1] values, Frac slope):
    '''
    tmp =
        values -
        np.arange(0, numerator * n, numerator, dtype=np.int64) // denominator
    res.offset = np.min(tmp)
    res.values = np.subtract(tmp, offset, dtype=np.uint32, casting='unsafe')

    except that division is done in C semantics
    (round toward 0, rather than -infinity)
    '''
    cdef Py_ssize_t n = len(values)
    cdef int64_t numerator = slope.numerator
    cdef uint64_t denominator = slope.denominator

    cdef LinearReduceResult res = \
        LinearReduceResult.__new__(LinearReduceResult)
    res.reduced_values = np.empty(n, dtype=np.uint32)

    cdef uint32_t[::1] out_view = res.reduced_values

    if n < 256:
        res.offset = __linear_reduce(values, numerator, denominator, out_view)
    else:
        with nogil:
            res.offset = __linear_reduce(values, numerator, denominator,
                                         out_view)
    return res


cdef class PrepareStrideResult:
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[uint32_t, uint32_t] __prepare_strides(
        const uint32_t[::1] values, uint32_t stride,
        uint32_t[::1] base_values, uint32_t[::1] delta) nogil:
    cdef uint32_t n = len(values)
    cdef uint32_t base_n = (n + stride - 1) // stride
    cdef uint32_t k = 0
    cdef uint32_t i = 0
    cdef uint32_t j
    cdef uint32_t upper
    cdef uint32_t minv
    cdef uint32_t delta_max = 0
    cdef uint32_t delta_nonzeros = 0
    cdef uint32_t v
    while i < n:
        upper = min(i + stride, n)
        minv = values[i]
        for j in range(i + 1, upper):
            v = values[j]
            if v < minv:
                minv = v
        base_values[k] = minv
        for j in range(i, upper):
            v = values[j] - minv
            delta[j] = v
            if v != 0:
                delta_nonzeros += 1
            if v > delta_max:
                delta_max = v
        k += 1
        i += stride

    return pair[uint32_t, uint32_t](delta_nonzeros, delta_max)


cdef PrepareStrideResult prepare_strides(uint32_t[::1] values,
                                         uint32_t stride):
    cdef uint32_t n = len(values)
    cdef uint32_t base_n = (n + stride - 1) // stride
    cdef PrepareStrideResult res = \
        PrepareStrideResult.__new__(PrepareStrideResult)
    res.base_values = np.empty(base_n, dtype=np.uint32)
    res.delta = np.empty(n, dtype=np.uint32)

    cdef uint32_t[::1] base_values_view = res.base_values
    cdef uint32_t[::1] delta_view = res.delta
    cdef pair[uint32_t, uint32_t] r

    if n < 1024 // 4:
        r = __prepare_strides(
            values, stride, base_values_view, delta_view)
    else:
        with nogil:
            r = __prepare_strides(
                values, stride, base_values_view, delta_view)
    res.delta_nonzeros = r.first
    res.delta_max = r.second
    return res
