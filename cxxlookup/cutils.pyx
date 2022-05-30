# distutils: language=c++
# cython: language_level=3
# cython: profile=True

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

cimport cython
from cpython.ref cimport PyObject
from libc.stdint cimport int64_t, uintptr_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector

from .pyx_helpers cimport abs as cpp_abs, flat_hash_set


def walk_dedup(node):
    '''Iterate over node and its children with deduplication.
    >>> class A:
    ...    children = []
    ...    def __init__(self, name): self.name = name
    ...    def __repr__(self): return self.name
    >>> a = A('a')
    >>> b = A('b')
    >>> a.children = [a, b]
    >>> b.children = [b]
    >>> list(walk_dedup(a))
    [a, b]
    >>> list(walk_dedup(b))
    [b]
    '''
    cdef PyObject* nodep
    cdef flat_hash_set[PyObject*] visited
    cdef vector[PyObject*] q

    nodep = <PyObject*>node
    q.push_back(nodep)
    visited.insert(nodep)

    while not q.empty():
        nodep = q.back()
        q.pop_back()
        node = <object>nodep
        yield node
        for child in node.children:
            nodep = <PyObject*>(child)
            if not visited.contains(nodep):
                visited.insert(nodep)
                q.push_back(nodep)


cdef vector[PyObject*] walk_dedup_fast(node):
    '''A faster version of walk_dedup, returning vector[PyObject*]
    '''
    cdef PyObject* nodep
    cdef flat_hash_set[PyObject*] visited
    cdef vector[PyObject*] res

    nodep = <PyObject*>node
    res.push_back(nodep)
    visited.insert(nodep)

    cdef size_t i = 0

    while i < res.size():
        nodep = res[i]
        i += 1
        for child in (<object>nodep).children:
            nodep = <PyObject*>(child)
            if not visited.contains(nodep):
                visited.insert(nodep)
                res.push_back(nodep)

    return std_move(res)


class cached_property:
    '''
    >>> class Test:
    ...     @cached_property
    ...     def f(self):
    ...         print('x')
    ...         return 2
    >>> a = Test()
    >>> a.f
    x
    2
    >>> a.f
    2
    '''
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__

    @cython.profile(False)
    def __get__(self, obj, cls):
        if obj is None:
            return self

        func = self.func
        res = obj.__dict__[func.__name__] = func(obj)
        return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
    cdef float x_bar = (n - 1) * <float>.5
    cdef float a = 0
    for i in range(n):
        a += y[i] * (i - x_bar)
    return 12 * a / (n * (n * n - 1))


@cython.nonecheck(False)
cdef float linregress_slope(uint32_t[::1] y):
    '''Compute linear regression slope

    Roughly equivalent to
    scipy.stats.linregress(np.arange(len(y), np.float32),
                           y.astype(np.float32)).slope,
    but much faster.

    >>> import numpy as np
    >>> linregress_slope(np.zeros(10, dtype=np.uint32))
    0.0
    >>> linregress_slope(np.arange(1, 100, 3, dtype=np.uint32))
    3.0
    >>> round(linregress_slope(np.arange(100, dtype=np.uint32) // 2), 3)
    0.5
    '''
    cdef int n = len(y)

    if n < 128:
        if n < 2:
            return 0
        return linregress_slope_internal(y, n)
    else:
        with nogil:
            return linregress_slope_internal(y, n)


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


@cython.cdivision(True)
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
