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

cimport cython
from cpython.ref cimport PyObject
from libc.stdint cimport uintptr_t, uint32_t
from libcpp.vector cimport vector

from cxxlookup.pyx_helpers cimport flat_hash_set


def is_pow2(x) -> bool:
    '''Test whether v is a power of 2
    >>> is_pow2(0)
    False
    >>> is_pow2(1)
    True
    >>> is_pow2(2)
    True
    >>> is_pow2(3)
    False
    '''
    cdef long long v = x
    return v > 0 and not (v & (v - 1))


def walk(node):
    '''Iterate over node and its children without deduplication.
    '''
    cdef PyObject* nodep
    cdef vector[PyObject*] q

    yield node
    nodep = <PyObject*>node;
    q.push_back(nodep)

    while not q.empty():
        nodep = q.back()
        q.pop_back()
        node = <object>nodep
        children = node.children
        yield from children
        for child in children:
            q.push_back(<PyObject*>(child))


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
def linregress_slope(uint32_t[::1] y) -> float:
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
    #          sigma (y_i - y_bar)(x_i - x_bar)
    # slope = ---------------------------------
    #           sigma (x_i - x_bar)^2
    #
    #          sigma  y_i (x_i - x_bar)
    #       = ---------------------------
    #          sigma x_i^2 - n * x_bar^2
    #
    # because x_i = i, so denominator = n(n^2-1)/12

    cdef int n = len(y)
    cdef float x_bar

    cdef float a
    cdef int i
    cdef float res = 0

    if n < 128:
        if n >= 2:
            x_bar = (n - 1) * <float>.5
            a = 0
            for i in range(n):
                a += y[i] * (i - x_bar)
            res = 12 * a / (n * (n * n - 1))

    else:
        with nogil:
            x_bar = (n - 1) * <float>.5
            a = 0
            for i in range(n):
                a += y[i] * (i - x_bar)
            res = 12 * a / (n * (n * n - 1))

    return res
