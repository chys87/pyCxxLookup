#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

# Copyright (c) 2014, 2016, chys <admin@CHYS.INFO>
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

import functools
import os
import sys

import numpy as np

try:
    from . import _speedups
except ImportError:
    _speedups = None


def make_numpy_array(values):
    return np.array(values, dtype=np.uint32)


def most_common_element(arr, *, mode_cnt=_speedups and _speedups.mode_cnt):
    """Return the most common element of a numpy array"""
    if mode_cnt:
        res = mode_cnt(arr)
        if res is not None:
            return res[0]
    u, indices = np.unique(arr, return_inverse=True)
    return int(u[np.argmax(np.bincount(indices))])


def most_common_element_count(arr, *,
                              mode_cnt=_speedups and _speedups.mode_cnt):
    """Return the most common element of a numpy array and its count
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.uint32))
    (5, 3)
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.int64))
    (5, 3)
    """
    if mode_cnt:
        res = mode_cnt(arr)
        if res is not None:
            return res
    u, indices = np.unique(arr, return_inverse=True)
    bincnt = np.bincount(indices)
    i = np.argmax(bincnt)
    return int(u[i]), int(bincnt[i])


def is_const(array, *, is_const=_speedups and _speedups.is_const):
    """Returns if the given array is constant"""
    if is_const:
        res = is_const(array)
        if res is not None:
            return res
    if array.size == 0:
        return True
    else:
        return (array == array[0]).all()


def is_linear(array, *, is_linear=_speedups and _speedups.is_linear):
    """Returns if the given array is linear"""
    if is_linear:
        res = is_linear(array)
        if res is not None:
            return res
    return is_const(slope_array(array, array.dtype.type))


def const_range(array):
    """Returns the max n, such that array[:n] is a constant array"""
    if array.size == 0:
        return 0
    k = (array != array[0]).tostring().find(1)
    if k < 0:
        k = array.size
    return k


def range_limit(array, threshold):
    '''
    Return the max k, such that max(array[:k]) - min(array[:k]) < threshold
    (array can be an iterator)
    '''
    ran = np.maximum.accumulate(array) - np.minimum.accumulate(array)
    return int(ran.searchsorted(threshold))


def is_monotonically_increasing(array):
    return array.size >= 2 and (array[1:] >= array[:-1]).all()


def trim_brackets(s):
    front = 0
    while front < len(s) and s[front] == '(':
        front += 1
    if not front:
        return s

    back = 0
    while back < len(s) and s[-1 - back] == ')':
        back += 1
    if not back:
        return s

    trim_cnt = min(front, back)
    s = s[trim_cnt:-trim_cnt]

    n = 0
    m = 0
    for c in s:
        if c == '(':
            n += 1
        elif c == ')':
            n -= 1
            m = min(m, n)
    if m < 0:
        s = '('*-m + s + ')'*-m
    return s


def compress_array(array, n):
    '''
    Compress several elements of one array into one.
    '''
    bits = 8 // n
    lo = array.size % n or n

    res = array[::n].copy()
    for i in range(1, lo):
        res |= array[i::n] << (i * bits)
    for i in range(lo, n):
        res[:-1] |= array[i::n] << (i * bits)

    return res


def slope_array(array, dtype=np.int64, *,
                slope_array=_speedups and _speedups.slope_array):
    '''
    slope_array is similar to np.diff, but with speedups for certain types.
    Additionally, we support output types different than the input type.

    >>> slope_array(np.array([1,2,4,0,2], np.uint32), np.int64).tolist()
    [1, 2, -4, 2]
    '''
    if slope_array:
        res = slope_array(array, dtype)
        if res is not None:
            return res
    return np.array(array[1:], dtype) - np.array(array[:-1], dtype)


def gcd_many(array, *, gcd_many=_speedups and _speedups.gcd_many):
    """
    >>> gcd_many(np.array([26, 39, 52], np.uint32))
    13
    >>> gcd_many(np.array([4, 8, 7], np.uint32))
    1
    """
    if gcd_many:
        res = gcd_many(array)
        if res is not None:
            return res
    res = 0
    for v in array:
        v = int(v)
        if not v:
            continue
        if res < 2:
            if not res:
                res = v
                continue
            else:
                break
        # We could have used fractions.gcd, but here we do it ourselves for
        # better performance (fractions.gcd has no C implementation)
        while v:
            res, v = v, res % v
    return res


def gcd_reduce(array):
    '''
    Return the max gcd, such that is_const(v % gcd for v in array)
    '''
    array = np_unique(array)
    return gcd_many(slope_array(array, array.dtype.type))


def np_unique(array, *, unique=_speedups and _speedups.unique):
    '''np.unique with speedups for certain types

    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.uint32)).tolist()
    [1, 2, 3, 4, 5, 7]
    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.int64)).tolist()
    [1, 2, 3, 4, 5, 7]
    '''
    if unique:
        res = unique(array)
        if res is not None:
            return res
    return np.unique(array)


def np_min(array, *, min_max=_speedups and _speedups.min_max):
    '''
    >>> np_min(np.array([3,2,1,2,3], np.uint32))
    1
    '''
    if min_max:
        res = min_max(array, 0)
        if res is not None:
            return res
    return int(array.min())


def np_max(array, *, min_max=_speedups and _speedups.min_max):
    '''
    >>> np_max(np.array([3,2,1,2,3], np.uint32))
    3
    '''
    if min_max:
        res = min_max(array, 1)
        if res is not None:
            return res
    return int(array.max())


def np_range(array, *, min_max=_speedups and _speedups.min_max):
    '''
    >>> np_range(np.array([3,2,1,2,3], np.uint32))
    2
    '''
    if min_max:
        res = min_max(array, 2)
        if res is not None:
            return res
    return int(array.max()) - int(array.min())


def np_array_equal(x, y, *, array_equal=_speedups and _speedups.array_equal):
    '''
    >>> np_array_equal(np.uint32([1, 2, 3]), np.uint32([1, 2, 3]))
    True
    >>> np_array_equal(np.uint32([1, 2, 3]), np.uint32([1, 2, 1]))
    False
    '''
    if array_equal:
        res = array_equal(x, y)
        if res is not None:
            return res
    return (x == y).all()


def profiling(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        # We hope it can be set or unset at run-time, so put the check here
        profiling_setting = os.environ.get('pyCxxLookup_Profiling')
        if not profiling_setting:
            return func(*args, **kwargs)

        try:
            import cProfile as profile
        except ImportError:
            import profile
        import pstats

        pr = profile.Profile()
        pr.enable()
        try:
            return func(*args, **kwargs)

        finally:
            pr.disable()

            try:
                stat_count = int(profiling_setting)
            except ValueError:
                stat_count = 0
            if stat_count < 10:
                stat_count = 30

            ps = pstats.Stats(pr, stream=sys.stderr)
            ps.sort_stats('cumulative', 'stdname')
            ps.print_stats(stat_count)

            ps.sort_stats('tottime', 'stdname')
            ps.print_stats(stat_count)

            if os.environ.get('pyCxxLookup_Profiling_Callers'):
                ps.print_callers()

    return _func


class cached_property:
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self

        func = self.func
        res = obj.__dict__[func.__name__] = func(obj)
        return res
