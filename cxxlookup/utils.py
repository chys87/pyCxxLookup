#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

# Copyright (c) 2014, chys <admin@CHYS.INFO>
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

import os
import sys

import numpy as np

try:
    from . import _speedups
except ImportError:
    _speedups = None


def make_numpy_array(values):
    return np.array(values, dtype=np.uint32)


def most_common_element(arr):
    """Return the most common element of a numpy array"""
    if _speedups:
        res = _speedups.mode_cnt(arr)
        if res is not None:
            return res[0]
    u, indices = np.unique(arr, return_inverse=True)
    return int(u[np.argmax(np.bincount(indices))])


def most_common_element_count(arr):
    """Return the most common element of a numpy array and its count
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.uint32))
    (5, 3)
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.int64))
    (5, 3)
    """
    if _speedups:
        res = _speedups.mode_cnt(arr)
        if res is not None:
            return res
    u, indices = np.unique(arr, return_inverse=True)
    bincnt = np.bincount(indices)
    i = np.argmax(bincnt)
    return int(u[i]), int(bincnt[i])


def is_const(array):
    """Returns if the given array is constant"""
    if array.size == 0:
        return True
    else:
        return (array == array[0]).all()


def is_linear(array):
    """Returns if the given array is linear"""
    if _speedups:
        res = _speedups.is_linear(array)
        if res is not None:
            return res
    return is_const(array[1:] - array[:-1])


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


def stridize(array, n, default=0):
    '''
    => (array[0], array[n], array[2*n], ...), (array[1], arra[n+1], ...), ...
    '''
    l = array.size
    if l % n == 0:
        return np.reshape(array, [l // n, n]).T
    else:
        padsize = n - (l % n)
        tmp = [array, np.zeros(padsize, dtype=array.dtype) + default]
        tmp = np.concatenate(tmp)
        return tmp.reshape([l // n + 1, n]).T


def gcd_many(array, _speedups=_speedups):
    """
    >>> gcd_many(np.array([26, 39, 52], np.uint32))
    13
    >>> gcd_many(np.array([4, 8, 7], np.uint32))
    1
    """
    if _speedups:
        res = _speedups.gcd_many(array)
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
    return gcd_many(array[1:] - array[:-1])


def np_unique(array):
    '''np.unique with speedups for certain types

    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.uint32)).tolist()
    [1, 2, 3, 4, 5, 7]
    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.int64)).tolist()
    [1, 2, 3, 4, 5, 7]
    '''
    if _speedups:
        res = _speedups.unique(array)
        if res is not None:
            return res
    return np.unique(array)


def np_min(array):
    '''
    >>> np_min(np.array([3,2,1,2,3], np.uint32))
    1
    '''
    if _speedups:
        res = _speedups.min_max(array, 0)
        if res is not None:
            return res
    return int(array.min())


def np_max(array):
    '''
    >>> np_max(np.array([3,2,1,2,3], np.uint32))
    3
    '''
    if _speedups:
        res = _speedups.min_max(array, 1)
        if res is not None:
            return res
    return int(array.max())


def np_range(array):
    '''
    >>> np_range(np.array([3,2,1,2,3], np.uint32))
    2
    '''
    if _speedups:
        res = _speedups.min_max(array, 2)
        if res is not None:
            return res
    return int(array.max()) - int(array.min())


def profiling(func):

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
            res = func(*args, **kwargs)

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

        return res

    return _func
