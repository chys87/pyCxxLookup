#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

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

import functools
import os
import sys
import threading
import time

import numpy as np

from . import cutils
from . import _speedups


is_pow2 = cutils.is_pow2
cached_property = cutils.cached_property


def make_numpy_array(values):
    return np.array(values, dtype=np.uint32)


def most_common_element(arr, /):
    """Return the most common element of a numpy array"""
    res = _speedups.mode_cnt(arr)
    if res is not None:
        return res[0]
    u, indices = np.unique(arr, return_inverse=True)
    return int(u[np.argmax(np.bincount(indices))])


def most_common_element_count(arr, /):
    """Return the most common element of a numpy array and its count
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.uint32))
    (5, 3)
    >>> most_common_element_count(np.array([1,3,5,0,5,1,5], np.int64))
    (5, 3)
    """
    res = _speedups.mode_cnt(arr)
    if res is not None:
        return res
    u, indices = np.unique(arr, return_inverse=True)
    bincnt = np.bincount(indices)
    i = np.argmax(bincnt)
    return int(u[i]), int(bincnt[i])


def is_const(array, /):
    """Returns if the given array is constant"""
    res = _speedups.is_const(array)
    if res is not None:
        return res
    if array.size == 0:
        return True
    else:
        return (array == array[0]).all()


def is_linear(array, /):
    """Returns if the given array is linear"""
    res = _speedups.is_linear(array)
    if res is not None:
        return res
    return is_const(slope_array(array, array.dtype.type))


def const_range(array, /):
    """Returns the max n, such that array[:n] is a constant array"""
    res = _speedups.const_range(array)
    if res is not None:
        return res

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


def slope_array(array, /, dtype=np.int64):
    '''
    slope_array is similar to np.diff, but with speedups for certain types.
    Additionally, we support output types different than the input type.

    >>> slope_array(np.array([1,2,4,0,2], np.uint32), np.int64).tolist()
    [1, 2, -4, 2]
    '''
    res = _speedups.slope_array(array, dtype)
    if res is not None:
        return res
    return np.array(array[1:], dtype) - np.array(array[:-1], dtype)


def gcd_many(array, /):
    """
    >>> gcd_many(np.array([26, 39, 52], np.uint32))
    13
    >>> gcd_many(np.array([4, 8, 7], np.uint32))
    1
    """
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
    return gcd_many(slope_array(array, array.dtype.type))


def np_unique(array, /):
    '''np.unique with speedups for certain types

    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.uint32)).tolist()
    [1, 2, 3, 4, 5, 7]
    >>> np_unique(np.array([1,3,5,7,1,2,3,4], np.int64)).tolist()
    [1, 2, 3, 4, 5, 7]
    '''
    res = _speedups.unique(array)
    if res is not None:
        return res
    return np.unique(array)


def np_min(array, /):
    '''
    >>> np_min(np.array([3,2,1,2,3], np.uint32))
    1
    '''
    res = _speedups.min_max(array, 0)
    if res is not None:
        return res
    return int(array.min())


def np_max(array, /):
    '''
    >>> np_max(np.array([3,2,1,2,3], np.uint32))
    3
    '''
    res = _speedups.min_max(array, 1)
    if res is not None:
        return res
    return int(array.max())


def np_min_by_chunk(array, chunk_size, /):
    '''Return the minimum values of each fixed-size chunk

    _speedups only implements uint32

    >>> np_min_by_chunk(np.array([1, 2, 2, 1, 3], np.uint32), 2).tolist()
    [1, 1, 3]
    >>> np_min_by_chunk(np.array([1, 2, 2, 1, 3], np.int64), 2).tolist()
    [1, 1, 3]
    '''
    res = _speedups.min_by_chunk(array, chunk_size)
    if res is not None:
        return res

    n, = array.shape
    chunks = (n + chunk_size - 1) // chunk_size
    padded_size = chunks * chunk_size
    if n == padded_size:
        padded = array
    else:
        padded = np.pad(array, (0, padded_size - n), 'edge')
    return np.min(np.reshape(padded, (chunks, chunk_size)), axis=1)


def np_range(array, /):
    '''
    >>> np_range(np.array([3,2,1,2,3], np.uint32))
    2
    '''
    res = _speedups.min_max(array, 2)
    if res is not None:
        return res
    return int(array.max()) - int(array.min())


def np_array_equal(x, y, /):
    '''
    >>> np_array_equal(np.uint32([1, 2, 3]), np.uint32([1, 2, 3]))
    True
    >>> np_array_equal(np.uint32([1, 2, 3]), np.uint32([1, 2, 1]))
    False
    '''
    res = _speedups.array_equal(x, y)
    if res is not None:
        return res
    return (x == y).all()


def np_cycle(array, /, *, max_cycle=None,
             np_array_equal=np_array_equal, range=range):
    '''Find minimun positive cycle of array.

    _speedups only implements uint32

    >>> np_cycle(np.array([25] * 54, dtype=np.uint32))
    1
    >>> np_cycle(np.array([25] * 54, dtype=np.int16))
    1
    >>> np_cycle(np.array(list(range(100)) * 9, dtype=np.uint32))
    100
    >>> np_cycle(np.array(list(range(100)) * 9, dtype=np.uint64))
    100
    >>> np_cycle(np.array(list(range(100)) * 9 + list(range(50)),
    ...                   dtype=np.uint32))
    100
    >>> np_cycle(np.array(list(range(100)) * 9 + list(range(50)),
    ...                   dtype=np.uint64))
    100
    >>> np_cycle(np.arange(100, dtype=np.uint32))
    0
    >>> np_cycle(np.arange(100, dtype=np.uint64))
    0
    '''
    res = _speedups.array_cycle(array, max_cycle or 0xffffffff)
    if res is not None:
        return res

    n, = array.shape
    if n < 2:
        return 0

    indices, = np.nonzero(array == array[0])
    ind_n, = indices.shape
    if ind_n == n:  # array is const
        return 1

    indices = indices.astype(np.uint32)

    max_cycle = max_cycle or n

    for i in range(1, ind_n):
        k = int(indices[i])
        if k > max_cycle:
            break

        # Check whether indices are likely correct
        ok = True
        for j in range(i * 2, ind_n, i):
            if indices[j] != indices[j - i] + k:
                ok = False
                break
        if not ok:
            continue

        # Compare array slices
        tail = n % k
        if tail and not np_array_equal(array[:tail], array[-tail:]):
            continue

        ref = array[:k]
        for j in range(k, n - k + 1, k):
            if not np_array_equal(ref, array[j:j+k]):
                break
        else:
            return k

    return 0


__thread_profiles = []
__profiler_active = False


def __new_profile():
    from cProfile import Profile
    return Profile(time.thread_time_ns, 1e-9)


def profiling(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        global __profiler_active

        if not os.environ.get('pyCxxLookup_Profiling'):
            return func(*args, **kwargs)

        __thread_profiles.clear()

        import pstats
        import time

        wall_clock = time.time()

        __profiler_active = True
        pr = __new_profile()
        pr.enable()
        try:
            return func(*args, **kwargs)

        finally:
            pr.disable()
            __profiler_active = False
            wall_time = time.time() - wall_clock
            print(f'Wall time: {wall_time:.3f} seconds')

            ps = pstats.Stats(pr, *__thread_profiles, stream=sys.stderr)
            __thread_profiles.clear()

            stat_count = 30
            ps.sort_stats('cumulative', 'stdname')
            ps.print_stats(stat_count)

            ps.sort_stats('tottime', 'stdname')
            ps.print_stats(stat_count)

            if os.environ.get('pyCxxLookup_Profiling_Callers'):
                ps.print_callers()

    return _func


def thread_profiling(func):
    '''This decorator should be applied to functions run in separate threads,
    so that the results are collected to the main thread
    '''
    @functools.wraps(func)
    def _func(*args, **kwargs):
        if not __profiler_active:
            return func(*args, **kwargs)

        try:
            pr = __thread_profiles.pop()
        except IndexError:
            pr = __new_profile()
        pr.enable()
        try:
            return func(*args, **kwargs)

        finally:
            pr.disable()
            __thread_profiles.append(pr)

    return _func


class __ThreadPoolTask:
    def __init__(self, f, arglist):
        self._f = f
        self._pending = list(enumerate(arglist))
        # Because we use pop to get tasks from _pending, we need a reversed
        # list so that tasks are executed in their original order.
        self._pending.reverse()
        self.results = [None] * len(arglist)
        self._semaphore = threading.Semaphore(0)

    @thread_profiling
    def run_in_thread(self):
        while True:
            try:
                i, arg = self._pending.pop()
            except IndexError:
                return
            try:
                self.results[i] = self._f(arg)
            finally:
                self._semaphore.release()

    def _ping_async_list(self, async_list):
        for i in range(len(async_list) - 1, -1, -1):
            async_obj = async_list[i]
            if async_obj.ready():
                async_obj.get()
                del async_list[i]

    def run_and_wait_all(self, thread_pool):
        n = len(self.results)
        async_list = [thread_pool.apply_async(self.run_in_thread)
                      for _ in range(min(os.cpu_count(), n) - 1)]

        # We should also run from main thread, to avoid deadlocking.
        done = 0
        while True:
            try:
                i, arg = self._pending.pop()
            except IndexError:
                break
            self.results[i] = self._f(arg)
            done += 1

            if self._semaphore.acquire(blocking=False):
                done += 1
                while self._semaphore.acquire(blocking=False):
                    done += 1
                # Whenever a task is finished in another thread, run
                # _ping_async_list immediately so that exceptions are
                # propagated to main thread as soon as possible.
                self._ping_async_list(async_list)

        while done < n:
            self._semaphore.acquire()
            done += 1
            self._ping_async_list(async_list)

        return self.results


def thread_pool_map(thread_pool, f, arglist):
    '''This function is like ThreadPool.map, but it can be safely called
    from a thread which is itself in the pool, without the possibility
    of deadlocking.
    This function also takes care of profiling already.
    '''
    n = len(arglist)
    if n == 0:
        return []
    if n == 1:
        return [f(arglist[0])]
    return __ThreadPoolTask(f, arglist).run_and_wait_all(thread_pool)
