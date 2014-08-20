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


import numpy as np

from . import utils


def __yield_linear_parts(array, linear_threshold, const_threshold):
    '''
    >>> import numpy as np
    >>> a = np.array([1,1,2,3,4,5,5,4,3,2,2,2,2])
    >>> list(__yield_linear_parts(a, 5, 4))
    [(1, 6), (9, 13)]
    >>> a = np.array([1,1,1,1,2,3,4,5,6,7,8,9,10])
    >>> list(__yield_linear_parts(a, 5, 4))
    [(0, 4), (4, 13)]
    '''
    L = array.size
    if L < 3:
        return

    delta = np.array(array[1:], np.int64) - np.array(array[:-1], np.int64)
    unequal = (delta[1:] != delta[:-1]).nonzero()[0].tolist()

    unequal.append(L - 2)

    lo = 0
    for i in unequal:
        hi = i + 2
        length = hi - lo
        if length >= linear_threshold or (
                length >= const_threshold and delta[lo] == 0):
            yield lo, hi
            lo = hi
        else:
            lo = hi - 1


def _yield_linear_parts(array, opt):
    const_threshold = opt.const_threshold
    linear_threshold = opt.linear_threshold
    return __yield_linear_parts(array, linear_threshold, const_threshold)


def _naive_groupify(base, values, opt):
    group = {}
    lo = 0

    for ls, le in _yield_linear_parts(values, opt):
        if ls > lo:
            group[base + lo] = values[lo:ls]
        group[base + ls] = values[ls:le]
        lo = le
    if lo < values.size:
        group[base + lo] = values[lo:]
    return group


def _refine_groups(group, hole, opt):
    # Split a group if it will result in two subgroups
    # with range < 256
    split_threshold = opt.split_threshold
    for threshold in (65536, 256):

        for lo, values in list(group.items()):

            if values.size <= 2 * split_threshold or utils.is_linear(values):
                continue

            l = utils.range_limit(values, threshold)
            if not split_threshold <= l <= values.size - split_threshold:
                continue

            right = values[l:]
            if right.max() - right.min() >= threshold:
                continue

            group[lo + l] = right
            group[lo] = values[:l]

    # Split a group if it takes the form: [a,...,a,b,...,b]
    # Do it unconditionally!
    for lo, values in list(group.items()):
        k = utils.const_range(values)
        if k < values.size and utils.is_const(values[k:]):
            group[lo + k] = values[k:]
            group[lo] = values[:k]


def _remove_holes(group, hole, opt):
    hole_threshold = opt.hole_threshold
    group_threshold = opt.group_threshold

    # Discard hole groups
    del_hole_list = []
    for lo, values in group.items():
        if (values == hole).all():
            del_hole_list.append(lo)
    for lo in del_hole_list:
        del group[lo]

    # Trim hole values from the beginning and end of groups.
    for lo, values in list(group.items()):
        if values[-1] == hole:
            k = utils.const_range(values[::-1])
            if (k >= hole_threshold) or (lo == max(group)):
                group[lo] = values = values[:-k]
        if values[0] == hole:
            k = utils.const_range(values)
            if (k >= hole_threshold) or (lo == min(group)):
                group[lo+k] = values[k:]
                del group[lo]

    # Split very small groups if they're nonlinear
    for lo, values in list(group.items()):
        if len(values) < group_threshold and not utils.is_linear(values):
            for v in values:
                group[lo] = [v]
                lo += 1


def groupify(base, values, hole, opt):
    """
    Result: a map: lo => (value,value,.....)
    """
    group = _naive_groupify(base, values, opt)
    _refine_groups(group, hole, opt)
    _remove_holes(group, hole, opt)
    return group
