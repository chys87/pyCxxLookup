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


import numpy as np

from . import utils


def _yield_linear_parts(array, opt):
    '''
    >>> import numpy as np
    >>> a = np.array([1,1,2,3,4,5,5,4,3,2,2,2,2])
    >>> list(_yield_linear_parts(a, (5, 4)))
    [(1, 6, 1), (9, 13, 0)]
    >>> a = np.array([1,1,1,1,2,3,4,5,6,7,8,9,10])
    >>> list(_yield_linear_parts(a, (5, 4)))
    [(0, 4, 0), (4, 13, 1)]
    '''
    L = array.size
    if L < 3:
        return

    try:
        linear_threshold = opt.linear_threshold
        const_threshold = opt.const_threshold
    except AttributeError:
        linear_threshold, const_threshold = opt

    delta = utils.slope_array(array, np.int64)
    unequal = (delta[1:] != delta[:-1]).nonzero()[0].tolist()

    unequal.append(L - 2)

    lo = 0
    for i in unequal:
        hi = i + 2
        length = hi - lo
        if length >= linear_threshold or (
                length >= const_threshold and delta[lo] == 0):
            yield lo, hi, int(delta[lo])
            lo = hi
        else:
            lo = hi - 1


def _get_slope(values):
    if values.size < 2:
        return None
    elif utils.is_linear(values):
        return int(values[1]) - int(values[0])
    else:
        return None


def _naive_groupify(base, values, opt):
    group_list = []
    lo = 0

    for ls, le, slope in _yield_linear_parts(values, opt):
        if ls > lo:
            group_list.append((lo, ls, _get_slope(values[lo:ls])))
        group_list.append((ls, le, slope))
        lo = le
    if lo < values.size:
        group_list.append((lo, values.size, _get_slope(values[lo:])))

    # Merge three or more consecutive linear groups with the same slope,
    # in order to reduce the number of switch branches
    group = {}

    temp_list = []
    temp_slope = None
    def emit_temp_list():
        nonlocal temp_slope
        if temp_slope is None:
            return
        if len(temp_list) >= 3:
            lo = temp_list[0][0]
            hi = temp_list[-1][-1]
            group[base + lo] = values[lo:hi]
        else:
            for lo, hi in temp_list:
                group[base + lo] = values[lo:hi]

        del temp_list[:]
        temp_slope = None

    for lo, hi, slope in group_list:
        if slope is None or temp_slope != slope or lo != temp_list[-1][-1]:
            emit_temp_list()
        if not slope:  # None or 0
            group[base + lo] = values[lo:hi]
        else:
            temp_slope = slope
            temp_list.append((lo, hi))

    emit_temp_list()

    return group


def _refine_groups(group, hole, opt):
    # Split a group if it takes the form: [a,...,a,b,...,b]
    # Do it almost unconditionally, but not if it's [a,a+1]
    for lo, values in list(group.items()):
        if values.size == 2 and values[1] == values[0] + 1:
            continue
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
        if values[0] == hole and utils.is_const(values):
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
            if (k >= hole_threshold) or (0 != lo == min(group)):
                group[lo+k] = values[k:]
                del group[lo]

    # Split very small groups if they're nonlinear
    for lo, values in list(group.items()):
        if len(values) < group_threshold and not utils.is_linear(values):
            for v in values:
                if v != hole:
                    group[lo] = utils.make_numpy_array([v])
                lo += 1


def groupify(base, values, hole, opt):
    """
    Result: a map: lo => (value,value,.....)
    """
    group = _naive_groupify(base, values, opt)
    _refine_groups(group, hole, opt)
    _remove_holes(group, hole, opt)
    return group
