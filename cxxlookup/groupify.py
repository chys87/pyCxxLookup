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


from . import utils


def _yield_linear_parts(array, opt):
    L = array.size
    const_threshold = opt.const_threshold
    linear_threshold = opt.linear_threshold
    limit = L - min(const_threshold, linear_threshold)
    i = 0
    while i <= limit:
        delta = int(array[i+1]) - int(array[i])
        j = i+2
        while j < L and int(array[j]) - int(array[j-1]) == delta:
            j += 1
        if j-i >= linear_threshold or (delta == 0 and j-i >= const_threshold):
            yield i, j
            i = j
        else:
            i = j - 1


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
    # Split a group if it will result in at least one subgroups
    # with range < 256
    for lo, values in list(group.items()):

        if values.size <= opt.split_threshold or utils.is_linear(values):
            continue

        # Split from backward.
        while True:
            h = utils.range_limit(values[::-1], 256)
            if (h == values.size) or (h < opt.split_threshold):
                break
            h = values.size - h
            group[lo + h] = values[h:]
            group[lo] = values = values[:h]

        # Split from front.
        while True:
            l = utils.range_limit(values, 256)
            if (l == values.size) or (l < opt.split_threshold):
                break
            group[lo+l] = values[l:]
            group[lo] = values[:l]
            lo += l
            values = group[lo]

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
