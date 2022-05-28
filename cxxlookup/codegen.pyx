# distutils: language=c++
# cython: language_level=3
# cython: profile=True

#!/usr/bin/env python3
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

from collections import defaultdict
from fractions import Fraction
import math
from multiprocessing.pool import ThreadPool
import string

import numpy as np

from .expr import *
from . import expr
from . import cutils
from . import groupify
from . import utils

cimport cython
from cpython.ref cimport PyObject
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.utility cimport move as std_move, pair
from libcpp.vector cimport vector

from .cutils cimport linregress_slope, walk_dedup_fast
from .pyx_helpers cimport bit_length, flat_hash_set


COMMON_HEADERS = r'''#include <stdint.h>
'''


def make_code(func_name, base, values, hole, opt):
    groups = groupify.groupify(base, values, hole, opt)

    codes = {}  # lo: (hi, code)
    statics = {}  # lo: statics
    # Do it in parallel.  Many NumPy/SciPy/_speedups methods release GIL
    with ThreadPool() as pool:
        def gen_group(pair):
            lo, values = pair
            range_name = f'{func_name}_{lo:x}_{lo+values.size:x}'
            expr, subexprs = MakeCodeForRange(
                lo, values, range_name, opt, pool).expr_tuple
            code, static = _format_code(expr, subexprs)
            codes[lo] = lo + values.size, code
            static = filter(None, static)
            if static:
                statics[lo] = static

        # Submit big groups before small ones, so that we likely get
        # better parallelization
        utils.thread_pool_map(
            pool, gen_group,
            sorted(groups.items(), key=lambda x: x[1].size, reverse=True))

    res = []
    res.append('namespace {\n\n')

    for lo, static in sorted(statics.items()):
        res.extend(static)

    hole_code = _format_code(Const(32, hole), None)[0]

    # Create a reverse map from code to range (For sharing code between case's)
    rcode = {}
    for lo, (hi, code) in sorted(codes.items()):
        if code != hole_code:
            rcode.setdefault(code, []).append((lo, hi))

    res.append('}}  // namespace\n'
               '\n'
               'uint32_t {}(uint32_t c) noexcept {{\n'
               '  [[maybe_unused]] uint64_t cl = c;\n'
               '  switch (c) {{\n'.format(func_name))
    for lo, (hi, code) in sorted(codes.items()):
        if rcode[code][0][0] != lo:  # Printed at other case's
            continue
        for i, (Lo, Hi) in enumerate(rcode[code]):
            if i > 0:
                res.append('\n')
            if Hi - Lo == 1:
                res.append('    case {:#x}:'.format(Lo))
            else:
                res.append('    case {:#x} ... {:#x}:'.format(Lo, Hi-1))
        res.append(code)
    res.append('    default:')
    res.append(hole_code)
    res.append('  }\n')
    res.append('}\n')

    return ''.join(res)


def _format_code(expr, subexprs):

    added_var = 0
    var_output_order = []

    def add_var_in_expr(expr):
        nonlocal added_var

        # Depth-first search
        for x in expr.walk_tempvar():
            var_id = x.var
            var_id_mask = 1 << var_id
            if not (added_var & var_id_mask):
                var_expr = subexprs[var_id]
                add_var_in_expr(var_expr)
                var_output_order.append(var_id)
                added_var |= var_id_mask

    add_var_in_expr(expr)

    visited_set = set()

    cdef flat_hash_set[PyObject*] renamed_set

    if not added_var:
        # No temporary variable
        main_str = '\n      return {};\n'.format(
            utils.trim_brackets(str(expr)))
        main_statics = expr.statics(visited_set)
        return main_str, (main_statics,)

    else:
        # Rename temporary variables
        rename_map = {var: k for (k, var) in enumerate(var_output_order)}
        # renamed_set = set()

        def rename_var(expr):
            for x in expr.walk_tempvar():
                if renamed_set.find(<PyObject*>x) == renamed_set.end():
                    renamed_set.insert(<PyObject*>x)
                    x.var = rename_map[x.var]

        rename_var(expr)

        code_str = [' {\n']
        statics = []

        for (k, var) in enumerate(var_output_order):
            var_expr = subexprs[var]
            rename_var(var_expr)
            var_type = var_expr.rtype
            # The outermost explicit cast can absolutely be trimmed
            # beucase the variable has an explicit type.
            if var_expr.IS_CAST:
                var_expr = var_expr.value
            # Inner upcasts can also be trimmed
            while var_expr.IS_CAST and var_expr.value.rtype <= var_expr.rtype:
                var_expr = var_expr.value

            code_str.append('      {} {} = {};\n'.format(
                type_name(var_type), ExprTempVar.get_name(k),
                utils.trim_brackets(str(var_expr))))
            statics.append(var_expr.statics(visited_set))

        main_str = '      return {};\n'.format(utils.trim_brackets(str(expr)))
        code_str.append(main_str)
        code_str.append('    }\n')

        statics.append(expr.statics(visited_set))
        statics.sort()

        return ''.join(code_str), statics


cdef int64_t _overhead(expr: Expr, opt):
    """Estimate the overhead of an expression.
    We use the total number of bytes in tables plus additional overheads
    for each Expr instance.
    """
    cdef unsigned int total_bytes = 0
    cdef unsigned int extra = 0
    cdef PyObject* nodep

    cdef vector[PyObject*] walk_list = walk_dedup_fast(expr)
    for nodep in walk_list:
        x = <object>nodep
        extra += <unsigned int>x.overhead
        total_bytes += <unsigned int>x.static_bytes

    return total_bytes + extra * <int64_t>opt.overhead_multiply


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _gen_linear(uint32_t[:] values, inexpr_base0, int addition):
    cdef int32_t slope = values[1] - values[0]
    # (c - lo) * slope + addition
    expr = inexpr_base0
    if slope != 1:
        expr = expr * slope
    add = <int>values[0] + addition
    if add:
        expr = expr + add
    return expr


@cython.boundscheck(False)
cdef _gen_uniq2_code(values, uint32_t[::1] uniqs, uint32_t num, inexpr,
                     inexpr_base0, uint32_t lo, int addition):
    cdef uint32_t value0
    cdef uint32_t value1
    cdef uint64_t bit_test_mask

    cdef uint32_t k = utils.const_range(values)
    cdef uint32_t rk = utils.const_range(values[::-1])
    if k + rk == num:  # [a,...,a,b,...,b]
        if k == 1:
            return Cond(inexpr == lo, values[0] + addition,
                        values[-1] + addition)
        elif rk == 1:
            return Cond(inexpr == lo + k, values[-1] + addition,
                        values[0] + addition)
        else:
            return Cond(inexpr < lo + k, values[0] + addition,
                        values[-1] + addition)

    elif values[0] == values[-1] and utils.is_const(values[k:num-rk]):
        # [a, ..., a, b, ..., b, a, ..., a]
        bcount = num - rk - k
        if bcount == 1:
            cond = (inexpr == lo + k)
        else:
            subinexpr = Cast(32, inexpr - (lo + k))
            cond = (subinexpr < bcount)
        return Cond(cond, values[k] + addition, values[0] + addition)

    elif num <= 64:
        # Use bit test.
        bit_test_mask = 0
        value0 = uniqs[0]
        value1 = uniqs[1]
        # Prefer value1 == value0 + 1, but if that's impossible,
        # prefer the bitmask to be smaller
        if value1 == value0 + 1:
            pass
        elif value1 == values[-1]:
            value0, value1 = value1, value0

        for k in (values == value1).nonzero()[0]:
            bit_test_mask |= 1 << int(k)

        Bits = 64 if num > 32 else 32
        expr = (Const(Bits, bit_test_mask) >> inexpr_base0) & 1

        if value1 == value0 + 1:
            if Bits > 32:
                expr = Cast(32, expr)
            if value0 + addition != 0:
                expr = expr + (value0 + addition)
        else:
            expr = Cond(expr, value1 + addition, value0 + addition)
        return expr


cdef _gen_split(uint32_t lo, values, inexpr, inexpr_long, inexpr_base0,
                uint32_t num, uint32_t maxv,
                uint32_t maxv_bits, str table_name, int addition, int maxdepth,
                _make_code, uint32_t threshold):
    cdef uint32_t const_prefix_len = utils.const_range(values)
    cdef uint32_t const_suffix_len
    if const_prefix_len >= threshold:
        split_pos = const_prefix_len
        comp_expr = (inexpr < (lo + split_pos))
        left_expr = Const(32, int(values[0]) + addition)
        right_expr = _make_code(
            lo + split_pos, values[split_pos:],
            table_name + '_r',
            inexpr, inexpr_long,
            addition=addition,
            maxdepth=maxdepth-1)
        return Cond(comp_expr, left_expr, right_expr)
    else:
        const_suffix_len = utils.const_range(values[::-1])
        if const_suffix_len >= threshold:
            split_pos = num - const_suffix_len
            comp_expr = (inexpr < (lo + split_pos))
            left_expr = _make_code(
                lo, values[:split_pos],
                table_name + '_l',
                inexpr, inexpr_long, inexpr_base0,
                addition=addition,
                maxdepth=maxdepth-1)
            right_expr = Const(32, int(values[-1]) + addition)
            return Cond(comp_expr, left_expr, right_expr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _gen_pack_as_64_code(uint32_t[:] values, inexpr_base0,
                          uint32_t num, uint32_t maxv,
                          uint32_t maxv_bits, int addition):
    '''Generate code to pack multiple values into one uint64_t
    '''
    cdef uint32_t offset = 0
    cdef uint32_t bits = maxv_bits
    cdef uint32_t Bits = 64
    if num * maxv_bits <= 32:  # 32-bit is sufficient
        Bits = 32
    if (addition > 0) and num * bit_length(addition + maxv) <= Bits:
        bits = bit_length(addition + maxv)
        offset = addition
    # Usually it's faster to multiply by power of 2
    cdef uint32_t BT = 8
    while BT <= 32:
        # +1: Multiply by (1 + power of 2) isn't slow.
        if (BT//2 + 1 < bits < BT) and (num * BT <= Bits):
            bits = BT
            break
        BT *= 2
    cdef uint64_t mask = 0
    for k in range(len(values)):
        mask |= (<uint64_t>values[k] + offset) << (<Py_ssize_t>k * bits)
    expr = Const(Bits, mask) >> (inexpr_base0 * bits)
    if Bits > 32:
        expr = Cast(32, expr)
    expr = expr & ((1 << bits) - 1)
    cdef int64_t add = addition - offset
    if add:
        expr = expr + add
    return expr


ctypedef pair[int32_t, uint32_t] SlopePair
cdef vector[SlopePair] _gen_possible_slopes(np_values):
    '''Is values almost linear?
    Return likely slopes list[(numerator, denominator), ...]
    Returned list is ordered by denominator from smallest to biggest
    '''
    cdef uint32_t[::1] values = np_values
    cdef uint32_t num = len(values)

    cdef vector[SlopePair] res

    # Most common adjacent differences.
    # The remainder array will have many equal values
    slope, slope_count = utils.most_common_element_count(
                utils.slope_array(np_values, np.int64))
    if slope and -0x7fffffff <= slope <= 0x80000000 and \
            <uint32_t>slope_count * 3 >= num:
        res.push_back(SlopePair(slope, 1))

    # Linear regression
    cdef float slope_linregress = linregress_slope(np_values)
    slope_frac = Fraction(slope_linregress)

    cdef uint32_t last_denominator = 0
    cdef uint32_t max_denominator
    cdef uint32_t i
    for i in range(17):
        max_denominator = <uint32_t>1 << <uint32_t>i
        slope_limited = slope_frac.limit_denominator(max_denominator)

        for slope in (slope_limited,
                      Fraction(int(slope * max_denominator), max_denominator)):
            numerator = slope.numerator
            denominator = slope.denominator
            if numerator != 0 and denominator > last_denominator and \
                    -0x80000000 <= numerator * (num - 1) <= 0x7fffffff:
                last_denominator = denominator
                res.push_back(SlopePair(<int32_t>numerator,
                                        <uint32_t>denominator))

        # If slope_limited is already very close to the real slope,
        # don't try more
        if abs(<float>float(slope_limited) - slope_linregress) * num <= 1:
            break

    return std_move(res)


cdef _prepare_almost_linear_tasks(values, uint32_t num, uint32_t uniq,
                                  uint32_t maxv_bits):
    '''Prepare for "almost linear" reduction: values = slope * index + reduced
    '''
    cdef uint32_t best_uniq = uniq
    cdef uint32_t best_bits = maxv_bits
    cdef uint32_t reduced_uniq
    cdef uint32_t reduced_maxv_bits

    cdef vector[SlopePair] slopes = _gen_possible_slopes(values)

    linear_tasks = []

    cdef SlopePair slope_pair
    # cdef int32_t slope_num
    # cdef uint32_t slope_denom
    for slope_pair in slopes:
        slope_num = slope_pair.first
        slope_denom = slope_pair.second
        reduced_values = values - \
            np.arange(0, slope_num * num, slope_num, dtype=np.int64) \
            // slope_denom
        # Negative values may cause problems
        offset = utils.np_min(reduced_values)
        reduced_values = np.subtract(reduced_values, offset,
                                     dtype=np.uint32, casting='unsafe')
        # Be careful to avoid infinite recursion
        reduced_uniqs = utils.np_unique(reduced_values)
        reduced_uniq = reduced_uniqs.shape[0]
        reduced_maxv_bits = bit_length(reduced_uniqs[-1])
        if (reduced_uniq * 2 <= uniq and reduced_uniq < best_uniq) or \
                reduced_maxv_bits < best_bits:
            best_uniq = min(best_uniq, reduced_uniq)
            best_bits = min(best_bits, reduced_maxv_bits)

            linear_tasks.append((reduced_values, reduced_uniqs, offset,
                                 slope_num, slope_denom))

    return linear_tasks


cdef _gen_strides(uint32_t lo, values, uint32_t num, uint32_t maxv_bits,
                  inexpr, inexpr_long, inexpr_base0,
                  int addition, int maxdepth, str table_name, _make_code):
    '''Generate strided expressions.

    Strides are consecutive values that are similar in value
    '''
    res = []
    values_nonzeros = np.count_nonzero(values)
    cdef uint32_t stride = 1024
    while stride >= 2:
        if stride * 8 > num:
            stride >>= 1
            continue
        base_values = utils.np_min_by_chunk(values, stride)
        delta = values - np.repeat(base_values, stride)[:num]
        if (np.count_nonzero(delta) < values_nonzeros / (stride * .9)
                or utils.np_max(delta).bit_length() <= maxv_bits // 2):
            base_inexpr = inexpr_base0 // stride
            base_expr = _make_code(
                    0, base_values,
                    table_name + '_stride{}'.format(stride),
                    base_inexpr, addition=addition, skip_stride=True,
                    maxdepth=maxdepth-1)
            delta_expr = _make_code(
                    lo, delta,
                    table_name + '_stride{}delta'.format(stride),
                    inexpr, inexpr_long, inexpr_base0,
                    maxdepth=maxdepth-1)
            res.append(base_expr + delta_expr)
        stride >>= 1
    return res


cdef _gen_compressed(uint32_t lo, values, uint32_t num, uint32_t uniq,
                     uint32_t maxv, uint32_t maxv_bits, inexpr, inexpr_long,
                     inexpr_base0, int addition, int maxdepth, str table_name,
                     _make_code):
    cdef int offset
    if maxv_bits in (3, 4) and num > 16 and maxdepth > 0:
        offset = 0
        if (addition > 0) and (addition + maxv < 16):
            offset = addition
        compressed_values = utils.compress_array(values + offset, 2)

        expr = inexpr_base0
        # (table[expr/2] >> (expr%2*4)) & 15
        expr_shift = expr >> 1
        expr_left = _make_code(0, compressed_values, table_name + '_4bits',
                               expr_shift, skip_split_4bits_hi_lo=False,
                               maxdepth=maxdepth-1)
        expr_right = (expr & 1) << 2
        expr = (expr_left >> expr_right) & 15
        expr = expr + (addition - offset)
        return expr

    # Try using "compressed" table. 4->1
    if maxv_bits == 2 and num > 32 and maxdepth > 0:
        compressed_values = utils.compress_array(values, 4)

        expr = inexpr_base0
        # (table[expr/4] >> (expr%4*2)) & 3
        expr_shift = expr >> 2
        expr_left = _make_code(0, compressed_values, table_name + '_2bits',
                               expr_shift, maxdepth=maxdepth-1)
        expr_right = (expr & 3) << 1
        expr = (expr_left >> expr_right) & 3
        expr = expr + addition
        return expr

    # Try using "bitvector". 8->1
    if (maxv == 1) and num > 64 and maxdepth > 0:
        compressed_values = utils.compress_array(values, 8)

        expr = inexpr_base0
        # (table[expr/8] >> (expr%8)) & 1
        expr_shift = expr >> 3
        expr_left = _make_code(0, compressed_values, table_name + '_bitvec',
                               expr_shift, maxdepth=maxdepth-1)
        expr_right = expr & 7
        expr = (expr_left >> expr_right) & 1
        expr = expr + addition
        return expr


@cython.boundscheck(False)
cdef _gen_gcd(uint32_t lo, values, uint32_t maxv, inexpr, inexpr_long,
              inexpr_base0, str table_name, int addition, int maxdepth,
              _make_code):
    cdef uint32_t gcd = utils.gcd_reduce(values)
    cdef uint32_t offset
    if gcd > 1:
        offset = <uint32_t>values[0] % gcd
        reduced_values = values // gcd
        expr = _make_code(lo, reduced_values, table_name + '_gcd',
                               inexpr, inexpr_long, inexpr_base0,
                               skip_gcd_reduce=True,
                               maxdepth=maxdepth-1)
        expr = expr * gcd
        if addition + offset:
            expr = expr + (addition + offset)

        # If the divided values are significantly smaller, we
        # likely dont' need to retry the origianl values
        return expr, const_type(maxv // gcd) == const_type(maxv)

    else:
        return None, True


cdef _gen_lo_hi(uint32_t lo, values, uint32_t num, uint32_t uniq,
                uint32_t maxv_bits, inexpr, inexpr_long,
                inexpr_base0, addition, int maxdepth, str table_name,
                _make_code, c_bool skip_split_4bits_hi_lo):
    '''Split values into low and high parts
    '''
    res = []
    cdef uint32_t k
    cdef uint32_t lo_uniq, hi_uniq
    for k in (4, 8, 16):
        if k >= maxv_bits:
            break
        if k == 4 and skip_split_4bits_hi_lo:
            continue

        lomask = np.uint32((1 << k) - 1)
        lo_values = values & lomask
        hi_values = values - lo_values

        lo_uniqs = utils.np_unique(lo_values)
        lo_uniq = lo_uniqs.size
        if lo_uniq < 2:
            continue
        hi_uniqs = utils.np_unique(hi_values)
        hi_uniq = hi_uniqs.size
        if hi_uniq < 2:
            break

        hi_gcd = utils.gcd_many(hi_values)
        hi_values //= hi_gcd

        if k == 4:
            # We must be very careful when k == 4
            # If we just split it, two 4-bit values will be joined
            # in the recursion, and then split again, resulting in
            # almost uncontrollable recursions
            if max(lo_uniq, hi_uniq) <= 4:
                pass
            else:
                continue

        elif min(lo_uniq, hi_uniq) <= min(1 << (k - 1), uniq // 2):
            pass

        elif bit_length(utils.np_range(hi_values)) <= k // 2:
            pass

        else:
            # If none of the conditions meet, we consider the split not
            # helpful.  We cannot just try out.  That'd be too slow.
            continue

        # Fix hi_uniqs as soon as we decide to proceed
        hi_uniqs //= hi_gcd

        lo_expr = _make_code(
            lo, lo_values, '{}_{}lo'.format(table_name, k),
            inexpr, inexpr_long, inexpr_base0, addition=addition,
            uniqs=lo_uniqs,
            maxdepth=maxdepth-1)
        hi_expr = _make_code(
            lo, hi_values, '{}_{}hi'.format(table_name, k),
            inexpr, inexpr_long, inexpr_base0,
            uniqs=hi_uniqs,
            maxdepth=maxdepth-1)
        res.append(lo_expr + hi_expr * hi_gcd)

        if hi_uniq <= 2:  # No reason to continue trying
            break

    return res


cdef _gen_two_level_lookup(values, uniqs, uint32_t uniq, uint32_t lo,
                           str table_name,
                           inexpr, inexpr_long, inexpr_base0, int addition,
                           int maxdepth, _make_code, _yield_code):
    indices = np.searchsorted(uniqs, values)
    # Level 1
    expr = _make_code(lo, indices, table_name + '_index',
                      inexpr, inexpr_long, inexpr_base0,
                      uniqs=np.arange(uniq, dtype=np.uint32),
                      maxdepth=maxdepth-1)

    if <uint32_t>expr.rtype % 8 != 0:
        # Signed return type - convert to unsigned, becuase inexpr
        # of _yield_code must be unsigned
        expr = Cast(32, expr)

    # Level 2
    return _yield_code(0, uniqs, table_name + '_value',
                       expr, expr,
                       addition=addition, uniqs=uniqs, maxdepth=maxdepth-1)


class MakeCodeForRange:
    def __init__(self, lo, values, table_name, opt, thread_pool):
        self._lo = lo
        self.values = values
        self._table_name = table_name
        self._opt = opt
        self._thread_pool = thread_pool

    __var_c = FixedVar(32, 'c')
    __var_cl = FixedVar(64, 'cl')

    @utils.cached_property
    def expr_tuple(self):
        expr = self._make_code(
            self._lo, self.values, self._table_name,
            self.__var_c, self.__var_cl, maxdepth=6)

        # Remember how many times each expression is visited
        # Expressions appearing more than once are always extracted
        # (except variables and constants)
        visited_times = defaultdict(int)

        def visit(expr):
            if expr.IS_VAR or expr.IS_CONST:
                return
            idx = id(expr)
            visited_times[idx] += 1
            if visited_times[idx] == 1:
                for subexpr in expr.children:
                    visit(subexpr)

        visit(expr)

        subexprs = []
        subexpr_rev = {}

        def make_subexpr(expr, allow_extract_table, prefer_extracted):
            if expr.IS_VAR:
                return expr
            idx = id(expr)
            ind = subexpr_rev.get(idx)
            if ind is None:
                if not prefer_extracted and visited_times.get(idx, 0) <= 1:
                    return expr
                if not allow_extract_table and expr.has_table:
                    return expr
                ind = len(subexprs)
                subexprs.append(expr)
                subexpr_rev[idx] = ind

            return TempVar(expr.rtype, ind)

        # Extract complicated expressions
        # as variables for code readability
        expr.extract_subexprs(5, make_subexpr, True)

        # Final optimization: Remove unnecessary explicit upcast
        while expr.IS_CAST and \
                (expr.rtype >= 31 or expr.rtype >= expr.value.rtype):
            expr = expr.value

        return expr, subexprs

    def _make_code(self, lo, values, table_name, inexpr, inexpr_long=None,
                   inexpr_base0=None, int addition=0, *, int maxdepth,
                   **kwargs):
        exprs = self._yield_code(lo, values, table_name, inexpr,
                                 inexpr_long, inexpr_base0, addition=addition,
                                 maxdepth=maxdepth, **kwargs)
        min_expr = None
        cdef int64_t min_overhead = 0
        cdef int64_t overhead
        for expr in exprs:
            expr = expr.optimized
            overhead = _overhead(expr, self._opt)
            if min_expr is None or overhead < min_overhead:
                min_expr = expr
                min_overhead = overhead
        return min_expr

    def _yield_code(self, uint32_t lo, values, str table_name, inexpr,
                    inexpr_long=None, inexpr_base0=None,
                    *,
                    int addition=0,
                    int maxdepth,
                    uniqs=None, skip_gcd_reduce=False,
                    skip_almost_linear_reduce=False,
                    skip_split_4bits_hi_lo=False, skip_stride=False):
        """
        Everything in values must be non-negative; addition may be negative.
        The final result has type uint32_t even if it may be negative.
        """
        res = []

        if values.dtype != np.uint32:
            values = np.array(values, np.uint32)

        cdef uint32_t num = values.size
        # hi = lo + num

        cdef uint32_t minv

        if uniqs is None:
            minv = utils.np_min(values)
            if minv != 0:
                values = values - minv
                addition += minv

            uniqs = utils.np_unique(values)

        else:
            minv = uniqs[0]
            if minv:
                values = values - minv
                uniqs = uniqs - minv
                addition += minv

        cdef uint32_t uniq = uniqs.size
        cdef uint32_t maxv = uniqs[-1]
        # From now on, we cannot use minv (assume it to be 0, but not actually
        # assign 0 to it)

        cdef uint32_t maxv_bits = bit_length(maxv)

        assert(uniq > 0)

        if inexpr is inexpr_long or inexpr_long is None:
            inexpr = inexpr_long = inexpr.optimized
        else:
            inexpr = inexpr.optimized
            inexpr_long = inexpr_long.optimized

        if lo == 0:
            inexpr_base0 = inexpr
        elif inexpr_base0:
            inexpr_base0 = inexpr_base0.optimized
        else:
            inexpr_base0 = (inexpr - lo).optimized

        assert inexpr.rtype in (8, 16, 32), inexpr
        assert inexpr_long.rtype in (8, 16, 32, 64), inexpr_long
        assert inexpr_base0.rtype in (8, 16, 32), inexpr_base0

        # Constant
        if uniq == 1:
            res.append(Const(32, addition))
            return res

        # [0,1] => [c1,c2]. Better to use ExprCond than linear
        cdef long long c0
        cdef long long c1
        if (lo == 0) and (num == 2):
            c0 = <long long>values[0] + addition
            c1 = <long long>values[1] + addition
            if c1 == c0 + 1:
                if c0 == 0:
                    res.append(inexpr)
                else:
                    res.append(inexpr + c0)
            elif c0 == 0 and (c1 & (c1 - 1)) == 0:
                res.append(inexpr << (bit_length(c1) - 1))
            else:
                res.append(Cond(inexpr, c1, c0))
            return res

        # Linear
        if (uniq == num) and utils.is_linear(values):
            res.append(_gen_linear(values, inexpr_base0, addition))
            return res

        # Not linear, but only two distinct values.
        if uniq == 2:
            expr = _gen_uniq2_code(values, uniqs, num, inexpr, inexpr_base0,
                                   lo, addition)
            if expr is not None:
                res.append(expr)
                return res

        # Has cycles
        if num > uniq >= 2 and maxdepth > 0:
            cycle = utils.np_cycle(values, max_cycle=max(num//2, num-32))
            if cycle:
                res.extend(self._yield_code(0, values[:cycle],
                                            table_name + '_cycle',
                                            inexpr_base0 % cycle,
                                            addition=addition,
                                            maxdepth=maxdepth-1))
                return res

        # Has long constant prefix or suffix.
        # Frequently resulting from bitvec or lo/hi partition
        if uniq >= 2 and num > self._opt.const_threshold and maxdepth > 0:
            threshold = max(self._opt.const_threshold, num // 4)
            expr = _gen_split(lo, values, inexpr, inexpr_long, inexpr_base0,
                              num, maxv, maxv_bits, table_name,
                              addition, maxdepth, self._make_code, threshold)
            if expr:
                res.append(expr)

        # Can we pack them into one 64-bit integer?
        if num * maxv_bits <= 64:
            res.append(_gen_pack_as_64_code(values, inexpr_base0, num, maxv,
                                       maxv_bits, addition))
            return res

        # Most elements are almost linear, but a few outliers exist.
        if not skip_almost_linear_reduce and maxdepth > 0 and num >= 3 and \
                maxv >= 4:

            linear_tasks = _prepare_almost_linear_tasks(
                values, num, uniq, maxv_bits)

            def gen_linear(args):
                reduced_values, reduced_uniqs, offset, slope_num, slope_denom\
                    = args
                if slope_num > 0:
                    linear_key = f'linear{slope_num}'
                else:
                    linear_key = f'linearM{-slope_num}'
                if slope_denom > 1:
                    linear_key += f'X{slope_denom}'
                expr = self._make_code(lo, reduced_values,
                                       f'{table_name}_{linear_key}',
                                       inexpr, inexpr_long,
                                       inexpr_base0,
                                       addition=offset+addition,
                                       uniqs=reduced_uniqs,
                                       skip_almost_linear_reduce=True,
                                       maxdepth=maxdepth-1)

                return inexpr_base0 * slope_num // slope_denom + expr

            res.extend(utils.thread_pool_map(
                self._thread_pool, gen_linear, linear_tasks))

        # Consecutive values are similar
        if not skip_stride and maxdepth > 0:
            res.extend(_gen_strides(lo, values, num, maxv_bits,
                                    inexpr, inexpr_long,
                                    inexpr_base0, addition, maxdepth,
                                    table_name, self._make_code))

        # Two-level lookup?
        if maxv > num > uniq * 4 // 3 and maxdepth > 0:
            res.extend(_gen_two_level_lookup(
                values, uniqs, uniq, lo, table_name, inexpr, inexpr_long,
                inexpr_base0, addition, maxdepth,
                self._make_code, self._yield_code))

        # Try using "compressed" table. 2->1
        expr = _gen_compressed(lo, values, num, uniq, maxv, maxv_bits, inexpr,
                               inexpr_long, inexpr_base0, addition, maxdepth,
                               table_name, self._make_code)
        if expr:
            res.append(expr)
            return res

        # GCD may help
        cdef int64_t gcd
        if not skip_gcd_reduce and maxdepth > 0:
            expr, cont = _gen_gcd(lo, values, maxv, inexpr, inexpr_long,
                                  inexpr_base0, table_name, addition, maxdepth,
                                  self._make_code)
            if expr:
                res.append(expr)
                if not cont:
                    return res

        # Try splitting the data into low and high parts
        if maxdepth > 0:
            res.extend(_gen_lo_hi(
                lo, values, num, uniq, maxv_bits,
                inexpr, inexpr_long, inexpr_base0,
                addition, maxdepth, table_name, self._make_code,
                skip_split_4bits_hi_lo))

        # Finally fall back to the simplest one-level table
        table_type = const_type(maxv)
        cdef uint32_t table_max
        if addition > 0:
            table_max = type_max(table_type)
            if maxv + addition <= table_max:
                values = values + addition
                addition = 0
            else:
                values = values + (table_max - maxv)
                addition -= table_max - maxv

        expr = ExprTable(table_type, table_name, values, inexpr_long, lo)
        if addition != 0:
            expr = expr + addition
        res.append(expr)
        return res
