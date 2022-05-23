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
from scipy.stats import linregress

from .expr import *
from . import groupify
from . import utils


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

    if not added_var:
        # No temporary variable
        main_str = '\n      return {};\n'.format(
            utils.trim_brackets(str(expr)))
        main_statics = expr.statics(visited_set)
        return main_str, (main_statics,)

    else:
        # Rename temporary variables
        rename_map = {var: k for (k, var) in enumerate(var_output_order)}
        renamed_set = set()

        def rename_var(expr):
            for x in expr.walk_tempvar():
                if id(x) not in renamed_set:
                    renamed_set.add(id(x))
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
                   inexpr_base0=None, addition=0, *, maxdepth, **kwargs):
        exprs = (expr.optimized for expr in
                 self._yield_code(lo, values, table_name, inexpr,
                                  inexpr_long, inexpr_base0, addition=addition,
                                  maxdepth=maxdepth, **kwargs))
        return min(exprs, key=self._overhead)

    def _yield_code(self, lo, values, table_name, inexpr, inexpr_long=None,
                    inexpr_base0=None,
                    *,
                    addition=0,
                    maxdepth,
                    uniqs=None, skip_gcd_reduce=False,
                    skip_almost_linear_reduce=False,
                    skip_split_4bits_hi_lo=False, skip_stride=False,
                    int=int, np=np, utils=utils, Const=Const, Cast=Cast,
                    Cond=Cond):
        """
        Everything in values must be non-negative; addition may be negative.
        The final result has type uint32_t even if it may be negative.

        Yield:
            instances of some subclass of Expr
        """
        if values.dtype != np.uint32:
            values = np.array(values, np.uint32)

        num = values.size
        # hi = lo + num

        if uniqs is None:
            minv = utils.np_min(values)
            if minv != 0:
                values = values - minv
                addition += minv

            uniqs = utils.np_unique(values)

        else:
            minv = int(uniqs[0])
            if minv:
                values = values - minv
                uniqs = uniqs - minv
                addition += minv

        uniq = uniqs.size
        maxv = int(uniqs[-1])
        minv = 0

        maxv_bits = maxv.bit_length()

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
            yield Const(32, addition)
            return

        # [0,1] => [c1,c2]. Better to use ExprCond than linear
        if (lo == 0) and (num == 2):
            c0, c1 = map(int, values + addition)
            if c1 == c0 + 1:
                if c0 == 0:
                    yield inexpr
                else:
                    yield inexpr + c0
            elif c0 == 0 and (c1 & (c1 - 1)) == 0:
                yield inexpr << (c1.bit_length() - 1)
            else:
                yield Cond(inexpr, c1, c0)
            return

        # Linear
        if (uniq == num) and utils.is_linear(values):
            slope = int(values[1]) - int(values[0])
            # (c - lo) * slope + addition
            expr = inexpr_base0
            if slope != 1:
                expr = expr * slope
            add = int(values[0]) + addition
            if add:
                expr = expr + add
            yield expr
            return

        # Not linear, but only two distinct values.
        if uniq == 2:
            k = utils.const_range(values)
            rk = utils.const_range(values[::-1])
            if k + rk == num:  # [a,...,a,b,...,b]
                if k == 1:
                    yield Cond(inexpr == lo,
                               values[0] + addition,
                               values[-1] + addition)
                elif rk == 1:
                    yield Cond(inexpr == lo + k,
                               values[-1] + addition,
                               values[0] + addition)
                else:
                    yield Cond(inexpr < lo + k,
                               values[0] + addition,
                               values[-1] + addition)
                return

            elif values[0] == values[-1] and utils.is_const(values[k:num-rk]):
                # [a, ..., a, b, ..., b, a, ..., a]
                bcount = num - rk - k
                if bcount == 1:
                    cond = (inexpr == lo + k)
                else:
                    subinexpr = Cast(32, inexpr - (lo + k))
                    cond = (subinexpr < bcount)
                yield Cond(cond, values[k] + addition, values[0] + addition)
                return

            elif num <= 64:
                # Use bit test.
                mask = 0
                value0 = int(uniqs[0])
                value1 = int(uniqs[1])
                # Prefer value1 == value0 + 1, but if that's impossible,
                # prefer the bitmask to be smaller
                if value1 == value0 + 1:
                    pass
                elif value1 == values[-1]:
                    value0, value1 = value1, value0

                for k in (values == value1).nonzero()[0]:
                    mask |= 1 << int(k)

                Bits = 64 if num > 32 else 32
                expr = (Const(Bits, mask) >> inexpr_base0) & 1

                if value1 == value0 + 1:
                    if Bits > 32:
                        expr = Cast(32, expr)
                    if value0 + addition != 0:
                        expr = expr + (value0 + addition)
                else:
                    expr = Cond(expr, value1 + addition, value0 + addition)
                yield expr
                return

        # Has cycles
        if num > uniq >= 2 and maxdepth > 0:
            min_cycle = 2
            cycle = utils.np_cycle(values,
                                   max_cycle=max(num//2, num-32))
            if cycle:
                yield from self._yield_code(0, values[:cycle],
                                            table_name + '_cycle',
                                            inexpr_base0 % cycle,
                                            addition=addition,
                                            maxdepth=maxdepth-1)
                return

        # Has long constant prefix or suffix.
        # Frequently resulting from bitvec or lo/hi partition
        if uniq >= 2 and num > self._opt.const_threshold and maxdepth > 0:
            threshold = max(self._opt.const_threshold, num // 4)
            const_prefix_len = utils.const_range(values)
            if const_prefix_len >= threshold:
                split_pos = const_prefix_len
                comp_expr = (inexpr < (lo + split_pos))
                left_expr = Const(32, int(values[0]) + addition)
                right_expr = self._make_code(
                    lo + split_pos, values[split_pos:],
                    table_name + '_r',
                    inexpr, inexpr_long,
                    addition=addition,
                    maxdepth=maxdepth-1)
                yield Cond(comp_expr, left_expr, right_expr)
            else:
                const_suffix_len = utils.const_range(values[::-1])
                if const_suffix_len >= threshold:
                    split_pos = num - const_suffix_len
                    comp_expr = (inexpr < (lo + split_pos))
                    left_expr = self._make_code(
                        lo, values[:split_pos],
                        table_name + '_l',
                        inexpr, inexpr_long, inexpr_base0,
                        addition=addition,
                        maxdepth=maxdepth-1)
                    right_expr = Const(32, int(values[-1]) + addition)
                    yield Cond(comp_expr, left_expr, right_expr)

        # Can we pack them into one 64-bit integer?
        if num * maxv_bits <= 64:
            offset = 0
            bits = maxv_bits
            Bits = 64
            if num * maxv_bits <= 32:  # 32-bit is sufficient
                Bits = 32
            if (addition > 0) and num * (addition + maxv).bit_length() <= Bits:
                bits = (addition + maxv).bit_length()
                offset = addition
            # Usually it's faster to multiply by power of 2
            for BT in 8, 16, 32:
                # +1: Multiply by (1 + power of 2) isn't slow.
                if (BT//2 + 1 < bits < BT) and (num * BT <= Bits):
                    bits = BT
                    break
            mask = 0
            for k, v in enumerate(values):
                mask |= (int(v) + offset) << (k * bits)
            expr = Const(Bits, mask) >> (inexpr_base0 * bits)
            if Bits > 32:
                expr = Cast(32, expr)
            expr = expr & ((1 << bits) - 1)
            add = addition - offset
            if add:
                expr = expr + add
            yield expr
            return

        # Most elements are almost linear, but a few outliers exist.
        if not skip_almost_linear_reduce and maxdepth > 0 and num >= 3 and \
                maxv >= 4:
            best_uniq = uniq
            best_bits = maxv_bits

            slopes = sorted(set(self._yield_possible_slopes(values)),
                            key=lambda x: x[1])

            linear_tasks = []

            for slope_num, slope_denom in slopes:
                reduced_values = values - \
                    np.arange(0, slope_num * num, slope_num, dtype=np.int64) \
                    // slope_denom
                # Negative values may cause problems
                offset = utils.np_min(reduced_values)
                reduced_values = np.subtract(reduced_values, offset,
                                             dtype=np.uint32, casting='unsafe')
                # Be careful to avoid infinite recursion
                reduced_uniqs = utils.np_unique(reduced_values)
                reduced_uniq, = reduced_uniqs.shape
                reduced_maxv_bits = int(reduced_uniqs[-1]).bit_length()
                if (reduced_uniq * 2 <= uniq and reduced_uniq < best_uniq) or \
                        reduced_maxv_bits < best_bits:
                    best_uniq = min(best_uniq, reduced_uniq)
                    best_bits = min(best_bits, reduced_maxv_bits)

                    linear_tasks.append((reduced_values, reduced_uniqs, offset,
                                         slope_num, slope_denom))

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

            yield from utils.thread_pool_map(
                self._thread_pool, gen_linear, linear_tasks)

        # Consecutive values are similar
        if not skip_stride and maxdepth > 0:
            values_nonzeros = np.count_nonzero(values)
            for stride in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2):
                if stride * 8 > num:
                    continue
                base_values = utils.np_min_by_chunk(values, stride)
                delta = values - np.repeat(base_values, stride)[:num]
                if (np.count_nonzero(delta) < values_nonzeros / (stride * .9)
                        or utils.np_max(delta).bit_length() <= maxv_bits // 2):
                    base_inexpr = inexpr_base0 // stride
                    base_expr = self._make_code(
                            0, base_values,
                            table_name + '_stride{}'.format(stride),
                            base_inexpr, addition=addition, skip_stride=True,
                            maxdepth=maxdepth-1)
                    delta_expr = self._make_code(
                            lo, delta,
                            table_name + '_stride{}delta'.format(stride),
                            inexpr, inexpr_long, inexpr_base0,
                            maxdepth=maxdepth-1)
                    yield base_expr + delta_expr

        # Two-level lookup?
        if maxv > num > uniq * 4 // 3 and maxdepth > 0:
            indices = np.searchsorted(uniqs, values)
            # Level 1
            expr = self._make_code(lo, indices, table_name + '_index',
                                   inexpr, inexpr_long, inexpr_base0,
                                   uniqs=np.arange(uniq, dtype=np.uint32),
                                   maxdepth=maxdepth-1)

            if expr.rtype % 8 != 0:
                # Signed return type - convert to unsigned, becuase inexpr
                # of _yield_code must be unsigned
                expr = Cast(32, expr)

            # Level 2
            yield from self._yield_code(0, uniqs, table_name + '_value',
                                        expr, expr,
                                        addition=addition,
                                        uniqs=uniqs,
                                        maxdepth=maxdepth-1)

        # Try using "compressed" table. 2->1
        if maxv_bits in (3, 4) and num > 16 and maxdepth > 0:
            offset = 0
            if (addition > 0) and (addition + maxv < 16):
                offset = addition
            compressed_values = utils.compress_array(values + offset, 2)

            expr = inexpr_base0
            # (table[expr/2] >> (expr%2*4)) & 15
            expr_shift = expr >> 1
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_4bits',
                                        expr_shift,
                                        skip_split_4bits_hi_lo=False,
                                        maxdepth=maxdepth-1)
            expr_right = (expr & 1) << 2
            expr = (expr_left >> expr_right) & 15
            expr = expr + (addition - offset)
            yield expr
            return

        # Try using "compressed" table. 4->1
        if maxv_bits == 2 and num > 32 and maxdepth > 0:
            compressed_values = utils.compress_array(values, 4)

            expr = inexpr_base0
            # (table[expr/4] >> (expr%4*2)) & 3
            expr_shift = expr >> 2
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_2bits',
                                        expr_shift, maxdepth=maxdepth-1)
            expr_right = (expr & 3) << 1
            expr = (expr_left >> expr_right) & 3
            expr = expr + addition
            yield expr
            return

        # Try using "bitvector". 8->1
        if (maxv == 1) and num > 64 and maxdepth > 0:
            compressed_values = utils.compress_array(values, 8)

            expr = inexpr_base0
            # (table[expr/8] >> (expr%8)) & 1
            expr_shift = expr >> 3
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_bitvec',
                                        expr_shift, maxdepth=maxdepth-1)
            expr_right = expr & 7
            expr = (expr_left >> expr_right) & 1
            expr = expr + addition
            yield expr
            return

        # GCD may help
        if not skip_gcd_reduce and maxdepth > 0:
            gcd = utils.gcd_reduce(values)
            if gcd > 1:
                offset = int(values[0]) % gcd
                reduced_values = values // gcd
                expr = self._make_code(lo, reduced_values, table_name + '_gcd',
                                       inexpr, inexpr_long, inexpr_base0,
                                       skip_gcd_reduce=True,
                                       maxdepth=maxdepth-1)
                expr = expr * gcd
                if addition + offset:
                    expr = expr + (addition + offset)
                yield expr

                # If the divided values are significantly smaller, we
                # likely dont' need to retry the origianl values
                if const_type(maxv // gcd) != const_type(maxv):
                    return

        # Try splitting the data into low and high parts
        if maxdepth > 0:
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

                elif utils.np_range(hi_values).bit_length() <= k // 2:
                    pass

                else:
                    # If none of the conditions meet, we consider the split not
                    # helpful.  We cannot just try out.  That'd be too slow.
                    continue

                # Fix hi_uniqs as soon as we decide to proceed
                hi_uniqs //= hi_gcd

                lo_expr = self._make_code(
                    lo, lo_values, '{}_{}lo'.format(table_name, k),
                    inexpr, inexpr_long, inexpr_base0, addition=addition,
                    uniqs=lo_uniqs,
                    maxdepth=maxdepth-1)
                hi_expr = self._make_code(
                    lo, hi_values, '{}_{}hi'.format(table_name, k),
                    inexpr, inexpr_long, inexpr_base0,
                    uniqs=hi_uniqs,
                    maxdepth=maxdepth-1)
                yield lo_expr + hi_expr * hi_gcd

                if hi_uniq <= 2:  # No reason to continue trying
                    break

        # Finally fall back to the simplest one-level table
        table_type = const_type(maxv)
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
        yield expr

    def _yield_possible_slopes(self, values):
        '''Is values almost linear?
        Yield likely slopes (numerator, denominator)
        '''
        num, = values.shape

        # Most common adjacent differences.
        # The remainder array will have many equal values
        slope, slope_count = utils.most_common_element_count(
                    utils.slope_array(values, np.int64))
        if slope and slope_count * 3 >= num:
            yield slope, 1

        # Linear regression
        res = linregress(np.arange(num, dtype=np.float32),
                         values.astype(np.float32))
        slope = res.slope
        slope_frac = Fraction(slope)

        last_denominator = 0

        def try_slope(slope):
            nonlocal last_denominator
            numerator = slope.numerator
            denominator = slope.denominator
            if numerator > 0 and denominator > last_denominator and \
                    -0x80000000 <= numerator * (num - 1) <= 0x7fffffff:
                last_denominator = denominator
                yield numerator, denominator

        for i in range(17):
            max_denominator = 1 << i
            slope_limited = slope_frac.limit_denominator(max_denominator)
            yield from try_slope(slope_limited)
            yield from try_slope(Fraction(int(slope * max_denominator),
                                          max_denominator))
            # If slope_limited is already very close to the real slope,
            # don't try more
            if abs(float(slope_limited) - float(slope_frac)) * num <= 1:
                break

    def _overhead(self, expr, *, id=id):
        """Estimate the overhead of an expression.
        We use the total number of bytes in tables plus additional overheads
        for each Expr instance.
        """
        total_bytes = 0
        extra = 0

        q = [expr]
        visited = set()
        visited_add = visited.add

        pop = q.pop
        extend = q.extend

        while q:
            x = pop()
            idx = id(x)
            if idx in visited:
                continue
            visited_add(idx)

            extra += x.overhead
            total_bytes += x.static_bytes

            extend(x.children)

        return total_bytes + extra * self._opt.overhead_multiply
