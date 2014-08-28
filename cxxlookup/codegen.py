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

import string

import numpy as np

from .expr import *
from . import groupify
from . import utils


COMMON_HEADERS = r'''#include <stdint.h>
#include <inttypes.h>

#ifndef UINT64_C
# define UINT64_C(x) uint64_t(x##ULL)
#endif
'''


def make_code(base, values, hole, opt):
    groups = groupify.groupify(base, values, hole, opt)

    res = []
    codes = {}  # lo: (hi, code)

    for lo, values in sorted(groups.items()):
        range_name = 'X_{:x}_{:x}'.format(lo, lo + values.size)
        expr, subexprs = MakeCodeForRange(
            lo, values, range_name, opt).make_code()

        code, static = _format_code(expr, subexprs)
        codes[lo] = lo + values.size, code
        res.extend(filter(None, static))

    hole_code = _format_code(Const(U32, hole), None)[0]

    # Create a reverse map from code to range (For sharing code between case's)
    rcode = {}
    for lo, (hi, code) in sorted(codes.items()):
        if code != hole_code:
            rcode.setdefault(code, []).append((lo, hi))

    res.append('inline uint32_t lookup(uint32_t c) noexcept {\n'
               '\tuint64_t cl = c;\n'
               '\t(void)cl;  /* Suppress warning if cl is never used */\n'
               '\tswitch (c) {\n')
    for lo, (hi, code) in sorted(codes.items()):
        if rcode[code][0][0] != lo:  # Printed at other case's
            continue
        for Lo, Hi in rcode[code]:
            if Hi - Lo == 1:
                res.append('\t\tcase {:#x}:\n'.format(Lo))
            else:
                res.append('\t\tcase {:#x} ... {:#x}:\n'.format(Lo, Hi-1))
        res.append(code)
    res.append('\t\tdefault:\n')
    res.append(hole_code)
    res.append('\t}\n')
    res.append('}\n')

    return ''.join(res)


def _format_code(expr, subexprs):

    added_var = 0
    var_output_order = []

    def add_var_in_expr(expr):
        nonlocal added_var

        # Depth-first search
        for x in expr.walk_tempvar():
            var_id = x._var
            var_id_mask = 1 << var_id
            if not (added_var & var_id_mask):
                var_expr = subexprs[var_id]
                add_var_in_expr(var_expr)
                var_output_order.append(var_id)
                added_var |= var_id_mask

    add_var_in_expr(expr)

    if not added_var:
        # No temporary variable
        main_str = '\t\t\treturn {};\n'.format(utils.trim_brackets(str(expr)))
        main_statics = expr.statics()
        return main_str, (main_statics,)

    else:
        # Rename temporary variables
        rename_map = {var: k for (k, var) in enumerate(var_output_order)}

        def rename_var(expr):
            for x in expr.walk_tempvar():
                if not getattr(x, '_var_renamed', False):
                    x._var_renamed = True
                    x._var = rename_map[x._var]

        rename_var(expr)

        code_str = ['\t\t{\n']
        statics = []

        for (k, var) in enumerate(var_output_order):
            var_expr = subexprs[var]
            rename_var(var_expr)
            var_type = var_expr.rettype()
            type_name = TypeNames[var_type]
            while var_expr.IS_CAST and var_expr._type >= var_type:
                var_expr = var_expr._value

            code_str.append('\t\t\t{} {} = {};\n'.format(
                type_name, ExprTempVar.get_name(k),
                utils.trim_brackets(str(var_expr))))
            statics.append(var_expr.statics())

        main_str = '\t\t\treturn {};\n'.format(utils.trim_brackets(str(expr)))
        code_str.append(main_str)
        code_str.append('\t\t}\n')

        statics.append(expr.statics())
        statics.sort()

        return ''.join(code_str), statics


class MakeCodeForRange:
    def __init__(self, lo, values, table_name, opt):
        self._lo = lo
        self._values = values
        self._table_name = table_name
        self._opt = opt

        self._expr = None
        self._subexprs = []

    def make_code(self):
        if self._expr:
            return self._expr, self._subexprs
        expr = self._make_code(
            self._lo, self._values, self._table_name,
            FixedVar(U32, 'c'), FixedVar(U64, 'cl'))

        # Find reachable subexpressions
        reachable = 0
        to_visit = [expr]
        while to_visit:
            x = to_visit.pop()
            for subexpr in x.walk_tempvar():
                subexpr_var_id = subexpr._var
                mask = 1 << subexpr_var_id
                if not (reachable & mask):
                    reachable |= mask
                    to_visit.append(self._subexprs[subexpr_var_id])

        # Extract complicated expressions
        # as variables for code readability
        expr.replace_complicated_subexpressions(8, self._make_subexpr)
        for i, subexpr in enumerate(self._subexprs):
            if reachable & (1 << i):
                subexpr.replace_complicated_subexpressions(
                    8, self._make_subexpr)

        # Final optimization: Remove unnecessary explicit cast
        while expr.IS_CAST and expr._type >= I32:
            expr = expr._value

        self._expr = expr
        return expr, self._subexprs

    def _make_subexpr(self, expr):
        ind = len(self._subexprs)
        self._subexprs.append(expr)
        return TempVar(expr.rettype(), ind)

    def _make_code(self, lo, values, table_name, inexpr, inexpr_long,
                   addition=0, **kwargs):
        exprs = [expr.optimize() for expr in
                 self._yield_code(lo, values, table_name, inexpr,
                                  inexpr_long, addition, **kwargs)]
        return min(exprs, key=self._overhead)

    def _yield_code(self, lo, values, table_name, inexpr, inexpr_long,
                    addition=0,
                    uniqs=None, skip_gcd_reduce=False,
                    skip_almost_linear_reduce=False,
                    skip_compress_4bits=False):
        """
        Everything in values must be positive; addition may be negative.
        The final result has type uint32_t even if it may be negative.

        Yield:
            instances of some subclass of Expr
        """
        if values.dtype != np.uint32:
            values = np.array(values, np.uint32)

        num = values.size
        hi = lo + num

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

        uniq = uniqs.size
        maxv = int(uniqs[-1])
        minv = 0

        maxv_bits = maxv.bit_length()

        assert(uniq > 0)

        if inexpr is inexpr_long:
            inexpr = inexpr_long = inexpr.optimize()
        else:
            inexpr = inexpr.optimize()
            inexpr_long = inexpr_long.optimize()

        # Constant
        if uniq == 1:
            yield Const(U32, addition)
            return

        # [0,1] => [c1,c2]. Better to use ExprCond than linear
        if (lo == 0) and (num == 2):
            c0, c1 = map(int, values + addition)
            if c1 == c0 + 1:
                if c0 == 0:
                    yield inexpr
                else:
                    yield Add(inexpr, Const(U32, c0))
            elif c0 == 0 and (c1 & (c1 - 1)) == 0:
                yield LShift(inexpr, Const(U32, c1.bit_length() - 1))
            else:
                yield Cond(inexpr, c1, c0)
            return

        # Linear
        if (uniq == num) and utils.is_linear(values):
            slope = int(values[1]) - int(values[0])
            # (c - lo) * slope + addition
            expr = Add(inexpr, -lo)
            if slope != 1:
                expr = Mul(expr, slope)
            add = int(values[0]) + addition
            if add:
                expr = Add(expr, add)
            yield expr
            return

        # Not linear, but only two distinct values.
        if uniq == 2:
            k = utils.const_range(values)
            rk = utils.const_range(values[::-1])
            if k + rk == num:  # [a,...,a,b,...,b]
                if k == 1:
                    yield Cond(Compare(inexpr, '==', lo),
                               values[0] + addition,
                               values[-1] + addition)
                elif rk == 1:
                    yield Cond(Compare(inexpr, '==', lo + k),
                               values[-1] + addition,
                               values[0] + addition)
                else:
                    yield Cond(Compare(inexpr, '<', lo + k),
                               values[0] + addition,
                               values[-1] + addition)
                return

            elif values[0] == values[-1] and utils.is_const(values[k:num-rk]):
                # [a, ..., a, b, ..., b, a, ..., a]
                bcount = num - rk - k
                if bcount == 1:
                    cond = Compare(inexpr, '==', lo + k)
                else:
                    subinexpr = Cast(U32, Add(inexpr, -lo - k))
                    cond = Compare(subinexpr, '<', bcount)
                yield Cond(cond, values[k] + addition, values[0] + addition)
                return

            elif num <= 64:
                # Use bit test.
                mask = 0
                value0, value1 = uniqs
                for k in (values == value1).nonzero()[0]:
                    mask |= 1 << int(k)

                Bits = U64 if num > 32 else U32
                expr = And(RShift(Const(Bits, mask), Add(inexpr, -lo)),
                           Const(U32, 1))
                if (value1 + addition != 1) or (value0 + addition != 0):
                    expr = Cond(expr, value1 + addition, value0 + addition)
                elif Bits > U32:
                    expr = Cast(U32, expr)
                yield expr
                return

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
            for BT in 4, 8, 16, 32:
                # +1: Multiply by (1 + power of 2) isn't slow.
                if (BT//2 + 1 < bits < BT) and (num * BT <= Bits):
                    bits = BT
                    break
            mask = 0
            for k, v in enumerate(values):
                mask |= (int(v) + offset) << (k * bits)
            expr = Mul(Add(inexpr, -lo), bits)
            expr = And(Cast(U32, RShift(Const(Bits, mask), expr)),
                       (1 << bits) - 1)
            yield Add(expr, addition - offset)
            return

        # Most elements are almost linear, but a few outliers exist.
        if not skip_almost_linear_reduce:
            slope, slope_count = utils.most_common_element_count(
                np.array(values[1:], np.int64) -
                np.array(values[:-1], np.int64))
            if slope and slope_count * 2 >= num:
                reduced_values = values - slope * (
                    lo + np.arange(num, dtype=np.int64))
                # Be careful to avoid infinite recursion
                reduced_uniqs = utils.np_unique(reduced_values)
                if reduced_uniqs.size <= uniq // 2 or \
                        int(reduced_uniqs[-1] - reduced_uniqs[0]).bit_length()\
                        <= maxv_bits // 2:
                    # utils.np_min(reduced_values)
                    offset = int(reduced_uniqs[0])
                    # Negative values may cause problems
                    reduced_values -= offset
                    reduced_values = np.array(reduced_values, np.uint32)
                    reduced_uniqs -= offset
                    reduced_uniqs = np.array(reduced_uniqs, np.uint32)

                    subexpr, subexpr_long = self._smart_subexpr(inexpr,
                                                                inexpr_long)

                    expr = self._make_code(lo, reduced_values,
                                           table_name + '_reduced',
                                           subexpr, subexpr_long,
                                           addition=offset+addition,
                                           uniqs=reduced_uniqs,
                                           skip_almost_linear_reduce=True)

                    # inexpr * slope + expr
                    yield Add(Mul(subexpr, slope), expr)

        # Two-level lookup?
        if maxv > num > uniq * 4 // 3:
            indices = np.searchsorted(uniqs, values)
            # Level 1
            expr = self._make_code(lo, indices, table_name + '_index',
                                   inexpr, inexpr_long,
                                   uniqs=np.arange(uniq, dtype=np.uint32))
            # Level 2
            expr = self._make_code(0, uniqs, table_name + '_value', expr, expr,
                                   addition=addition,
                                   uniqs=uniqs)

            yield expr

        # Try using "compressed" table. 2->1
        if not skip_compress_4bits and maxv_bits in (3, 4) and num > 16:
            offset = 0
            if (addition > 0) and (addition + maxv < 16):
                offset = addition
            compressed_values = utils.compress_array(values + offset, 2)

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/2] >> (expr%2*4)) & 15
            expr_shift = RShift(expr, 1)
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_4bits',
                                        expr_shift, expr_shift)
            expr_right = LShift(And(expr, 1), 2)
            expr = And(RShift(expr_left, expr_right), 15)
            expr = Add(expr, addition - offset)
            yield expr
            return

        # Try using "compressed" table. 4->1
        if maxv_bits == 2 and num > 32:
            compressed_values = utils.compress_array(values, 4)

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/4] >> (expr%4*2)) & 3
            expr_shift = RShift(expr, 2)
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_2bits',
                                        expr_shift, expr_shift)
            expr_right = LShift(And(expr, 3), 1)
            expr = And(RShift(expr_left, expr_right), 3)
            expr = Add(expr, addition)
            yield expr
            return

        # Try using "bitvector". 8->1
        if (maxv == 1) and num > 64:
            compressed_values = utils.compress_array(values, 8)

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/8] >> (expr%8)) & 1
            expr_shift = RShift(expr, 3)
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_bitvec',
                                        expr_shift, expr_shift)
            expr_right = And(expr, 7)
            expr = And(RShift(expr_left, expr_right), 1)
            expr = Add(expr, addition)
            yield expr
            return

        # GCD may help
        if not skip_gcd_reduce:
            gcd = utils.gcd_reduce(values)
            if gcd > 1:
                offset = int(values[0]) % gcd
                reduced_values = values // gcd
                expr = self._make_code(lo, reduced_values, table_name + '_gcd',
                                       inexpr, inexpr_long,
                                       skip_gcd_reduce=True)
                yield Add(Mul(expr, gcd), addition + offset)

        # Try splitting the data into low and high parts
        for k in (16, 8, 4):
            if not k < maxv_bits <= 2 * k:
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
                continue

            hi_gcd = utils.gcd_many(hi_values)
            hi_values //= hi_gcd

            if k == 4:
                # We must be very careful when k == 4
                # If we just split it, two 4-bit values will be joined
                # in the recursion, and then split again, resulting in almost
                # uncontrollable recursions
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

            subexpr, subexpr_long = self._smart_subexpr(inexpr, inexpr_long)

            lo_expr = self._make_code(
                lo, lo_values, '{}_{}lo'.format(table_name, k),
                subexpr, subexpr_long, addition=addition,
                uniqs=lo_uniqs, skip_compress_4bits=(k == 4))
            hi_expr = self._make_code(
                lo, hi_values, '{}_{}hi'.format(table_name, k),
                subexpr, subexpr_long,
                uniqs=hi_uniqs, skip_compress_4bits=(k == 4))
            yield Add(lo_expr, Mul(hi_expr, hi_gcd))

        # Finally fall back to the simplest one-level table
        table_type = const_type(maxv)
        if addition > 0 and table_type == const_type(maxv + addition):
            expr = Table(table_type, table_name, values + addition,
                         inexpr_long, lo)
        else:
            expr = Table(table_type, table_name, values, inexpr_long, lo)
            if addition != 0:
                expr = Add(expr, addition)
        yield expr

    def _smart_subexpr(self, expr, expr_long):
        if self._very_simple(expr):
            return expr, expr_long
        else:
            subexpr = self._make_subexpr(expr)
            return subexpr, subexpr

    @staticmethod
    def _very_simple(expr):
        # Var
        if expr.IS_VAR:
            return True
        # Var + const
        if expr.IS_ADD and \
                len(expr._exprs) == 1 and \
                expr._exprs[0].IS_VAR:
            return True
        # Var >> const, Var << const
        if expr.IS_SHIFT and \
                expr._left.IS_VAR and \
                expr._right.IS_CONST:
            return True
        # (Var - const) >> const
        if expr.IS_RSHIFT and \
                expr._left.IS_ADD and \
                len(expr._left._exprs) == 1 and \
                expr._left._exprs[0].IS_VAR and \
                expr._right.IS_CONST:
            return True
        return False

    def _overhead(self, expr):
        """Estimate the overhead of an expression.
        We use the total number of bytes in tables plus additional overheads
        for each Expr instance.
        """
        total_bytes = 0
        extra = 0

        visited_subexprs = 0
        to_scan_list = [expr]
        while to_scan_list:
            expr = to_scan_list.pop()

            for x in expr.walk():
                if x.IS_VAR:
                    if x.IS_TEMPVAR:
                        var_id = x._var
                        mask = 1 << var_id
                        if not (visited_subexprs & mask):
                            visited_subexprs |= mask
                            to_scan_list.append(self._subexprs[var_id])
                elif x.IS_TABLE:
                    total_bytes += x.table_bytes()
                    extra += 2
                elif x.IS_CAST:
                    pass
                elif x.IS_COND:
                    extra += 3
                else:
                    extra += 2

        return total_bytes + extra * self._opt.overhead_multiply // 2


WRAP_TEMPLATE = string.Template(r'''namespace {
namespace CxxLookup_$func_name {

$code

} // namespace CxxLookup_$func_name
} // namespace

uint32_t $func_name(uint32_t c) noexcept {
    return CxxLookup_$func_name::lookup(c);
}''')


def wrap_code(func_name, code):
    return WRAP_TEMPLATE.substitute(func_name=func_name, code=code)
