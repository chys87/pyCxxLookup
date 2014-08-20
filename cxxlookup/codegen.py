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

    def format_code(expr, subexprs):
        main_str = '\t\t\treturn {};\n'.format(utils.trim_brackets(str(expr)))
        main_statics = expr.statics()

        added_var_name = {'c', 'cl'}
        subexpr_str = []
        statics = []

        def add_var_in_expr(expr):
            # Depth-first search
            for x in expr.walk(ExprVar):
                var_name = x._name
                if var_name not in added_var_name:
                    var_expr = subexprs[var_name]
                    add_var_in_expr(var_expr)
                    subexpr_str.append(
                        '\t\t\t{} {} = {};\n'.format(
                            TypeNames[var_expr.rettype()],
                            var_name,
                            utils.trim_brackets(str(var_expr))))
                    statics.append(var_expr.statics())
                    added_var_name.add(var_name)

        add_var_in_expr(expr)
        statics.append(main_statics)

        if not subexpr_str:
            code = main_str
        else:
            code = '\t\t{\n'
            code += ''.join(subexpr_str)
            code += main_str
            code += '\t\t}\n'
        return code, statics

    for lo, values in sorted(groups.items()):
        range_name = 'X_{:x}_{:x}'.format(lo, lo + values.size)
        expr, subexprs = MakeCodeForRange.make(lo, values, range_name, opt)

        code, static = format_code(expr, subexprs)
        codes[lo] = lo + values.size, code
        res.extend(filter(None, static))

    hole_code = format_code(Const(U32, hole), None)[0]

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


class MakeCodeForRange:
    def __init__(self, lo, values, table_name, opt):
        self._lo = lo
        self._values = values
        self._table_name = table_name
        self._opt = opt

        self._expr = None
        self._subexpr_ind = 0
        self._named_subexprs = {}

    @staticmethod
    def make(lo, values, table_name, opt):
        obj = MakeCodeForRange(lo, values, table_name, opt)
        obj.make_code()
        return obj._expr, obj._named_subexprs

    def make_code(self):
        if self._expr:
            return
        expr = self._make_code(
            self._lo, self._values, self._table_name,
            Var(U32, 'c'), Var(U64, 'cl'))
        expr = expr.optimize()

        # Final optimization: Remove unnecessary explicit cast
        while isinstance(expr, ExprCast) and expr._type >= I32:
            expr = expr._value

        self._expr = expr

    def _make_subexpr(self, expr):
        ind = self._subexpr_ind
        self._subexpr_ind = ind + 1

        s = ''
        while ind >= 26:
            s = chr(ord('A') + (ind % 26)) + s
            ind //= 26
        s = chr(ord('A') + ind) + s
        self._named_subexprs[s] = expr
        return Var(expr.rettype(), s)

    def _make_code(self, lo, values, table_name, inexpr, inexpr_long,
                   addition=0):
        """
        Everything in values must be positive; addition may be negative.
        The final result has type uint32_t even if it may be negative.

        Return:
            A instance of some child class of Expr
        """
        minv = int(values.min())
        if minv != 0:
            values = values - minv
            addition += minv
            minv = 0

        num = values.size
        hi = lo + num
        uniqs = np.unique(values)
        uniq = uniqs.size
        maxv = int(uniqs[-1])
        minv = 0

        maxv_bits = maxv.bit_length()

        assert(uniq > 0)

        inexpr = inexpr.optimize()
        inexpr_long = inexpr_long.optimize()

        # Constant
        if uniq == 1:
            return Const(U32, addition)

        # [0,1] => [c1,c2]. Better to use ExprCond than linear
        if (lo == 0) and (num == 2):
            c0, c1 = values + addition
            if c1 == c0 + 1:
                if c0 == 0:
                    return inexpr
                else:
                    return Add(inexpr, Const(U32, c0))
            elif c0 == 0 and (c1 & (c1 - 1)) == 0:
                return LShift(inexpr, Const(U32, int(c1).bit_length() - 1))
            else:
                return Cond(inexpr, c1, c0)

        # Linear
        if (uniq == num) and utils.is_linear(values):
            slope = int(values[1]) - int(values[0])
            # (c - lo) * slope + addition
            return Add(Mul(Add(inexpr, -lo), slope), int(values[0]) + addition)

        # Not linear, but only two distinct values.
        if uniq == 2:
            k = utils.const_range(values)
            rk = utils.const_range(values[::-1])
            if k + rk == num:  # [a,...,a,b,...,b]
                if k == 1:
                    return Cond(Compare(inexpr, '==', lo),
                                values[0] + addition,
                                values[-1] + addition)
                elif rk == 1:
                    return Cond(Compare(inexpr, '==', lo + k),
                                values[-1] + addition,
                                values[0] + addition)
                else:
                    return Cond(Compare(inexpr, '<', lo + k),
                                values[0] + addition,
                                values[-1] + addition)

            elif values[0] == values[-1] and utils.is_const(values[k:num-rk]):
                # [a, ..., a, b, ..., b, a, ..., a]
                bcount = num - rk - k
                if bcount == 1:
                    cond = Compare(inexpr, '==', lo + k)
                else:
                    subinexpr = Cast(U32, Add(inexpr, -lo - k))
                    cond = Compare(subinexpr, '<', bcount)
                return Cond(cond, values[k] + addition, values[0] + addition)

            elif num <= 64:
                # Use bit test.
                mask = 0
                value0, value1 = uniqs
                for k in range(num):
                    if values[k] == value1:
                        mask |= 1 << k

                Bits = U64 if num > 32 else U32
                expr = And(RShift(Const(Bits, mask), Add(inexpr, -lo)),
                           Const(U32, 1))
                if (value1 + addition != 1) or (value0 + addition != 0):
                    expr = Cond(expr, value1 + addition, value0 + addition)
                elif Bits > U32:
                    expr = Cast(U32, expr)
                return expr

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
            return Add(expr, addition - offset)

        # Most elements are almost linear, but a few outliers exist.
        slope, slope_count = map(int, utils.most_common_element_count(
            np.array(values[1:], np.int64) - np.array(values[:-1], np.int64)))
        if slope and slope_count >= num // 2:
            reduced_values = values - slope * (
                lo + np.arange(num, dtype=np.int64))
            # Be careful to avoid infinite recursion
            reduced_uniqs = np.unique(reduced_values)
            if reduced_uniqs.size <= uniq // 2 or \
                    int(reduced_uniqs[-1] - reduced_uniqs[0]).bit_length() <= \
                    maxv_bits // 2:
                offset = int(reduced_values.min())
                # Negative values may cause problems
                reduced_values -= offset
                reduced_values = np.array(reduced_values, np.uint32)

                subexpr, subexpr_long = self._smart_subexpr(inexpr,
                                                            inexpr_long)

                expr = self._make_code(lo, reduced_values,
                                       table_name + '_reduced',
                                       subexpr, subexpr_long,
                                       addition=offset+addition)

                if (self._table_size(expr) <
                        num * TypeBytes[const_type(maxv)] - 16):
                    # inexpr * slope + expr
                    return Add(Mul(subexpr, slope), expr)

        # Two-level lookup?
        if maxv > num > uniq * 4 // 3:
            indices = np.searchsorted(uniqs, values)
            # Level 1
            expr = self._make_code(lo, indices, table_name + '_index',
                                   inexpr, inexpr_long)
            # Level 2
            expr = self._make_code(0, uniqs, table_name + '_value', expr, expr,
                                   addition=addition)

            if (self._table_size(expr) <
                    num * TypeBytes[const_type(maxv)] - 16):
                return expr

        # Try using "compressed" table. 2->1
        if maxv_bits in (3, 4) and num > 16:
            offset = 0
            if (addition > 0) and (addition + maxv < 16):
                offset = addition
            lo_chunk, hi_chunk = utils.stridize(values, 2, -offset)
            compressed_values = (lo_chunk + offset) | \
                ((hi_chunk + offset) << 4)

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/2] >> (expr%2*4)) & 15
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_4bits',
                                        RShift(expr, 1), RShift(expr, 1))
            expr_right = LShift(And(expr, 1), 2)
            expr = And(RShift(expr_left, expr_right), 15)
            expr = Add(expr, addition - offset)
            return expr

        # Try using "compressed" table. 4->1
        if maxv_bits == 2 and num > 32:
            chunk_a, chunk_b, chunk_c, chunk_d = utils.stridize(values, 4, 0)
            compressed_values = chunk_a | (chunk_b << 2) | \
                (chunk_c << 4) | (chunk_d << 6)

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/4] >> (expr%4*2)) & 3
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_2bits',
                                        RShift(expr, 2), RShift(expr, 2))
            expr_right = LShift(And(expr, 3), 1)
            expr = And(RShift(expr_left, expr_right), 3)
            expr = Add(expr, addition)
            return expr

        # Try using "bitvector". 8->1
        if (maxv == 1) and num > 64:
            chunks_v = utils.stridize(values, 8, 0)
            compressed_values = sum(v << i for (i, v) in enumerate(chunks_v))

            subexpr, _ = self._smart_subexpr(inexpr, inexpr_long)

            expr = Add(subexpr, -lo)
            # (table[expr/8] >> (expr%8)) & 1
            expr_left = self._make_code(0, compressed_values,
                                        table_name + '_bitvec',
                                        RShift(expr, 3), RShift(expr, 3))
            expr_right = And(expr, 7)
            expr = And(RShift(expr_left, expr_right), 1)
            expr = Add(expr, addition)
            return expr

        # GCD may help
        gcd = utils.gcd_reduce(values)
        if gcd > 1:
            offset = int(values[0]) % gcd
            reduced_values = values // gcd
            expr = self._make_code(lo, reduced_values, table_name + '_gcd',
                                   inexpr, inexpr_long)
            if (self._table_size(expr) <
                    num * TypeBytes[const_type(maxv)] - 16):
                return Add(Mul(expr, gcd), addition + offset)

        # Try splitting the data into low and high parts
        for k in (16, 8, 4):
            if not k < maxv_bits <= 2 * k:
                continue

            lomask = (1 << k) - 1
            lo_values = values & lomask
            hi_values = values & ~lomask

            lo_uniq = np.unique(lo_values).size
            if lo_uniq < 2:
                continue
            hi_uniq = np.unique(hi_values).size
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

            elif int(hi_values.max() - hi_values.min()).bit_length() <= k // 2:
                pass

            else:
                # If none of the conditions meet, we consider the split not
                # helpful.  We cannot just try out.  That'd be too slow.
                continue

            subexpr, subexpr_long = self._smart_subexpr(inexpr, inexpr_long)

            lo_expr = self._make_code(
                lo, lo_values, '{}_{}lo'.format(table_name, k),
                subexpr, subexpr_long, addition=addition)
            hi_expr = self._make_code(
                lo, hi_values, '{}_{}hi'.format(table_name, k),
                subexpr, subexpr_long)
            table_size = self._table_size(lo_expr) + \
                self._table_size(hi_expr)
            if table_size < num * TypeBytes[const_type(maxv)] - 16:
                return Add(lo_expr, Mul(hi_expr, hi_gcd))

        # Finally fall back to the simplest one-level table
        if addition > 0 and const_type(maxv) == const_type(maxv + addition):
            expr = Table(table_name, values + addition, inexpr_long, lo)
        else:
            expr = Table(table_name, values, inexpr_long, lo)
            if addition != 0:
                expr = Add(expr, addition)
        return expr

    def _smart_subexpr(self, expr, expr_long):
        if self._very_simple(expr):
            return expr, expr_long
        else:
            subexpr = self._make_subexpr(expr)
            return subexpr, subexpr

    @staticmethod
    def _very_simple(expr):
        # Var
        if isinstance(expr, ExprVar):
            return True
        # Var + const
        if isinstance(expr, ExprAdd) and \
                len(expr._exprs) == 1 and \
                isinstance(expr._exprs[0], ExprVar) and \
                isinstance(expr._const, ExprConst):
            return True
        # Var >> const, Var << const
        if isinstance(expr, (ExprRShift, ExprLShift)) and \
                isinstance(expr._left, ExprVar) and \
                isinstance(expr._right, ExprConst):
            return True
        # (Var - const) >> const
        if isinstance(expr, ExprRShift) and \
                isinstance(expr._left, ExprAdd) and \
                len(expr._left._exprs) == 1 and \
                isinstance(expr._left._exprs[0], ExprVar) and \
                isinstance(expr._right, ExprConst):
            return True
        return False

    def _table_size(self, expr):
        """Get the total number of bytes tables in the expression.
        This is useful to determine whether an optimization is effective.
        """
        total_bytes = 0

        visited_subexprs = {'c', 'cl'}
        to_scan_list = [expr]
        while to_scan_list:
            expr = to_scan_list.pop()
            for x in expr.walk(ExprVar):
                var_name = x._name
                if var_name not in visited_subexprs:
                    visited_subexprs.add(var_name)
                    to_scan_list.append(self._named_subexprs[var_name])
            for x in expr.walk(ExprTable):
                total_bytes += x.table_bytes()

        return total_bytes


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
