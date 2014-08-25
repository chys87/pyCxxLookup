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
            for x in expr.walk_var():
                var_name = x._name
                if var_name not in added_var_name:
                    var_expr = subexprs[var_name]
                    add_var_in_expr(var_expr)

                    var_type = var_expr.rettype()
                    type_name = TypeNames[var_type]
                    while var_expr.IS_CAST and var_expr._type >= var_type:
                        var_expr = var_expr._value

                    subexpr_str.append('\t\t\t{} {} = {};\n'.format(
                        type_name, var_name,
                        utils.trim_brackets(str(var_expr))))
                    statics.append(var_expr.statics())
                    added_var_name.add(var_name)

        add_var_in_expr(expr)
        statics.append(main_statics)
        statics.sort()

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

        # Remove unreachable subexpressions
        reachable = {'c', 'cl'}
        to_visit = [expr]
        while to_visit:
            x = to_visit.pop()
            for subexpr in x.walk_var():
                if subexpr._name not in reachable:
                    reachable.add(subexpr._name)
                    to_visit.append(self._named_subexprs[subexpr._name])

        for name in list(self._named_subexprs):
            if name not in reachable:
                del self._named_subexprs[name]

        # Extract complicated expressions
        # as variables for code readability
        expr.replace_complicated_subexpressions(8, self._make_subexpr)
        # The purpose of the sort is to get deterministic results
        for _, subexpr in sorted(self._named_subexprs.items()):
            subexpr.replace_complicated_subexpressions(8, self._make_subexpr)

        # Final optimization: Remove unnecessary explicit cast
        while expr.IS_CAST and expr._type >= I32:
            expr = expr._value

        self._expr = expr

    def _make_subexpr(self, expr):
        ind = self._subexpr_ind
        self._subexpr_ind = ind + 1

        bits = 1
        expressible = 26

        while ind >= expressible:
            bits += 1
            ind -= expressible
            expressible *= 26

        s = ''
        for _ in range(bits):
            s = chr(ord('A') + (ind % 26)) + s
            ind //= 26

        self._named_subexprs[s] = expr
        return Var(expr.rettype(), s)

    def _make_code(self, lo, values, table_name, inexpr, inexpr_long,
                   addition=0):
        exprs = [expr.optimize() for expr in
                 self._yield_code(lo, values, table_name, inexpr,
                                  inexpr_long, addition)]
        return min(exprs, key=self._overhead)

    def _yield_code(self, lo, values, table_name, inexpr, inexpr_long,
                    addition=0):
        """
        Everything in values must be positive; addition may be negative.
        The final result has type uint32_t even if it may be negative.

        Yield:
            instances of some child class of Expr
        """
        if values.dtype != np.uint32:
            values = np.array(values, np.uint32)

        minv = utils.np_min(values)
        if minv != 0:
            values = values - minv
            addition += minv
            minv = 0

        num = values.size
        hi = lo + num
        uniqs = utils.np_unique(values)
        uniq = uniqs.size
        maxv = int(uniqs[-1])
        minv = 0

        maxv_bits = maxv.bit_length()

        assert(uniq > 0)

        inexpr = inexpr.optimize()
        inexpr_long = inexpr_long.optimize()

        # Constant
        if uniq == 1:
            yield Const(U32, addition)
            return

        # [0,1] => [c1,c2]. Better to use ExprCond than linear
        if (lo == 0) and (num == 2):
            c0, c1 = values + addition
            if c1 == c0 + 1:
                if c0 == 0:
                    yield inexpr
                else:
                    yield Add(inexpr, Const(U32, c0))
            elif c0 == 0 and (c1 & (c1 - 1)) == 0:
                yield LShift(inexpr, Const(U32, int(c1).bit_length() - 1))
            else:
                yield Cond(inexpr, c1, c0)
            return

        # Linear
        if (uniq == num) and utils.is_linear(values):
            slope = int(values[1]) - int(values[0])
            # (c - lo) * slope + addition
            yield Add(Mul(Add(inexpr, -lo), slope), int(values[0]) + addition)
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
        slope, slope_count = map(int, utils.most_common_element_count(
            np.array(values[1:], np.int64) - np.array(values[:-1], np.int64)))
        if slope and slope_count >= num // 2:
            reduced_values = values - slope * (
                lo + np.arange(num, dtype=np.int64))
            # Be careful to avoid infinite recursion
            reduced_uniqs = utils.np_unique(reduced_values)
            if reduced_uniqs.size <= uniq // 2 or \
                    int(reduced_uniqs[-1] - reduced_uniqs[0]).bit_length() <= \
                    maxv_bits // 2:
                offset = int(reduced_uniqs[0])  # utils.np_min(reduced_values)
                # Negative values may cause problems
                reduced_values -= offset
                reduced_values = np.array(reduced_values, np.uint32)

                subexpr, subexpr_long = self._smart_subexpr(inexpr,
                                                            inexpr_long)

                expr = self._make_code(lo, reduced_values,
                                       table_name + '_reduced',
                                       subexpr, subexpr_long,
                                       addition=offset+addition)

                # inexpr * slope + expr
                yield Add(Mul(subexpr, slope), expr)

        # Two-level lookup?
        if maxv > num > uniq * 4 // 3:
            indices = np.searchsorted(uniqs, values)
            # Level 1
            expr = self._make_code(lo, indices, table_name + '_index',
                                   inexpr, inexpr_long)
            # Level 2
            expr = self._make_code(0, uniqs, table_name + '_value', expr, expr,
                                   addition=addition)

            yield expr

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
            yield expr
            return

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
            yield expr
            return

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
            yield expr
            return

        # GCD may help
        gcd = utils.gcd_reduce(values)
        if gcd > 1:
            offset = int(values[0]) % gcd
            reduced_values = values // gcd
            expr = self._make_code(lo, reduced_values, table_name + '_gcd',
                                   inexpr, inexpr_long)
            yield Add(Mul(expr, gcd), addition + offset)

        # Try splitting the data into low and high parts
        for k in (16, 8, 4):
            if not k < maxv_bits <= 2 * k:
                continue

            lomask = np.uint32((1 << k) - 1)
            lo_values = values & lomask
            hi_values = values & ~lomask

            lo_uniq = utils.np_unique(lo_values).size
            if lo_uniq < 2:
                continue
            hi_uniq = utils.np_unique(hi_values).size
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

            subexpr, subexpr_long = self._smart_subexpr(inexpr, inexpr_long)

            lo_expr = self._make_code(
                lo, lo_values, '{}_{}lo'.format(table_name, k),
                subexpr, subexpr_long, addition=addition)
            hi_expr = self._make_code(
                lo, hi_values, '{}_{}hi'.format(table_name, k),
                subexpr, subexpr_long)
            yield Add(lo_expr, Mul(hi_expr, hi_gcd))

        # Finally fall back to the simplest one-level table
        if addition > 0 and const_type(maxv) == const_type(maxv + addition):
            expr = Table(table_name, values + addition, inexpr_long, lo)
        else:
            expr = Table(table_name, values, inexpr_long, lo)
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

        visited_subexprs = {'c', 'cl'}
        to_scan_list = [expr]
        while to_scan_list:
            expr = to_scan_list.pop()

            for x in expr.walk():
                extra += 2
                if x.IS_VAR:
                    var_name = x._name
                    if var_name not in visited_subexprs:
                        visited_subexprs.add(var_name)
                        to_scan_list.append(self._named_subexprs[var_name])
                    extra -= 2
                elif x.IS_TABLE:
                    total_bytes += x.table_bytes()
                elif x.IS_CAST:
                    extra -= 2
                elif x.IS_COND:
                    extra += 1

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
