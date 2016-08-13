#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

# Copyright (c) 2014, 2016, chys <admin@CHYS.INFO>
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

try:
    from . import _speedups
except ImportError:
    _speedups = None


# Signed types only allowed for intermediate values
# Unsigned: Number of bits
# Signed: Number of bits - 1
# E.g.: 31 = int32_t; 32 = uint32_t
def type_name(type):
    '''
    >>> type_name(7), type_name(32)
    ('int8_t', 'uint32_t')
    '''
    if (type & 1):
        return 'int{}_t'.format(type + 1)
    else:
        return 'uint{}_t'.format(type)


def type_bytes(type):
    '''
    >>> list(map(type_bytes, [7, 8, 15, 16, 31, 32, 63, 64]))
    [1, 1, 2, 2, 4, 4, 8, 8]
    '''
    return (type + 7) // 8


def const_type(value):
    if value >= 2**16:
        if value >= 2**32:
            return 64
        else:
            return 32
    elif value >= 2**8:
        return 16
    else:
        return 8


class ExprMeta(type):
    """This is for performance purpose.

    Add IS_*** constants to Expr* classes to replace ininstance,
    which turned out to be one of the bottlenecks of pyCxxLookup

    >>> Expr.IS_CONST, Expr.IS_VAR, Expr.IS_RSHIFT
    (False, False, False)
    >>> ExprConst.IS_CONST, ExprConst.IS_VAR, ExprConst.IS_RSHIFT
    (True, False, False)
    >>> ExprVar.IS_CONST, ExprVar.IS_VAR, ExprVar.IS_RSHIFT
    (False, True, False)
    >>> ExprRShift.IS_CONST, ExprRShift.IS_VAR, ExprRShift.IS_RSHIFT
    (False, False, True)
    """
    def __new__(cls, name, bases, namespace, **kwds):
        result = type.__new__(cls, name, bases, dict(namespace))
        if name != 'Expr' and name.startswith('Expr'):
            is_name = 'IS_' + name[4:].upper()
            setattr(result, is_name, True)
            setattr(Expr, is_name, False)

        for extra_name in result.__dict__.get('IS_ALSO', ()):
            is_name = 'IS_' + extra_name
            setattr(result, is_name, True)
            setattr(Expr, is_name, False)

        return result


class Expr(metaclass=ExprMeta):
    __slots__ = ()

    def __str__(self):
        raise NotImplementedError

    def statics(self, vs):
        return ''.join(filter(None, (x.statics(vs) for x in self.children)))

    children = ()
    rtype = None

    @property
    def optimized(self):
        return self

    def walk(self):
        """Recursively visit itself and all children."""
        yield self
        q = [self]
        q_pop = q.pop
        q_extend = q.extend
        while q:
            expr = q_pop()
            children = expr.children
            yield from children
            q_extend(children)

    def walk_tempvar(self):
        """Shortcut for filter(lambda x: x.IS_TEMPVAR, self.walk())
        """
        return (x for x in self.walk() if x.IS_TEMPVAR)

    def _complicated(self, threshold):
        for expr in self.walk():
            threshold -= 1
            if not threshold:
                return True
        return False

    def extract_subexprs(self, threshold, callback, allow_new):
        for subexpr in self.children:
            subexpr.extract_subexprs(threshold, callback, allow_new)

    def __add__(self, r):
        return Add(self, r)

    def __mul__(self, r):
        return ExprMul(self, exprize(r))

    def __floordiv__(self, r):
        return ExprDiv(self, exprize(r))

    def __rfloordiv__(self, r):
        return ExprDiv(exprize(r), self)

    def __mod__(self, r):
        return ExprMod(self, exprize(r))

    def __rmod__(self, r):
        return ExprMod(exprize(r), self)

    def __sub__(self, r):
        r = exprize(r)
        if not r.IS_CONST:
            return NotImplemented
        return Add(self, -r)

    def __and__(self, r):
        return ExprAnd(self, exprize(r))

    def __lshift__(self, r):
        return ExprLShift(self, exprize(r))

    def __rshift__(self, r):
        return ExprRShift(self, exprize(r))

    def __rlshift__(self, r):
        return ExprLShift(exprize(r), self)

    def __rrshift__(self, r):
        return ExprRShift(exprize(r), self)

    def __eq__(self, r):
        return ExprCompare(self, '==', exprize(r))

    def __ne__(self, r):
        return ExprCompare(self, '!=', exprize(r))

    def __lt__(self, r):
        return ExprCompare(self, '<', exprize(r))

    def __le__(self, r):
        return ExprCompare(self, '<=', exprize(r))

    def __gt__(self, r):
        return ExprCompare(self, '>', exprize(r))

    def __ge__(self, r):
        return ExprCompare(self, '>=', exprize(r))

    __radd__ = __add__
    __rmul__ = __mul__
    __rand__ = __and__


class ExprVar(Expr):
    __slots__ = 'rtype',

    def __init__(self, type):
        self.rtype = type


class ExprFixedVar(ExprVar):
    __slots__ = 'name',

    def __init__(self, type, name):
        super().__init__(type)
        self.name = name

    def __str__(self):
        return self.name


class ExprTempVar(ExprVar):
    __slots__ = 'var',
    _name_cache = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self, type, var):
        super().__init__(type)
        self.var = var

    @classmethod
    def get_name(cls, var):
        '''
        >>> list(map(ExprTempVar.get_name, (0, 26, 52)))
        ['A', 'AA', 'BA']
        '''
        cache = cls._name_cache
        try:
            s = cache[var]
        except IndexError:
            cache += [None] * (var + 1 - len(cache))
        else:
            if s is not None:
                return s

        length = 1
        expressible = 26

        ind = var
        while ind >= expressible:
            length += 1
            ind -= expressible
            expressible *= 26

        s = ''
        for _ in range(length):
            s = chr(ord('A') + (ind % 26)) + s
            ind //= 26
        cache[var] = s
        return s

    def __str__(self):
        return self.get_name(self.var)


class ExprConst(Expr):
    __slots__ = 'rtype', 'value'

    def __init__(self, type, value, *, int=int):
        self.rtype = type
        self.value = int(value)

    def __str__(self):
        if -10 < self.value < 10:
            value_s = str(self.value)
        else:
            value_s = hex(self.value)
        if self.rtype < 64:
            return value_s + 'u'
        else:
            return 'UINT64_C({})'.format(value_s)

    def _complicated(self, threshold):
        # Always assign 64-bit constant to a variable for readability.
        return (self.rtype == 64)

    @staticmethod
    def combine(const_exprs):
        """Combine multiple ExprConst into one."""
        const_value = 0
        const_type = 32
        for expr in const_exprs:
            const_value += expr.value
            const_type = max(const_type, expr.rtype)
        if const_value == 0:
            return None
        else:
            return ExprConst(const_type, const_value)

    def __neg__(self):
        return ExprConst(self.rtype, -self.value)


class ExprAdd(Expr):
    def __init__(self, exprs, const, *, max=max, tuple=tuple):
        assert const is None or const.IS_CONST
        self.exprs = tuple(exprs)
        self.const = const
        rtype = max([x.rtype for x in self.children])
        self.rtype = max(rtype, 31)  # C type-promotion rule

    def __str__(self):
        res = ' + '.join(map(str, self.exprs))
        if self.const:
            const_value = self.const.value
            if const_value >= 0:
                res += ' + ' + str(self.const)
            else:
                res += ' - ' + str(ExprConst(self.const.rtype,
                                             -const_value))
        return '(' + res + ')'

    @property
    def children(self):
        const = self.const
        if const:
            return self.exprs + (const,)
        else:
            return self.exprs

    @utils.cached_property
    def optimized(self):
        exprs = []
        const_exprs = []

        if self.const:
            const_exprs.append(self.const)

        for expr in self.exprs:
            expr = expr.optimized
            if expr.IS_ADD:
                exprs.extend(expr.exprs)
                if expr.const:
                    const_exprs.append(expr.const)
            elif expr.IS_CONST:
                const_exprs.append(expr)
            else:
                exprs.append(expr)

        const = ExprConst.combine(const_exprs)

        # (a ? c1 : c2) + c3 ==> (a ? c1 + c3 : c2 + c3)
        if const:
            const_value = const.value
            for i, expr in enumerate(exprs):
                if expr.IS_COND and \
                        expr.exprT.IS_CONST and expr.exprF.IS_CONST and \
                        (min(expr.exprT.value, expr.exprF.value) + const_value
                         >= 0):
                    expr = ExprCond(
                        expr.cond,
                        ExprConst(self.rtype, expr.exprT.value + const_value),
                        ExprConst(self.rtype, expr.exprF.value + const_value))
                    exprs[i] = expr.optimized
                    const = None
                    break

        self.exprs = exprs = tuple(exprs)
        self.const = const

        if len(exprs) == 1 and not const:
            return exprs[0]

        return self

    def extract_subexprs(self, threshold, callback, allow_new):
        exprs = []
        for expr in self.exprs:
            expr.extract_subexprs(threshold, callback, allow_new)
            expr = callback(expr, allow_new and expr._complicated(threshold))
            exprs.append(expr)
        self.exprs = tuple(exprs)

        if self.const:
            self.const = callback(
                self.const, allow_new and self.const._complicated(threshold))


class ExprBinary(Expr):
    __slots__ = 'left', 'right', 'rtype'

    def __init__(self, left, right, rtype=None, *, max=max):
        self.left = left = left.optimized
        self.right = right = right.optimized
        self.rtype = rtype or max(31, left.rtype, right.rtype)

    @property
    def children(self):
        return self.left, self.right

    def extract_subexprs(self, threshold, callback, allow_new):
        super().extract_subexprs(threshold, callback, allow_new)
        self.left = callback(self.left,
                             allow_new and self.left._complicated(threshold))
        self.right = callback(self.right,
                              allow_new and self.right._complicated(threshold))


class ExprShift(ExprBinary):
    def __init__(self, left, right):
        super().__init__(left, right, max(31, left.rtype))


class ExprLShift(ExprShift):
    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self.right
        if right.IS_CONST:
            if right.value in (1, 2, 3):
                return '{} * {}'.format(self.left, 1 << right.value)
            return '({} << {})'.format(self.left, right.value)
        else:
            return '({} << {})'.format(self.left, right)

    @utils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        if right_const and left.IS_CONST:
            return ExprConst(self.rtype, left.value << right.value)

        # "(a & c1) << c2" ==> (a << c2) & (c1 << c2) (where c2 <= 3)
        # This takes advantage of x86's LEA instruction
        if right_const and right.value <= 3 and \
                left.IS_AND and \
                left.right.IS_CONST:
            expr_left = ExprLShift(left.left, right)
            expr_right = ExprConst(left.right.rtype,
                                   left.right.value << right.value)
            return ExprAnd(expr_left, expr_right).optimized

        # (cond ? c1 : c2) << c3 ==> (cond ? c1 << c3 : c2 << c3)
        if right_const and \
                left.IS_COND and \
                left.exprT.IS_CONST and \
                left.exprF.IS_CONST:
            expr = ExprCond(left.cond,
                            ExprLShift(left.exprT, right),
                            ExprLShift(left.exprF, right))
            return expr.optimized

        # (a >> c1) << c2
        # (a << (c2 - c1)) & ~((1 << c2) - 1)   (c2 > c1)
        # (a >> (c1 - c2)) & ~((1 << c2) - 1)   (c2 <= c1)
        if right_const and \
                left.IS_RSHIFT and \
                left.right.IS_CONST:
            c2 = right.value
            c1 = left.right.value
            if c2 > c1:
                expr = ExprLShift(left.left, ExprConst(32, c2 - c1))
            elif c2 == c1:
                expr = left.left
            else:
                expr = ExprRShift(left.left, ExprConst(32, c1 - c2))
            and_value = ((1 << c2) - 1) ^ ((1 << expr.rtype) - 1)
            expr = ExprAnd(expr, Const(expr.rtype, and_value))
            return expr.optimized

        # "(a + c1) << c2" ==> (a << c2) + (c1 << c2)
        if right_const and \
                left.IS_ADD and len(left.exprs) == 1 and \
                left.const:
            expr_left = ExprLShift(left.exprs[0], right)
            expr_right = ExprConst(left.const.rtype,
                                   left.const.value << right.value)
            return ExprAdd((expr_left,), expr_right).optimized

        return self


class ExprRShift(ExprShift):
    def __init__(self, left, right):
        # Always logical shift
        if left.rtype < 32 or left.rtype == 63:
            left = ExprCast(max(32, left.rtype + 1), left)
        super().__init__(left, right)

    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self.right
        if right.IS_CONST:
            right_s = str(right.value)
        else:
            right_s = str(right)

        return '({} >> {})'.format(self.left, right_s)

    @utils.cached_property
    def optimized(self):
        '''
        >>> expr = (Add(FixedVar(32, 'c'), 30) >> 2)
        >>> str(expr.optimized)
        '(((c + 2u) >> 2) + 7u)'
        >>> expr = (Add(FixedVar(32, 'c'), FixedVar(32, 'd'), -30) >> 2)
        >>> str(expr.optimized)
        '(((c + d + 2u) >> 2) - 8u)'
        '''
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        # (a + c1) >> c2
        # Convert to ((a + c1 % (1 << c2)) >> c2) + (c1 >> c2).
        if right_const and left.IS_ADD and left.const:
            ctype = left.const.rtype
            c1 = left.const.value
            c2 = right.value

            if c1 >> c2:

                compensation = c1 >> c2
                remainder = c1 - (compensation << c2)
                if remainder < 0:
                    compensation += 1
                    remainder -= 1 << c2

                expr = ExprAdd(left.exprs, ExprConst(ctype, remainder))
                expr = ExprRShift(expr, ExprConst(32, c2))
                expr = ExprAdd((expr,), ExprConst(ctype, compensation))
                return expr.optimized

        # (a >> c1) >> c2 ==> a >> (c1 + c2)
        if right_const and \
                left.IS_RSHIFT and \
                left.right.IS_CONST:
            self.right = right = Add(right, left.right).optimized
            self.left = left = left.left

        return self


class ExprMul(ExprBinary):
    def __str__(self):
        return '({} * {})'.format(self.left, self.right)

    @utils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        # Both constants
        if right_const and left.IS_CONST:
            return ExprConst(self.rtype,
                             left.value * right.value)

        # Put constant on the right side
        if not right_const and left.IS_CONST:
            self.left, self.right = left, right = right, left
            right_const = True

        if right_const:
            # Strength reduction (* => <<)
            rv = right.value
            if rv == 0:
                return ExprConst(32, 0)
            elif rv == 1:
                return left
            elif (rv > 0) and (rv & (rv - 1)) == 0:  # Power of 2
                expr = ExprLShift(left, ExprConst(32, rv.bit_length() - 1))
                return expr.optimized

            # (a + c1) * c2 ==> (a * c2 + c1 * c2)
            if left.IS_ADD and len(left.exprs) == 1 and left.const:
                expr_left = ExprMul(left.exprs[0], right)
                expr_right = ExprMul(left.const, right)
                return ExprAdd((expr_left, expr_right), None).optimized

            # (cond ? c1 : c2) * c3 ==> (cond ? c1 * c3 : c2 * c3)
            if left.IS_COND and \
                    left.exprT.IS_CONST and \
                    left.exprF.IS_CONST:
                expr = ExprCond(left.cond,
                                ExprMul(left.exprT, right),
                                ExprMul(left.exprF, right))
                return expr.optimized

            # (a & 1) * c ==> (a & 1) ? c : 0
            if left.IS_AND and \
                    left.right.IS_CONST and \
                    left.right.value == 1:
                expr = ExprCond(left, right, ExprConst(self.rtype, 0))
                return expr.optimized

        return self


class ExprDiv(ExprBinary):
    IS_ALSO = 'DIV_MOD',

    def __init__(self, left, right):
        if left.rtype < 32 or left.rtype == 63:
            left = ExprCast(max(32, left.rtype + 1), left)
        super().__init__(left, right)

    def __str__(self):
        return '({} / {})'.format(self.left, self.right)

    @utils.cached_property
    def optimized(self):
        left = self.left
        right = self.right
        if right.IS_CONST:
            rv = right.value
            if rv == 0:
                raise ZeroDivisionError
            elif rv == 1:
                return left
            elif (rv & (rv - 1)) == 0:
                expr = ExprRShift(left, ExprConst(32, rv.bit_length() - 1))
                return expr.optimized

        return self


class ExprMod(ExprBinary):
    IS_ALSO = 'DIV_MOD',

    def __str__(self):
        return '({} % {})'.format(self.left, self.right)

    @utils.cached_property
    def optimized(self):
        right = self.right
        if right.IS_CONST:
            value = right.value
            if value and (value & (value - 1)) == 0:
                return ExprAnd(self.left,
                               ExprConst(right.rtype, value - 1)).optimized
        return self


class ExprAnd(ExprBinary):
    def __str__(self):
        return '({} & {})'.format(self.left, self.right)

    @utils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST
        right_value = None

        if right_const:
            right_value = right.value

        # (a + c1) & c2 ==> (a + c1') & c2
        # where c1' = c1 with high bits cleared
        if right_const and right_value and \
                left.IS_ADD and left.const:
            rv = right_value
            bt = rv.bit_length() + 1
            c1 = left.const.value
            c1p = c1 & ((1 << bt) - 1)
            # If its high bit is set, make it negative
            if c1p & (1 << (bt - 1)):
                c1p |= ~((1 << bt) - 1)
            if c1p != c1:
                left = ExprAdd(left.exprs, ExprConst(left.const.rtype, c1p))
                self.left = left = left.optimized

        # (a & c1) & c2 ==> a & (c1 & c2)
        if right_const and \
                left.IS_AND and \
                left.right.IS_CONST:
            c1 = left.right.value
            c2 = right_value
            expr = ExprAnd(left.left,
                           Const(max(left.right.rtype, right.rtype), c1 & c2))
            return expr.optimized

        # (a & 0xff) ==> (uint8_t)a
        # (a & 0xffff) ==> (uint16_t)a
        if right_const:
            # Must cast back
            if right_value == 0xff:
                expr = ExprCast(self.rtype, ExprCast(8, left))
                return expr.optimized
            elif right_value == 0xffff:
                expr = ExprCast(self.rtype, ExprCast(16, left))
                return expr.optimized

        return self


class ExprCompare(ExprBinary):
    def __init__(self, left, compare, right):
        super().__init__(left, right, 31)
        self.compare = compare

    def __str__(self):
        return '({} {} {})'.format(self.left, self.compare, self.right)

    @utils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        # (a >> c1) == c2
        # a >= (c2 << c1) && a < ((c2 + 1) << c1)
        # unsinged(a - (c2 << c1)) < (1 << c1)
        if right_const and self.compare == '==' and \
                left.IS_RSHIFT and \
                left.left.rtype == 32 and \
                left.right.IS_CONST and \
                right.rtype == 32:
            c1 = left.right.value
            c2 = right.value
            if ((c2 + 1) << c1) <= 2**32:
                expr = ExprAdd((left.left,), ExprConst(32, -(c2 << c1)))
                expr = ExprCompare(expr, '<', ExprConst(32, 1 << c1))
                return expr.optimized

        # (a >> c1) < c2
        # a < (c2 << c1)
        if right_const and self.compare == '<' and \
                left.IS_RSHIFT and \
                left.left.rtype == 32 and \
                left.right.IS_CONST and \
                right.rtype == 32:
            c1 = left.right.value
            c2 = right.value
            if (c2 << c1) < 2**32:
                expr = ExprCompare(left.left, '<', ExprConst(32, c2 << c1))
                return expr.optimized

        return self


class ExprCast(Expr):
    def __init__(self, type, value):
        self.rtype = type
        self.value = value.optimized

    def __str__(self):
        return '{}({})'.format(type_name(self.rtype),
                               utils.trim_brackets(str(self.value)))

    @property
    def children(self):
        return self.value,

    @utils.cached_property
    def optimized(self):
        rtype = self.rtype
        value = self.value

        if value.rtype == rtype:
            return value

        if value.IS_CAST and rtype <= value.rtype:
            return ExprCast(rtype, value.value).optimized

        return self


class ExprCond(Expr):
    __slots__ = 'cond', 'exprT', 'exprF', 'rtype'

    def __init__(self, cond, exprT, exprF):
        self.cond = cond.optimized
        self.exprT = exprT.optimized
        self.exprF = exprF.optimized
        self.rtype = max(31, self.exprT.rtype, self.exprF.rtype)

    def __str__(self):
        return '({} ? {} : {})'.format(self.cond, self.exprT, self.exprF)

    @property
    def children(self):
        return self.cond, self.exprT, self.exprF

    def extract_subexprs(self, threshold, callback, allow_new):
        # It can be unsafe to evaluate exprT or exprF without first checking
        # cond
        if not self.cond.IS_VAR:
            self.cond.extract_subexprs(threshold, callback, allow_new)
            self.cond = callback(
                self.cond, allow_new and self.cond._complicated(threshold))
        if not allow_new:
            self.exprT = callback(self.exprT, False)
            self.exprF = callback(self.exprF, False)
        self.exprT.extract_subexprs(threshold, callback, False)
        self.exprF.extract_subexprs(threshold, callback, False)


class ExprTable(Expr):
    def __init__(self, type, name, values, var, offset):
        self.rtype = type
        self.name = name
        self.values = values
        self.var = var = var.optimized
        self.offset = offset

    def __str__(self):
        if self.offset > 0:
            # Add an extra 'l' so that the constant is absorbed by the
            # address of the array
            offset_s = '{:#x}'.format(self.offset)
            if self.var.rtype < 63:
                offset_s += 'l'
            return '{}[{} - {}]'.format(
                self.name, self.var, offset_s)
        elif self.offset < 0:
            # Don't add 'l' in this case, to avoid signed/unsigned
            # extension problems
            return '{}[{} + {:#x}]'.format(self.name,
                                           self.var, -self.offset)
        else:
            var = utils.trim_brackets(str(self.var))
            return '{}[{}]'.format(self.name, var)

    def statics(self, vs):
        id_ = id(self)
        if id_ in vs:
            return ''

        vs.add(id_)

        var_statics = self.var.statics(vs)

        if _speedups:
            c_array = _speedups.format_c_array(
                self.values, self.rtype, self.name)
            if c_array is not None:
                return var_statics + c_array

        res = [var_statics]
        res_append = res.append

        indlen = len(hex(self.values.size))
        maxlen = len(hex(utils.np_max(self.values)))

        # I understand this is not the "correct" way to go, but this is
        # for performance.
        # If I don't care about performance, I could do '{:#0{}x}'.format(v, l)
        line_start_format = '\t/* {{:#0{}x}} */'.format(indlen).format
        value_format = ' {{:#0{}x}},'.format(maxlen).format

        line = 'const {} {}[{:#x}] = {{'.format(
            type_name(self.rtype), self.name, self.values.size)
        for i, v in enumerate(self.values):
            if not (i & 7):
                res_append(line + '\n')
                line = line_start_format(i)
            line += value_format(v)
        res_append(line.rstrip(',') + '\n')
        res_append('};\n\n')
        return ''.join(res)

    @property
    def children(self):
        return self.var,

    @utils.cached_property
    def optimized(self):
        var = self.var
        # Absorb constants into offset
        if var.IS_ADD and var.const:
            self.offset -= var.const.value
            self.var = var = ExprAdd(var.exprs, None).optimized
        return self

    def table_bytes(self, *, type_bytes=type_bytes):
        return self.values.size * type_bytes(self.rtype)

    def extract_subexprs(self, threshold, callback, allow_new):
        super().extract_subexprs(threshold, callback, allow_new)
        self.var = callback(self.var,
                            allow_new and self.var._complicated(threshold))

    def _complicated(self, _threshold):
        return True


### Factory functions
def exprize(expr, *,
            isinstance=isinstance, Expr=Expr, ExprConst=ExprConst):
    '''Convert int to ExprConst'''
    if isinstance(expr, Expr):
        return expr
    else:
        return ExprConst(32, expr)


FixedVar = ExprFixedVar


TempVar = ExprTempVar


Const = ExprConst


def Add(*in_exprs):
    exprs = []
    const_exprs = []

    for expr in in_exprs:
        expr = exprize(expr)
        if expr.IS_CONST:
            const_exprs.append(expr)
        elif expr.IS_ADD:
            exprs.extend(expr.exprs)
            if expr.const:
                const_exprs.append(expr.const)
        else:
            exprs.append(expr)

    const_expr = ExprConst.combine(const_exprs)
    if not exprs:
        return const_expr or ExprConst(32, 0)
    elif len(exprs) == 1 and not const_expr:
        return exprs[0]
    else:
        return ExprAdd(exprs, const_expr)


def Cast(type, value):
    return ExprCast(type, exprize(value))


def Cond(cond, exprT, exprF):
    return ExprCond(exprize(cond), exprize(exprT), exprize(exprF))
