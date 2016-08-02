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
        return result


class Expr(metaclass=ExprMeta):
    def __str__(self):
        raise NotImplementedError

    def statics(self):
        return ''.join(filter(None, (x.statics() for x in self.children())))

    def children(self):
        return ()

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
            children = expr.children()
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

    def replace_complicated_subexpressions(self, threshold, callback):
        for subexpr in self.children():
            subexpr.replace_complicated_subexpressions(threshold, callback)


class ExprVar(Expr):
    def __init__(self, type):
        self.rtype = type


class ExprFixedVar(ExprVar):
    def __init__(self, type, name):
        super().__init__(type)
        self._name = name

    def __str__(self):
        return self._name


class ExprTempVar(ExprVar):
    _name_cache = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self, type, var):
        super().__init__(type)
        self._var = var

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
        return self.get_name(self._var)


class ExprConst(Expr):
    def __init__(self, type, value):
        self.rtype = type
        self._value = value

    def __str__(self):
        if -10 < self._value < 10:
            value_s = str(self._value)
        else:
            value_s = hex(self._value)
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
            const_value += expr._value
            const_type = max(const_type, expr.rtype)
        if const_value == 0:
            return None
        else:
            return ExprConst(const_type, const_value)


class ExprAdd(Expr):
    def __init__(self, exprs, const):
        assert const is None or const.IS_CONST
        self._exprs = tuple(exprs)
        self._const = const

    def __str__(self):
        res = ' + '.join(map(str, self._exprs))
        if self._const:
            const_value = self._const._value
            if const_value >= 0:
                res += ' + ' + str(self._const)
            else:
                res += ' - ' + str(ExprConst(self._const.rtype,
                                             -const_value))
        return '(' + res + ')'

    def children(self):
        const = self._const
        if const:
            return self._exprs + (const,)
        else:
            return self._exprs

    @utils.cached_property
    def rtype(self):
        rtype = max([x.rtype for x in self.children()])
        return max(rtype, 31)  # C type-promotion rule

    @utils.cached_property
    def optimized(self):
        exprs = []
        const_exprs = []

        if self._const:
            const_exprs.append(self._const)

        for expr in self._exprs:
            expr = expr.optimized
            if expr.IS_ADD:
                exprs.extend(expr._exprs)
                if expr._const:
                    const_exprs.append(expr._const)
            elif expr.IS_CONST:
                const_exprs.append(expr)
            else:
                exprs.append(expr)

        self._exprs = tuple(exprs)
        self._const = ExprConst.combine(const_exprs)

        if len(self._exprs) == 1 and not self._const:
            return self._exprs[0]

        return self

    def replace_complicated_subexpressions(self, threshold, callback):
        exprs = []
        for expr in self._exprs:
            expr.replace_complicated_subexpressions(threshold, callback)
            if expr._complicated(threshold):
                expr = callback(expr)
            exprs.append(expr)
        self._exprs = tuple(exprs)

        if self._const and self._const._complicated(threshold):
            self._const = callback(self._const)


class ExprBinary(Expr):
    def __init__(self, left, right):
        self._left = left
        self._right = right

    def children(self):
        return self._left, self._right

    @utils.cached_property
    def optimized(self):
        self._left = self._left.optimized
        self._right = self._right.optimized
        return self

    @property
    def rtype(self):
        return max(31, self._left.rtype, self._right.rtype)

    def replace_complicated_subexpressions(self, threshold, callback):
        super().replace_complicated_subexpressions(threshold, callback)
        if self._left._complicated(threshold):
            self._left = callback(self._left)
        if self._right._complicated(threshold):
            self._right = callback(self._right)


class ExprShift(ExprBinary):
    @property
    def rtype(self):
        return max(31, self._left.rtype)


class ExprLShift(ExprShift):
    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self._right
        if right.IS_CONST:
            if right._value in (1, 2, 3):
                return '{} * {}'.format(self._left, 1 << right._value)
            return '({} << {})'.format(self._left, right._value)
        else:
            return '({} << {})'.format(self._left, right)

    @utils.cached_property
    def optimized(self):
        self._left = left = self._left.optimized
        self._right = right = self._right.optimized

        right_const = right.IS_CONST

        if right_const and left.IS_CONST:
            return ExprConst(self.rtype, left._value << right._value)

        # "(a & c1) << c2" ==> (a << c2) & (c1 << c2) (where c2 <= 3)
        # This takes advantage of x86's LEA instruction
        if right_const and right._value <= 3 and \
                left.IS_AND and \
                left._right.IS_CONST:
            expr_left = ExprLShift(left._left, right)
            expr_right = ExprConst(left._right.rtype,
                                   left._right._value << right._value)
            return ExprAnd(expr_left, expr_right).optimized

        # (cond ? c1 : c2) << c3 ==> (cond ? c1 << c3 : c2 << c3)
        if right_const and \
                left.IS_COND and \
                left._exprT.IS_CONST and \
                left._exprF.IS_CONST:
            expr = ExprCond(left._cond,
                            ExprLShift(left._exprT, right),
                            ExprLShift(left._exprF, right))
            return expr.optimized

        # (a >> c1) << c2
        # (a << (c2 - c1)) & ~((1 << c2) - 1)   (c2 > c1)
        # (a >> (c1 - c2)) & ~((1 << c2) - 1)   (c2 <= c1)
        if right_const and \
                left.IS_RSHIFT and \
                left._right.IS_CONST:
            c2 = right._value
            c1 = left._right._value
            if c2 > c1:
                expr = ExprLShift(left._left, ExprConst(32, c2 - c1))
            elif c2 == c1:
                expr = left._left
            else:
                expr = ExprRShift(left._left, ExprConst(32, c1 - c2))
            and_value = ((1 << c2) - 1) ^ ((1 << expr.rtype) - 1)
            expr = ExprAnd(expr, Const(expr.rtype, and_value))
            return expr.optimized

        # "(a + c1) << c2" ==> (a << c2) + (c1 << c2)
        if right_const and \
                left.IS_ADD and len(left._exprs) == 1 and \
                left._const:
            expr_left = ExprLShift(left._exprs[0], right)
            expr_right = ExprConst(left._const.rtype,
                                   left._const._value << right._value)
            return ExprAdd((expr_left,), expr_right).optimized

        return self


class ExprRShift(ExprShift):
    def __init__(self, left, right):
        super().__init__(left, right)
        # Always logical shift
        assert left.rtype in (8, 16, 32, 64)

    def __str__(self):
        # Avoid the spurious 'u' after the constant
        if self._right.IS_CONST:
            right_s = str(self._right._value)
        else:
            right_s = str(self._right)

        return '({} >> {})'.format(self._left, right_s)

    @utils.cached_property
    def optimized(self):
        '''
        >>> expr = RShift(Add(FixedVar(32, 'c'), 30), 2)
        >>> str(expr.optimized)
        '(((c + 2u) >> 2) + 7u)'
        >>> expr = RShift(Add(FixedVar(32, 'c'), FixedVar(32, 'd'), -30), 2)
        >>> str(expr.optimized)
        '(((c + d + 2u) >> 2) - 8u)'
        '''
        self._left = left = self._left.optimized
        self._right = right = self._right.optimized

        right_const = right.IS_CONST

        # (a + c1) >> c2
        # Convert to ((a + c1 % (1 << c2)) >> c2) + (c1 >> c2).
        if right_const and left.IS_ADD and left._const:
            ctype = left._const.rtype
            c1 = left._const._value
            c2 = right._value

            if c1 >> c2:

                remainder = c1 - (c1 >> c2 << c2)

                expr = ExprAdd(left._exprs, ExprConst(ctype, remainder))
                expr = ExprRShift(expr, ExprConst(32, c2))
                expr = ExprAdd((expr,), ExprConst(ctype, c1 >> c2))
                return expr.optimized

        # (a >> c1) >> c2 ==> a >> (c1 + c2)
        if right_const and \
                left.IS_RSHIFT and \
                left._right.IS_CONST:
            self._right = right = Add(right, left._right).optimized
            self._left = left = left._left

        return self


class ExprMul(ExprBinary):
    def __str__(self):
        return '{} * {}'.format(self._left, self._right)

    @utils.cached_property
    def optimized(self):
        self._left = left = self._left.optimized
        self._right = right = self._right.optimized

        right_const = right.IS_CONST

        # Both constants
        if right_const and left.IS_CONST:
            return ExprConst(self.rtype,
                             left._value * right._value)

        # Put constant on the right side
        if not right_const and left.IS_CONST:
            self._left, self._right = left, right = right, left
            right_const = True

        # (a + c1) * c2 ==> (a * c2 + c1 * c2)
        if right_const and \
                left.IS_ADD and len(left._exprs) == 1 and \
                left._const:
            expr_left = ExprMul(left._exprs[0], right)
            expr_right = ExprMul(left._const, right)
            return ExprAdd((expr_left, expr_right), None).optimized

        # (cond ? c1 : c2) * c3 ==> (cond ? c1 * c3 : c2 * c3)
        if right_const and \
                left.IS_COND and \
                left._exprT.IS_CONST and \
                left._exprF.IS_CONST:
            expr = ExprCond(left._cond,
                            Mul(left._exprT, right),
                            Mul(left._exprF, right))
            return expr.optimized

        # Strength reduction (* => <<)
        if right_const:
            rv = right._value
            if rv == 0:
                return ExprConst(32, 0)
            elif rv == 1:
                return left
            elif (rv > 0) and (rv & (rv - 1)) == 0:  # Power of 2
                expr = ExprLShift(left, ExprConst(32, rv.bit_length() - 1))
                return expr.optimized

        return self


class ExprAnd(ExprBinary):
    def __str__(self):
        return '({} & {})'.format(self._left, self._right)

    @utils.cached_property
    def optimized(self):
        left = self._left
        right = self._right

        right_const = right.IS_CONST
        right_value = None

        if right_const:
            self._left = left = left.optimized
            right_value = right._value
        else:
            self._left = left = left.optimized
            self._right = right = right.optimized

        # (a + c1) & c2 ==> (a + c1') & c2
        # where c1' = c1 with high bits cleared
        if right_const and right_value and \
                left.IS_ADD and left._const:
            rv = right_value
            bt = rv.bit_length() + 1
            c1 = left._const._value
            c1p = c1 & ((1 << bt) - 1)
            # If its high bit is set, make it negative
            if c1p & (1 << (bt - 1)):
                c1p |= ~((1 << bt) - 1)
            if c1p != c1:
                left = ExprAdd(left._exprs, ExprConst(left._const.rtype, c1p))
                self._left = left = left.optimized

        # (a & c1) & c2 ==> a & (c1 & c2)
        if right_const and \
                left.IS_AND and \
                left._right.IS_CONST:
            c1 = left._right._value
            c2 = right_value
            expr = ExprAnd(left._left,
                           Const(max(left._right.rtype, right.rtype), c1 & c2))
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
        super().__init__(left, right)
        self._compare = compare

    def __str__(self):
        return '({} {} {})'.format(self._left, self._compare, self._right)

    rtype = 31

    @utils.cached_property
    def optimized(self):
        self._left = left = self._left.optimized
        self._right = right = self._right.optimized

        right_const = right.IS_CONST

        # (a >> c1) == c2
        # a >= (c2 << c1) && a < ((c2 + 1) << c1)
        # unsinged(a - (c2 << c1)) < (1 << c1)
        if right_const and self._compare == '==' and \
                left.IS_RSHIFT and \
                left._left.rtype == 32 and \
                left._right.IS_CONST and \
                right.rtype == 32:
            c1 = left._right._value
            c2 = right._value
            if ((c2 + 1) << c1) <= 2**32:
                expr = ExprAdd((left._left,), ExprConst(32, -(c2 << c1)))
                expr = ExprCompare(expr, '<', ExprConst(32, 1 << c1))
                return expr.optimized

        # (a >> c1) < c2
        # a < (c2 << c1)
        if right_const and self._compare == '<' and \
                left.IS_RSHIFT and \
                left._left.rtype == 32 and \
                left._right.IS_CONST and \
                right.rtype == 32:
            c1 = left._right._value
            c2 = right._value
            if (c2 << c1) < 2**32:
                expr = ExprCompare(left._left, '<', ExprConst(32, c2 << c1))
                return expr.optimized

        return self


class ExprCast(Expr):
    def __init__(self, type, value):
        self.rtype = type
        self._value = value
        self._optimized = False

    def __str__(self):
        return '{}({})'.format(type_name(self.rtype),
                               utils.trim_brackets(str(self._value)))

    def children(self):
        return self._value,

    @utils.cached_property
    def optimized(self):
        self._value = value = self._value.optimized

        if value.rtype == self.rtype:
            return value

        if value.IS_CAST and self.rtype <= value.rtype:
            return ExprCast(self.rtype, value._value).optimized

        return self


class ExprCond(Expr):
    def __init__(self, cond, exprT, exprF):
        self._cond = cond
        self._exprT = exprT
        self._exprF = exprF
        self._optimized = False

    def __str__(self):
        return '({} ? {} : {})'.format(self._cond, self._exprT, self._exprF)

    def children(self):
        return self._cond, self._exprT, self._exprF

    @property
    def rtype(self):
        # FIXME: What does C standard say about this?
        return max(31, self._exprT.rtype, self._exprF.rtype)

    @utils.cached_property
    def optimized(self):
        self._cond = self._cond.optimized
        self._exprT = self._exprT.optimized
        self._exprF = self._exprF.optimized
        return self


class ExprTable(Expr):
    def __init__(self, type, name, values, var, offset):
        self.rtype = type
        self._name = name
        self._values = values
        self._var = var
        self._offset = offset
        self._optimized = False

    def __str__(self):
        if self._offset > 0:
            # Add an extra 'l' so that the constant is absorbed by the
            # address of the array
            offset_s = '{:#x}'.format(self._offset)
            if self._var.rtype < 63:
                offset_s += 'l'
            return '{}[{} - {}]'.format(
                self._name, self._var, offset_s)
        elif self._offset < 0:
            # Don't add 'l' in this case, to avoid signed/unsigned
            # extension problems
            return '{}[{} + {:#x}]'.format(self._name,
                                           self._var, -self._offset)
        else:
            var = utils.trim_brackets(str(self._var))
            return '{}[{}]'.format(self._name, var)

    def statics(self):
        var_statics = self._var.statics()

        if _speedups:
            c_array = _speedups.format_c_array(
                self._values, self.rtype, self._name)
            if c_array is not None:
                return var_statics + c_array

        res = [var_statics]
        res_append = res.append

        indlen = len(hex(self._values.size))
        maxlen = len(hex(utils.np_max(self._values)))

        # I understand this is not the "correct" way to go, but this is
        # for performance.
        # If I don't care about performance, I could do '{:#0{}x}'.format(v, l)
        line_start_format = '\t/* {{:#0{}x}} */'.format(indlen).format
        value_format = ' {{:#0{}x}},'.format(maxlen).format

        line = 'const {} {}[{:#x}] = {{'.format(
            type_name(self.rtype), self._name, self._values.size)
        for i, v in enumerate(self._values):
            if not (i & 7):
                res_append(line + '\n')
                line = line_start_format(i)
            line += value_format(v)
        res_append(line.rstrip(',') + '\n')
        res_append('};\n\n')
        return ''.join(res)

    def children(self):
        return self._var,

    @utils.cached_property
    def optimized(self):
        self._var = self._var.optimized
        # Absorb constants into offset
        if self._var.IS_ADD and self._var._const:
            self._offset -= self._var._const._value
            self._var = ExprAdd(self._var._exprs, None).optimized
        return self

    def table_bytes(self):
        return self._values.size * type_bytes(self.rtype)

    def replace_complicated_subexpressions(self, threshold, callback):
        super().replace_complicated_subexpressions(threshold, callback)
        if self._var._complicated(threshold):
            self._var = callback(self._var)


### Factory functions
def exprize(expr,
            isinstance=isinstance, Expr=Expr, ExprConst=ExprConst):
    '''Convert int to ExprConst'''
    if isinstance(expr, Expr):
        return expr
    else:
        return ExprConst(32, int(expr))


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
        else:
            exprs.append(expr)

    const_expr = ExprConst.combine(const_exprs)
    if not exprs:
        return const_expr or ExprConst(32, 0)
    elif len(exprs) == 1 and not const_expr:
        return exprs[0]
    else:
        return ExprAdd(exprs, const_expr)


def LShift(left, right):
    return ExprLShift(exprize(left), exprize(right))


def RShift(left, right):
    # Promote left
    left = exprize(left)
    if left.rtype < 32:
        left = ExprCast(32, left)
    elif left.rtype == 63:
        left = ExprCast(64, left)
    return ExprRShift(left, exprize(right))


def Mul(left, right):
    return ExprMul(exprize(left), exprize(right))


def And(left, right):
    return ExprAnd(exprize(left), exprize(right))


def Compare(left, compare, right):
    return ExprCompare(exprize(left), compare, exprize(right))


def Cast(type, value):
    return ExprCast(type, exprize(value))


def Cond(cond, exprT, exprF):
    return ExprCond(exprize(cond), exprize(exprT), exprize(exprF))


def Table(type, name, values, var, offset):
    return ExprTable(type, name, values, exprize(var), offset)
