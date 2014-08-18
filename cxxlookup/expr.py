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


# Signed typed only allowed for intermediate values
I8 = 7
U8 = 8
I16 = 15
U16 = 16
I32 = 31
U32 = 32
I64 = 63
U64 = 64


TypeNames = {
    I8: 'int8_t',
    U8: 'uint8_t',
    I16: 'int16_t',
    U16: 'uint16_t',
    I32: 'int32_t',
    U32: 'uint32_t',
    I64: 'int64_t',
    U64: 'uint64_t',
}


TypeBytes = {
    I8: 1,
    U8: 1,
    I16: 2,
    U16: 2,
    I32: 4,
    U32: 4,
    I64: 8,
    U64: 8,
}


def const_type(value):
    value = int(value)
    if value >= 2**32:
        return U64
    elif value >= 2**16:
        return U32
    elif value >= 2**8:
        return U16
    else:
        return U8


class Expr:
    def __str__(self):
        raise NotImplementedError

    def statics(self):
        return ''.join(x.statics() for x in self.children())

    def children(self):
        return ()

    def rettype(self):
        type = max(x.rettype() for x in self.children())
        return max(type, I32)  # C type-promotion rule

    def optimize(self, flags=0):
        return self

    optimize_absorb_constant = 1

    def walk(self, type_or_type_tuple=None,
             len=len, isinstance=isinstance):
        res = []
        stk = [self]
        stk_pop = stk.pop
        stk_extend = stk.extend
        res_append = res.append
        while stk:
            expr = stk_pop()
            if not type_or_type_tuple or isinstance(expr, type_or_type_tuple):
                res_append(expr)
            stk_extend(expr.children())
        return res

    def complexity(self):
        return len(self.walk())


class ExprVar(Expr):
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def rettype(self):
        return U32  # Assume so


class ExprConst(Expr):
    def __init__(self, type, value):
        self._type = type
        self._value = value

    def __str__(self):
        if -10 < self._value < 10:
            value_s = str(self._value)
        else:
            value_s = hex(self._value)
        if self._type < U64:
            return value_s + 'u'
        else:
            return 'UINT64_C({})'.format(value_s)

    def rettype(self):
        return self._type

    @staticmethod
    def combine(const_exprs):
        """Combine multiple ExprConst into one."""
        const_value = 0
        const_type = U32
        for expr in const_exprs:
            const_value += expr._value
            const_type = max(const_type, expr._type)
        if const_value == 0:
            return None
        else:
            return ExprConst(const_type, const_value)


class ExprAdd(Expr):
    def __init__(self, exprs, const):
        assert isinstance(const, (type(None), ExprConst))
        self._exprs = tuple(exprs)
        self._const = const

    def __str__(self):
        res = ' + '.join(map(str, self._exprs))
        if self._const:
            const_value = self._const._value
            if const_value >= 0:
                res += ' + ' + str(self._const)
            else:
                res += ' - ' + str(ExprConst(self._const._type,
                                             -const_value))
        return '(' + res + ')'

    def children(self):
        if self._const:
            return self._exprs + (self._const,)
        else:
            return self._exprs

    def optimize(self, flags=0):

        if self._const:
            flags |= Expr.optimize_absorb_constant

        exprs = []
        const_exprs = []
        if self._const:
            const_exprs.append(self._const)

        for expr in self._exprs:
            expr = expr.optimize(flags)
            if isinstance(expr, ExprAdd):
                exprs.extend(expr._exprs)
                if expr._const:
                    const_exprs.append(expr._const)
            elif isinstance(expr, ExprConst):
                const_exprs.append(expr)
            else:
                exprs.append(expr)

        self._exprs = tuple(exprs)
        self._const = ExprConst.combine(const_exprs)

        if len(self._exprs) == 1 and not self._const:
            return self._exprs[0]

        return self


class ExprBinary(Expr):
    def __init__(self, left, right):
        self._left = left
        self._right = right

    def children(self):
        return self._left, self._right

    def optimize_children(self):
        self._left = self._left.optimize()
        self._right = self._right.optimize()

    def optimize(self, flags=0):
        self.optimize_children()
        return self

    def rettype(self):
        return max(I32, self._left.rettype(), self._right.rettype())


class ExprShift(ExprBinary):
    def rettype(self):
        return max(I32, self._left.rettype())


class ExprLShift(ExprShift):
    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self._right
        if isinstance(right, ExprConst):
            return '({} << {})'.format(self._left, right._value)
        else:
            return '({} << {})'.format(self._left, right)

    def optimize(self, flags=0):
        self.optimize_children()

        left = self._left
        right = self._right

        if isinstance(left, ExprConst) and isinstance(right, ExprConst):
            return ExprConst(self.rettype(), left._value << right._value)

        # "(a & c1) << c2" ==> (a << c2) & (c1 << c2) (where c2 <= 3)
        # This takes advantage of x86's LEA instruction
        if isinstance(left, ExprAnd) and \
                isinstance(left._right, ExprConst) and \
                isinstance(right, ExprConst) and right._value <= 3:
            expr_left = ExprLShift(left._left, right)
            expr_right = ExprConst(left._right._type,
                                   left._right._value << right._value)
            return ExprAnd(expr_left, expr_right).optimize()

        # (cond ? c1 : c2) << c3 ==> (cond ? c1 << c3 : c2 << c3)
        if isinstance(left, ExprCond) and \
                isinstance(left._exprT, ExprConst) and \
                isinstance(left._exprF, ExprConst) and \
                isinstance(right, ExprConst):
            expr = ExprCond(left._cond,
                            ExprLShift(left._exprT, right),
                            ExprLShift(left._exprF, right))
            return expr.optimize()

        # (a >> c1) << c2  (c2 > c1)
        # (a << (c2 - c1)) & ~((1 << c2) - 1)
        if isinstance(left, ExprRShift) and \
                isinstance(left._right, ExprConst) and \
                isinstance(right, ExprConst) and \
                right._value > left._right._value:
            c2 = right._value
            c1 = left._right._value
            expr = ExprLShift(left._left, ExprConst(U32, c2 - c1))
            and_value = ((1 << c2) - 1) ^ ((1 << expr.rettype()) - 1)
            expr = ExprAnd(expr, Const(expr.rettype(), and_value))
            return expr.optimize()

        # "(a + c1) << c2" ==> (a << c2) + (c1 << c2)
        if isinstance(left, ExprAdd) and len(left._exprs) == 1 and \
                left._const and isinstance(right, ExprConst):
            expr_left = ExprLShift(left._exprs[0], right)
            expr_right = ExprConst(left._const._type,
                                   left._const._value << right._value)
            return ExprAdd((expr_left,), expr_right).optimize()

        return self


class ExprRShift(ExprShift):
    def __str__(self):
        # Always logical shift
        left_type = self._left.rettype()
        if left_type < U32:
            convert_to = 'uint32_t'
        elif left_type == I64:
            convert_to = 'uint64_t'
        else:
            convert_to = None

        if convert_to:
            left_s = '{}({})'.format(convert_to,
                                     utils.trim_brackets(str(self._left)))
        else:
            left_s = str(self._left)

        # Avoid the spurious 'u' after the constant
        if isinstance(self._right, ExprConst):
            right_s = str(self._right._value)
        else:
            right_s = str(self._right)

        return '({} >> {})'.format(left_s, right_s)

    def optimize(self, flags=0):
        self.optimize_children()
        left = self._left
        right = self._right

        if (flags & Expr.optimize_absorb_constant):
            # (a + c1) >> c2
            # Convert to ((a + c1 % (1 << c2)) >> c2) + (c1 >> c2).
            if isinstance(left, ExprAdd) and isinstance(right, ExprConst) and \
                    len(left._exprs) == 1 and left._const:
                ctype = left._const._type
                c1 = left._const._value
                c2 = right._value

                if c1 >> c2:

                    remainder = c1 - (c1 >> c2 << c2)

                    expr = left._exprs[0]
                    if remainder:
                        expr = ExprAdd((expr,), ExprConst(ctype, c1))
                    expr = ExprRShift(expr, ExprConst(U32, c2))
                    expr = ExprAdd((expr,), ExprConst(ctype, c1 >> c2))
                    return expr

        # (a >> c1) >> c2 ==> a >> (c1 + c2)
        if isinstance(right, ExprConst) and \
                isinstance(left, ExprRShift) and \
                isinstance(left._right, ExprConst):
            self._right = right = Add(right, left._right).optimize()
            self._left = left = left._left

        return self


class ExprMul(ExprBinary):
    def __str__(self):
        return '{} * {}'.format(self._left, self._right)

    def optimize(self, flags=0):
        self.optimize_children()

        left = self._left
        right = self._right

        # Both constants
        if isinstance(left, ExprConst) and isinstance(right, ExprConst):
            return ExprConst(self.rettype(),
                             left._value * right._value)

        # Put constant on the right side
        if isinstance(left, ExprConst):
            self._left, self._right = left, right = right, left

        # (a + c1) * c2 ==> (a * c2 + c1 * c2)
        if isinstance(left, ExprAdd) and len(left._exprs) == 1 and \
                left._const and \
                isinstance(right, ExprConst):
            expr_left = ExprMul(left._exprs[0], right)
            expr_right = ExprMul(left._const, right)
            return ExprAdd((expr_left, expr_right), None).optimize()

        # (cond ? c1 : c2) * c3 ==> (cond ? c1 * c3 : c2 * c3)
        if isinstance(left, ExprCond) and \
                isinstance(left._exprT, ExprConst) and \
                isinstance(left._exprF, ExprConst) and \
                isinstance(right, ExprConst):
            expr = ExprCond(left._cond,
                            Mul(left._exprT, right),
                            Mul(left._exprF, right))
            return expr.optimize()

        # Strength reduction (* => <<)
        if isinstance(right, ExprConst):
            rv = right._value
            if rv == 0:
                return ExprConst(U32, 0)
            elif rv == 1:
                return left
            elif (rv > 0) and (rv & (rv - 1)) == 0:  # Power of 2
                expr = ExprLShift(left, ExprConst(U32, rv.bit_length() - 1))
                return expr.optimize()

        return self


class ExprAnd(ExprBinary):
    def __str__(self):
        return '({} & {})'.format(self._left, self._right)

    def optimize(self, flags=0):
        self.optimize_children()

        left = self._left
        right = self._right

        # (a + c1) & c2 ==> (a + c1') & c2
        # where c1' = c1 with high bits cleared
        if isinstance(left, ExprAdd) and left._const and \
                isinstance(right, ExprConst) and right._value != 0:
            rv = right._value
            bt = rv.bit_length() + 1
            c1p = left._const._value & ((1 << bt) - 1)
            # If its high bit is set, make it negative
            if c1p & (1 << (bt - 1)):
                c1p |= ~((1 << bt) - 1)
            left = ExprAdd(left._exprs, ExprConst(left._const._type, c1p))
            self._left = left = left.optimize()

        # (a & c1) & c2 ==> a & (c1 & c2)
        if isinstance(left, ExprAnd) and \
                isinstance(left._right, ExprConst) and \
                isinstance(right, ExprConst):
            c1 = left._right._value
            c2 = right._value
            expr = ExprAnd(left._left,
                           Const(max(left._right._type, right._type), c1 & c2))
            return expr.optimize()

        # (a & 0xff) ==> (uint8_t)a
        # (a & 0xffff) ==> (uint16_t)a
        if isinstance(right, ExprConst):
            # Must cast back
            if right._value == 0xff:
                return ExprCast(self.rettype(), ExprCast(U8, left)).optimize()
            elif right._value == 0xffff:
                return ExprCast(self.rettype(), ExprCast(U16, left)).optimize()
        return self


class ExprCompare(ExprBinary):
    def __init__(self, left, compare, right):
        super().__init__(left, right)
        self._compare = compare

    def __str__(self):
        return '({} {} {})'.format(self._left, self._compare, self._right)

    def rettype(self):
        return I32


class ExprCast(Expr):
    def __init__(self, type, value):
        self._type = type
        self._value = value

    def __str__(self):
        return '{}({})'.format(TypeNames[self._type],
                               utils.trim_brackets(str(self._value)))

    def children(self):
        return self._value,

    def rettype(self):
        return self._type

    def optimize(self, flags=0):
        self._value = value = self._value.optimize()

        if value.rettype() == self._type:
            return value

        if isinstance(value, ExprCast) and self._type <= value._type:
            return ExprCast(self._type, value._value).optimize(flags=flags)

        return self


class ExprCond(Expr):
    def __init__(self, cond, exprT, exprF):
        self._cond = cond
        self._exprT = exprT
        self._exprF = exprF

    def __str__(self):
        return '({} ? {} : {})'.format(self._cond, self._exprT, self._exprF)

    def children(self):
        return self._cond, self._exprT, self._exprF

    def rettype(self):
        # FIXME: What does C standard say about this?
        return max(I32, self._exprT.rettype(), self._exprF.rettype())

    def optimize(self, flags=0):
        self._cond = self._cond.optimize()
        self._exprT = self._exprT.optimize()
        self._exprF = self._exprF.optimize()
        return self


class ExprTable(Expr):
    def __init__(self, name, values, var, offset):
        self._name = name
        self._values = values
        self._type = const_type(values.max())
        self._var = var
        self._offset = offset

    def __str__(self):
        if self._offset > 0:
            return '{}[{} - {:#x}l]'.format(
                self._name, self._var, self._offset)
        elif self._offset < 0:
            # Don't add 'l' in this case, to avoid signed/unsigned
            # extension problems
            return '{}[{} + {:#x}]'.format(self._name,
                                           self._var, -self._offset)
        else:
            var = utils.trim_brackets(str(self._var))
            return '{}[{}]'.format(self._name, var)

    def statics(self):
        res = [self._var.statics()]

        indlen = len(hex(self._values.size))
        maxlen = len(hex(self._values.max()))
        line = 'const {} {}[{:#x}] = {{'.format(
            TypeNames[self._type], self._name, self._values.size)
        for i, v in enumerate(self._values):
            if (i % 8) == 0:
                res.append(line.rstrip() + '\n')
                line = '\t/* {:#0{}x} */ '.format(i, indlen)
            line += '{:#0{}x}, '.format(v, maxlen)
        res.append(line.rstrip(', ') + '\n')
        res.append('};\n\n')
        return ''.join(res)

    def rettype(self):
        return self._type

    def children(self):
        return self._var,

    def optimize(self, flags=0):
        self.var = self._var.optimize(flags=Expr.optimize_absorb_constant)
        # Absorb constants into offset
        if isinstance(self._var, ExprAdd) and \
                self._var._const:
            self._offset -= self._var._const._value
            self._var = ExprAdd(self._var._exprs, None)
        return self

    def table_bytes(self):
        return self._values.size * TypeBytes[self._type]


### Factory functions
def exprize(expr,
            isinstance=isinstance, Expr=Expr, U32=U32, ExprConst=ExprConst):
    '''Convert int to ExprConst'''
    if isinstance(expr, Expr):
        return expr
    else:
        return ExprConst(U32, int(expr))


def Var(name):
    return ExprVar(name)


def Const(type, value):
    return ExprConst(type, value)


def Add(*in_exprs):
    exprs = []
    const_exprs = []

    for expr in in_exprs:
        expr = exprize(expr)
        if isinstance(expr, ExprConst):
            const_exprs.append(expr)
        else:
            exprs.append(expr)

    const_expr = ExprConst.combine(const_exprs)
    if not exprs:
        return const_expr or ExprConst(U32, 0)
    elif len(exprs) == 1 and not const_expr:
        return exprs[0]
    else:
        return ExprAdd(exprs, const_expr)


def LShift(left, right):
    return ExprLShift(exprize(left), exprize(right))


def RShift(left, right):
    return ExprRShift(exprize(left), exprize(right))


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


def Table(name, values, var, offset):
    return ExprTable(name, values, exprize(var), offset)
