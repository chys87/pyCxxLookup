# distutils: language=c++
# cython: language_level=3
# cython: profile=True


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

import threading

cimport cython
from libc.stdint cimport int64_t, uint32_t, uint64_t

from . import cutils
from . import utils
from . import _speedups

from .cutils cimport is_pow2


# Signed types only allowed for intermediate values
# Unsigned: Number of bits
# Signed: Number of bits - 1
# E.g.: 31 = int32_t; 32 = uint32_t
cdef str type_name(uint32_t type):
    '''
    >>> type_name(7), type_name(32)
    ('int8_t', 'uint32_t')
    '''
    if (type & 1):
        return f'int{type + 1}_t'
    else:
        return f'uint{type}_t'


cdef uint32_t type_bytes(uint32_t type_) nogil:
    '''
    >>> list(map(type_bytes, [7, 8, 15, 16, 31, 32, 63, 64]))
    [1, 1, 2, 2, 4, 4, 8, 8]
    '''
    return (type_ + 7) // 8


cdef uint32_t const_type(uint64_t value) nogil:
    if value >= (1ull << 16):
        if value >= (1ull << 32):
            return 64
        else:
            return 32
    elif value >= (1ull << 8):
        return 16
    else:
        return 8


cdef uint64_t type_max(uint32_t type) nogil:
    return (2ull << (type - 1)) - 1


class Expr:
    IS_ADD = False
    IS_ALSO = False
    IS_AND = False
    IS_BINARY = False
    IS_CAST = False
    IS_COMPARE = False
    IS_COND = False
    IS_CONST = False
    IS_DIV = False
    IS_LSHIFT = False
    IS_MOD = False
    IS_MUL = False
    IS_RSHIFT = False
    IS_SHIFT = False
    IS_TEMPVAR = False
    IS_VAR = False
    __slots__ = '__has_table',

    def __new__(cls, *args, **kwargs):
        self = super(Expr, cls).__new__(cls)
        self.__has_table = None
        return self

    def __str__(self):
        raise NotImplementedError

    def __format__(self, spec):
        '''
        specs:
        U: Can safely rely on implicit type promotion
        '''
        return self.__str__()

    def statics(self, vs):
        return ''.join(filter(None, (x.statics(vs) for x in self.children)))

    children = ()
    rtype = None

    @property
    def optimized(self):
        '''Optimize the expression

        The property should never modify self, but instead return new optimized
        instances if any optimization is applicable.
        '''
        return self

    @property
    def force_optimized(self):
        self.__dict__['optimized'] = self
        return self

    def walk(self):
        """Recursively visit itself and all children."""
        return cutils.walk(self)

    def walk_tempvar(self):
        """Shortcut for filter(lambda x: x.IS_TEMPVAR, self.walk())
        """
        return (x for x in self.walk() if x.IS_TEMPVAR)

    def _complicated(self, int threshold) -> bool:
        for expr in self.walk():
            threshold -= 1
            if not threshold:
                return True
        return False

    def has_table(self):
        '''Recursively checks whether the expression contains ExprTable.
        If true, the expression can be unsafe to extract in ExprCond
        '''
        res = self.__has_table
        if res is None:
            res = False
            for expr in self.children:
                if expr.has_table:
                    res = True
                    break
            self.__has_table = res
        return res

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        for subexpr in self.children:
            subexpr.extract_subexprs(threshold, callback, allow_extract_table)

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

    # Whether the expression is likely a predicate
    # (i.e. only returns 0 or 1)
    is_predicate = False

    # Overhead of this expression only (not counting children)
    overhead = 0
    static_bytes = 0


class ExprVar(Expr):
    IS_VAR = True

    def __init__(self, type):
        super().__init__()
        self.rtype = type

    def _complicated(self, threshold):
        return False


class ExprFixedVar(ExprVar):
    def __init__(self, type, name):
        super().__init__(type)
        self.name = name

    def __str__(self):
        return self.name


class ExprTempVar(ExprVar):
    IS_TEMPVAR = True
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
    IS_CONST = True
    _POOL_SIZE = 4096
    _pool = [None] * _POOL_SIZE
    _pool_lock = threading.Lock()
    __slots__ = 'optimized', 'rtype', 'value'

    def __new__(cls, int type, value):
        if type == 32 and 0 <= value < cls._POOL_SIZE:
            self = cls._pool[value]
            if self is None:
                with cls._pool_lock:
                    self = cls._pool[value]
                    if self is None:
                        self = super(ExprConst, cls).__new__(cls)
                        super(ExprConst, self).__init__()
                        self.rtype = type
                        self.value = int(value)
                        self.optimized = self
                        cls._pool[value] = self
            return self

        self = super(ExprConst, cls).__new__(cls)
        self.rtype = type
        self.value = int(value)
        self.optimized = self
        return self

    def __str(self, omit_type=False):
        value = self.value
        rtype = self.rtype
        if rtype % 8 == 0:
            value &= type_max(rtype)
        if -16 <= value <= 16:
            value_s = str(value)
        else:
            value_s = hex(value)
        if not omit_type:
            if self.rtype < 64:
                value_s += 'u'
            else:
                value_s = f'UINT64_C({value_s})'
        return value_s

    def __str__(self):
        return self.__str()

    def __format__(self, spec):
        return self.__str(omit_type='U' in spec)

    @staticmethod
    def combine(const_exprs):
        """Combine multiple ExprConst into one."""
        if len(const_exprs) == 1:
            expr = const_exprs[0]
            if expr.value:
                return expr
            else:
                return None

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
        return Const(self.rtype, -self.value)

    def _complicated(self, threshold):
        return False

    @property
    def overhead(self):
        return (<int>self.rtype > 32) + 1


class ExprAdd(Expr):
    IS_ADD = True

    def __init__(self, exprs, const_):
        super().__init__()
        assert const_ is None or const_.IS_CONST
        self.exprs = tuple(expr.optimized for expr in exprs)
        self.const = const_
        rtype = max([x.rtype for x in self.children])
        self.rtype = max(rtype, 31)  # C type-promotion rule

    def __str__(self):
        res = ' + '.join(map(str, self.exprs))
        const_ = self.const
        if const_:
            left_rtype = max(expr.rtype for expr in self.exprs)
            const_value = const_.value
            const_rtype = const_.rtype
            if const_value >= 0:
                if left_rtype >= const_rtype:
                    res += f' + {const_:U}'
                else:
                    res += f' + {const_}'
            else:
                abs_const = Const(const_rtype, -const_value)
                if left_rtype >= 31 and left_rtype >= const_rtype:
                    res += f' - {abs_const:U}'
                else:
                    res += f' - {abs_const}'
        return '(' + res + ')'

    @property
    def children(self):
        const_ = self.const
        if const_:
            return self.exprs + (const_,)
        else:
            return self.exprs

    @cutils.cached_property
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

        const_ = ExprConst.combine(const_exprs)

        # (a ? c1 : c2) + c3 ==> (a ? c1 + c3 : c2 + c3)
        if const_:
            const_value = const_.value
            for i, expr in enumerate(exprs):
                if expr.IS_COND and \
                        expr.exprT.IS_CONST and expr.exprF.IS_CONST:
                    expr = ExprCond(
                        expr.cond,
                        Const(self.rtype, expr.exprT.value + const_value),
                        Const(self.rtype, expr.exprF.value + const_value))
                    exprs[i] = expr.optimized
                    const_ = None
                    break

        if len(exprs) == 1 and not const_:
            return exprs[0]

        cdef Py_ssize_t n_exprs
        cdef Py_ssize_t j
        if const_ is self.const:
            self_exprs = self.exprs
            n_exprs = len(exprs)
            if n_exprs == len(self_exprs):
                for j in range(n_exprs):
                    if exprs[j] is self_exprs[j]:
                        pass
                    else:
                        break
                else:
                    return self

        return ExprAdd(exprs, const_).force_optimized

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        exprs = []
        for expr in self.exprs:
            expr.extract_subexprs(threshold, callback, allow_extract_table)
            expr = callback(expr, allow_extract_table,
                            expr._complicated(threshold))
            exprs.append(expr)
        self.exprs = tuple(exprs)
        # Don't bother to do callback on self.const; it's never required

    @cutils.cached_property
    def overhead(self):
        n = len(self.exprs)
        if self.const:
            n += 1
        return (n - 1) * 2


class ExprBinary(Expr):
    IS_BINARY = True

    def __init__(self, left, right, rtype=None):
        super().__init__()
        self.left = left = left.optimized
        self.right = right = right.optimized
        self.rtype = rtype or max(31, left.rtype, right.rtype)

    @property
    def children(self):
        return self.left, self.right

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.left = callback(self.left, allow_extract_table,
                             self.left._complicated(threshold))
        self.right = callback(self.right, allow_extract_table,
                              self.right._complicated(threshold))

    overhead = 2


class ExprShift(ExprBinary):
    IS_SHIFT = True

    def __init__(self, left, right):
        super().__init__(left, right, max(31, left.rtype))


class ExprLShift(ExprShift):
    IS_LSHIFT = True

    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self.right
        if right.IS_CONST:
            right_value = right.value
            if right_value in (1, 2, 3):
                return '{} * {}'.format(self.left, 1 << right_value)
            return '({} << {})'.format(self.left, right_value)
        else:
            return '({} << {})'.format(self.left, right)

    @cutils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        if right_const and left.IS_CONST:
            return Const(self.rtype, left.value << right.value)

        # "(a & c1) << c2" ==> (a << c2) & (c1 << c2) (where c2 <= 3)
        # This takes advantage of x86's LEA instruction
        if right_const and right.value <= 3 and \
                left.IS_AND and \
                left.right.IS_CONST:
            expr_left = ExprLShift(left.left, right)
            expr_right = Const(left.right.rtype,
                               left.right.value << right.value)
            return ExprAnd(expr_left, expr_right).optimized

        # PREDICATE << c  ==>  PREDICATE ? (1 << c) : 0
        if right_const and left.is_predicate:
            expr = ExprCond(left,
                            Const(self.rtype, 1 << right.value),
                            Const(self.rtype, 0))
            return expr.optimized

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
                expr = ExprLShift(left.left, Const(32, c2 - c1))
            elif c2 == c1:
                expr = left.left
            else:
                expr = ExprRShift(left.left, Const(32, c1 - c2))
            and_value = ((1 << c2) - 1) ^ ((1 << expr.rtype) - 1)
            expr = ExprAnd(expr, Const(expr.rtype, and_value))
            return expr.optimized

        # "(a + c1) << c2" ==> (a << c2) + (c1 << c2)
        if right_const and \
                left.IS_ADD and len(left.exprs) == 1 and \
                left.const:
            expr_left = ExprLShift(left.exprs[0], right)
            expr_right = Const(left.const.rtype,
                               left.const.value << right.value)
            return ExprAdd((expr_left,), expr_right).optimized

        return self


class ExprRShift(ExprShift):
    IS_RSHIFT = True

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

    @cutils.cached_property
    def optimized(self):
        '''
        >>> expr = (Add(FixedVar(32, 'c'), 30) >> 2)
        >>> str(expr.optimized)
        '(((c + 2) >> 2) + 7)'
        >>> expr = (Add(FixedVar(32, 'c'), FixedVar(32, 'd'), -30) >> 2)
        >>> str(expr.optimized)
        '(((c + d + 2) >> 2) - 8)'
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

                expr = ExprAdd(left.exprs, Const(ctype, remainder))
                expr = ExprRShift(expr, Const(32, c2))
                expr = ExprAdd((expr,), Const(ctype, compensation))
                return expr.optimized

        # (a >> c1) >> c2 ==> a >> (c1 + c2)
        if right_const and \
                left.IS_RSHIFT and \
                left.right.IS_CONST:
            right = Add(right, left.right).optimized
            left = left.left
            return ExprRShift(left, right).optimized

        return self


class ExprMul(ExprBinary):
    IS_MUL = True

    def __str__(self):
        left = self.left
        right = self.right
        if left.rtype >= right.rtype:
            return f'({left} * {right:U})'
        else:
            return f'({left} * {right})'

    @cutils.cached_property
    def optimized(self):
        left = self.left
        right = self.right

        right_const = right.IS_CONST

        # Both constants
        if right_const and left.IS_CONST:
            return Const(self.rtype, left.value * right.value)

        # Put constant on the right side
        if not right_const and left.IS_CONST:
            return ExprMul(right, left).optimized

        if right_const:
            # Strength reduction (* => <<)
            rv = right.value
            if rv == 0:
                return Const(32, 0)
            elif rv == 1:
                return left
            elif is_pow2(rv):
                expr = ExprLShift(left, Const(32, rv.bit_length() - 1))
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

            # PREDICATE * c ==> PREDICATE ? c : 0
            if left.is_predicate:
                return ExprCond(left, right, Const(self.rtype, 0)).optimized

        return self

    @cutils.cached_property
    def overhead(self):
        if self.left.IS_CONST or self.right.IS_CONST:
            return 2
        else:
            return 4


class ExprDiv(ExprBinary):
    IS_DIV = True

    def __init__(self, left, right):
        if left.rtype < 32 or left.rtype == 63:
            left = ExprCast(max(32, left.rtype + 1), left)
        super().__init__(left, right)

    def __str__(self):
        left = self.left
        right = self.right
        if left.rtype >= right.rtype:
            return f'({left} / {right:U})'
        else:
            return f'({left} / {right})'

    @cutils.cached_property
    def optimized(self):
        left = self.left
        right = self.right
        if right.IS_CONST:
            rv = right.value
            if rv == 0:
                raise ZeroDivisionError
            elif rv == 1:
                return left
            elif is_pow2(rv):
                expr = ExprRShift(left, Const(32, rv.bit_length() - 1))
                return expr.optimized

        return self

    overhead = 7


class ExprMod(ExprBinary):
    IS_MOD = True

    def __str__(self):
        left = self.left
        right = self.right
        if left.rtype >= right.rtype:
            return f'({left} % {right:U})'
        else:
            return f'({left} % {right})'

    @cutils.cached_property
    def optimized(self):
        right = self.right
        if right.IS_CONST:
            value = right.value
            if value and (value & (value - 1)) == 0:
                return ExprAnd(self.left,
                               Const(right.rtype, value - 1)).optimized
        return self

    overhead = 7


class ExprAnd(ExprBinary):
    IS_AND = True

    def __str__(self):
        left = self.left
        right = self.right
        if left.rtype >= right.rtype:
            return f'({left} & {right:U})'
        else:
            return f'({left} & {right})'

    @cutils.cached_property
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
                left = ExprAdd(left.exprs, Const(left.const.rtype, c1p))
                return ExprAnd(left, right).optimized

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

    @cutils.cached_property
    def is_predicate(self):
        return self.right.IS_CONST and self.right.value == 1

    @cutils.cached_property
    def overhead(self):
        if self.left.IS_RSHIFT and self.right.IS_CONST and \
                self.right.value == 1:
            # Bit-test, reduce overhead counting
            return 0
        return 2


class ExprCompare(ExprBinary):
    IS_COMPARE = True

    __negate = {
        '==': '!=',
        '!=': '==',
        '>': '<=',
        '<': '>=',
        '>=': '<',
        '<=': '>',
    }

    is_predicate = True

    def __init__(self, left, compare, right):
        super().__init__(left, right, 31)
        self.compare = compare

    def __str__(self):
        return f'({self.left:U} {self.compare} {self.right:U})'

    @cutils.cached_property
    def negated(self):
        neg = ExprCompare(self.left, self.__negate[self.compare], self.right)
        neg.__dict__['negated'] = self
        return neg

    @cutils.cached_property
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
                expr = ExprAdd((left.left,), Const(32, -(c2 << c1)))
                expr = ExprCompare(expr, '<', Const(32, 1 << c1))
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
                expr = ExprCompare(left.left, '<', Const(32, c2 << c1))
                return expr.optimized

        return self


class ExprCast(Expr):
    IS_CAST = True
    __slots__ = 'rtype', 'value', '__optimized'

    def __new__(cls, type, value):
        self = super(ExprCast, cls).__new__(cls)
        self.rtype = type
        self.value = value.optimized
        self.__optimized = None
        return self

    def __str__(self):
        return '{}({})'.format(type_name(self.rtype),
                               utils.trim_brackets(str(self.value)))

    @property
    def children(self):
        return self.value,

    @property
    def optimized(self):
        res = self.__optimized
        if res is None:
            rtype = self.rtype
            value = self.value

            if value.rtype == rtype:
                res = value
            elif value.IS_CAST and rtype <= value.rtype:
                res = ExprCast(rtype, value.value).optimized
            else:
                res = self

            self.__optimized = res
        return res

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.value = callback(self.value, allow_extract_table,
                              self.value._complicated(threshold))

    @property
    def is_predicate(self):
        return self.value.is_predicate

    overhead = 0


class ExprCond(Expr):
    IS_COND = True

    def __init__(self, cond, exprT, exprF):
        super().__init__()
        self.cond = cond.optimized
        self.exprT = exprT.optimized
        self.exprF = exprF.optimized
        self.rtype = max(31, self.exprT.rtype, self.exprF.rtype)

    def __str__(self):
        return '({} ? {} : {})'.format(self.cond, self.exprT, self.exprF)

    @property
    def children(self):
        return self.cond, self.exprT, self.exprF

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        if not self.cond.IS_VAR:
            self.cond.extract_subexprs(threshold, callback,
                                       allow_extract_table)
            self.cond = callback(self.cond, allow_extract_table,
                                 self.cond._complicated(threshold))

        # Tables in exprT and exprF cannot be safely extracted
        allow_extract_table = False

        self.exprT.extract_subexprs(threshold, callback, allow_extract_table)
        self.exprF.extract_subexprs(threshold, callback, allow_extract_table)
        self.exprT = callback(self.exprT, allow_extract_table,
                              self.exprT._complicated(threshold))
        self.exprF = callback(self.exprF, allow_extract_table,
                              self.exprF._complicated(threshold))

    @cutils.cached_property
    def optimized(self):
        cond = self.cond
        exprT = self.exprT
        exprF = self.exprF

        # cast(a cmp b) ? T : F ==> (a cmp b) ? T : F
        # cast(a & 1) ? T : F ==> (a & 1) ? T : F
        while cond.IS_CAST and \
                (cond.value.IS_COMPARE or
                 (cond.value.IS_AND and cond.value.right.IS_CONST and
                  cond.value.right.value == 1)):
            cond = cond.value
        if cond is not self.cond:
            return ExprCond(cond, exprT, exprF).optimized

        # cond ? 1 : 0 ==> Cast(cond)
        if (cond.IS_COMPARE and exprT.IS_CONST and exprF.IS_CONST and
                exprT.value == 1 and exprF.value == 0):
            return ExprCast(self.rtype, cond).optimized

        # cond is (** != 0) or (** == 0)
        if cond.IS_COMPARE and cond.right.IS_CONST and cond.right.value == 0:
            if cond.compare == '!=':
                return ExprCond(cond.left, exprT, exprF).optimized
            elif cond.compare == '==':
                return ExprCond(cond.left, exprF, exprT).optimized

        # Sometimes we might want to negate thecomparator
        if cond.IS_COMPARE:
            # (1) If both exprT and exprF are constants, swap smaller ones
            #     to right to increase readability.
            # (2) If one of exprT is constant and the other is not, swap
            #     the constant to left to increase readability.
            if (
                    (exprT.IS_CONST and exprF.IS_CONST and
                     exprT.value < exprF.value) or
                    (exprF.IS_CONST and not exprT.IS_CONST)):
                return ExprCond(cond.negated, exprF, exprT).optimized

        return self

    overhead = 3


class ExprTable(Expr):
    has_table = True
    __slots__ = 'rtype', 'name', 'values', 'var', 'offset', '__optimized'

    def __new__(cls, type, name, values, var, offset):
        self = super(ExprTable, cls).__new__(cls)
        self.rtype = type
        self.name = name
        self.values = values
        self.var = var.optimized
        self.offset = offset
        self.__optimized = None
        return self

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
        line_start_format = '  /* {{:#0{}x}} */'.format(indlen).format
        value_format = ' {{:#0{}x}},'.format(maxlen).format

        line = 'alignas({type}) const {type} {name}[{size:#x}] = {{'.format(
            type=type_name(self.rtype), name=self.name, size=self.values.size)
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

    @property
    def optimized(self):
        res = self.__optimized
        if res is None:
            self.__optimized = res = self.__optimized_impl()
        return res

    def __optimized_impl(self):
        var = self.var
        # If var contains a cast, it's usually unnecessary
        while var.IS_CAST and var.value.rtype < var.rtype:
            var = var.value.optimized
        if var is not self.var:
            return ExprTable(self.rtype, self.name, self.values, var,
                             self.offset).optimized

        # Absorb constants into offset
        if var.IS_ADD and var.const:
            offset = self.offset - var.const.value
            var = ExprAdd(var.exprs, None).optimized
            return ExprTable(self.rtype, self.name, self.values, var,
                             offset).optimized

        return self

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.var = callback(self.var, allow_extract_table,
                            self.var._complicated(threshold))

    def _complicated(self, _threshold):
        return True

    # Bytes taken by the table is independently calculated
    overhead = 2

    @property
    def static_bytes(self):
        return self.values.size * type_bytes(self.rtype)


### Factory functions
@cython.nonecheck(False)
cdef exprize(expr):
    '''Convert int to ExprConst'''
    if type(expr) is int:
        return ExprConst(32, expr)
    else:
        return expr


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
        return const_expr or Const(32, 0)
    elif len(exprs) == 1 and not const_expr:
        return exprs[0]
    else:
        return ExprAdd(exprs, const_expr)


def Cast(type, value):
    return ExprCast(type, exprize(value))


def Cond(cond, exprT, exprF):
    return ExprCond(exprize(cond), exprize(exprT), exprize(exprF))
