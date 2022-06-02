# distutils: language=c++
# cython: language_level=3
# cython: profile=True
# cython: cdivision=True


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
from cpython.object cimport PyObject, PyTypeObject, Py_TYPE
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector

from . import cutils
from . import utils
from . import _speedups

from numpy cimport ndarray

from .pyx_helpers cimport bit_length, flat_hash_set, is_pow2


# Signed types only allowed for intermediate values
# Unsigned: Number of bits
# Signed: Number of bits - 1
# E.g.: 31 = int32_t; 32 = uint32_t
cpdef str type_name(uint32_t type):
    '''
    >>> type_name(7), type_name(32)
    ('int8_t', 'uint32_t')
    '''
    if (type & 1):
        return f'int{type + 1}_t'
    else:
        return f'uint{type}_t'


cpdef uint32_t type_bytes(uint32_t type_) nogil:
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


cdef class Expr:
    def __cinit__(self, *args):
        self._has_table = -1
        self._optimized = None

    # Let's avoid __init__ as much as possible, so that we may try to
    # use Type.__new__(Type, *args) to create objects

    def __str__(self):
        raise NotImplementedError

    def __format__(self, spec):
        '''
        specs:
        U: Can safely rely on implicit type promotion
        '''
        return self.__str__()

    cdef statics(self, vs):
        return ''.join(filter(None,
                              ((<Expr>x).statics(vs)
                               for x in self.children())))

    cdef tuple children(self):
        return ()

    @cython.final
    cdef Expr optimize(self):
        res = self._optimized
        if res is None:
            try:
                self._optimized = res = self.do_optimize()
            except Exception:
                import sys
                print('Optimization error: ', self, file=sys.stderr)
                raise
        return res

    cdef Expr do_optimize(self):
        '''Optimize the expression

        This method should never modify self, but instead return new optimized
        instances if any optimization is applicable.
        '''
        return self

    @cython.final
    cdef force_optimized(self):
        self._optimized = self
        return self

    def walk_tempvar(self):
        """Shortcut for filter(lambda x: type(x) is ExprTempVar, self.walk())
        """
        cdef vector[PyObject*] descendants = self.walk_dedup_fast()
        cdef PyObject* x
        for x in descendants:
            expr = <object>x
            if type(expr) is ExprTempVar:
                yield expr

    def _complicated(self, int threshold) -> bool:
        if threshold <= 0:
            return True
        cdef vector[PyObject*] descendants = self.walk_dedup_fast()
        return descendants.size() >= <uint32_t>threshold

    @cython.final
    cdef c_bool has_table(self):
        '''Recursively checks whether the expression contains ExprTable.
        If true, the expression can be unsafe to extract in ExprCond
        '''
        cdef c_bool res
        if self._has_table < 0:
            res = False
            for expr in self.children():
                if (<Expr>expr).has_table():
                    res = True
                    break
            self._has_table = res
        return self._has_table

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        for subexpr in self.children():
            subexpr.extract_subexprs(threshold, callback, allow_extract_table)

    # Whether the expression is likely a predicate
    # (i.e. only returns 0 or 1)
    cdef c_bool is_predicate(self):
        return False

    # Overhead of this expression only (not counting children)
    cdef uint32_t overhead(self, uint32_t multiply):
        return 0

    @cython.final
    cdef vector[PyObject*] walk_dedup_fast(self):
        '''Walk over self and all descendants, returning vector[PyObject*]
        '''
        cdef flat_hash_set[PyObject*] visited
        cdef vector[PyObject*] res

        cdef PyObject* nodep = <PyObject*>self
        res.push_back(nodep)
        visited.insert(nodep)

        cdef size_t i = 0

        while i < res.size():
            nodep = res[i]
            i += 1
            for child in (<Expr><object>nodep).children():
                nodep = <PyObject*>(child)
                if not visited.contains(nodep):
                    visited.insert(nodep)
                    res.push_back(nodep)

        return std_move(res)


cdef class ExprVar(Expr):
    def _complicated(self, threshold):
        return False


@cython.final
cdef class ExprFixedVar(ExprVar):
    def __cinit__(self, uint32_t type, str name):
        self.rtype = type
        self.name = name
        self._has_table = 0

    def __str__(self):
        return self.name


cdef list __temp_var_name_cache = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def _test_temp_var_name(uint32_t var):
    '''
    >>> list(map(_test_temp_var_name, (0, 26, 52)))
    ['A', 'AA', 'BA']
    '''
    return get_temp_var_name(var)

cdef str get_temp_var_name(uint32_t var):
    global __temp_var_name_cache
    cdef str s
    try:
        s = __temp_var_name_cache[var]
    except IndexError:
        __temp_var_name_cache += [None] * (
            var + 1 - len(__temp_var_name_cache))
    else:
        if s is not None:
            return s

    cdef uint32_t length = 1
    cdef uint32_t expressible = 26

    cdef uint32_t ind = var
    while ind >= expressible:
        length += 1
        ind -= expressible
        expressible *= 26

    s = ''
    for _ in range(length):
        s = chr(ord('A') + (ind % 26)) + s
        ind //= 26
    __temp_var_name_cache[var] = s
    return s


@cython.final
cdef class ExprTempVar(ExprVar):
    def __cinit__(self, uint32_t type, uint32_t var):
        self.rtype = type
        self.var = var
        self._has_table = 0

    def __str__(self):
        return get_temp_var_name(self.var)


DEF CONST_POOL_SIZE = 4096
cdef __const_pool = [None] * CONST_POOL_SIZE
cdef __const_pool_lock = threading.Lock()


cdef ExprConst Const(uint32_t type, value):
    cdef uint32_t idx
    cdef ExprConst self
    if type == 32 and 0 <= value < CONST_POOL_SIZE:
        idx = value
        self = __const_pool[idx]
        if self is None:
            with __const_pool_lock:
                self = __const_pool[idx]
                if self is None:
                    self = ExprConst.__new__(ExprConst, type, value)
                    __const_pool[idx] = self
        return self
    return ExprConst.__new__(ExprConst, type, value)


@cython.final
cdef class ExprConst(Expr):
    def __cinit__(self, uint32_t type, value):
        self.rtype = type
        self.value = value
        self._optimized = self
        self._has_table = 0

    cdef __str(self, c_bool omit_type):
        value = self.value
        cdef uint32_t rtype = self.rtype
        if rtype % 8 == 0:
            value &= type_max(rtype)
        if -16 <= value <= 16:
            value_s = str(value)
        else:
            value_s = hex(value)
        if not omit_type:
            if rtype < 64:
                value_s += 'u'
            else:
                value_s = f'UINT64_C({value_s})'
        return value_s

    def __str__(self):
        return self.__str(omit_type=False)

    def __format__(self, spec):
        return self.__str(omit_type='U' in spec)

    cdef ExprConst negated(self):
        return Const(self.rtype, -self.value)

    def _complicated(self, threshold):
        return False

    cdef uint32_t overhead(self, uint32_t multiply):
        return ((self.rtype > 32) + 1) * multiply


@cython.boundscheck(False)
cdef ExprConst CombineConsts(const_exprs):
    """Combine multiple ExprConst into one."""
    cdef ExprConst expr
    if len(const_exprs) == 1:
        expr = const_exprs[0]
        if expr.value == 0:
            return None
        else:
            return expr

    const_value = 0
    cdef uint32_t const_type = 32
    for expr in const_exprs:
        const_value += expr.value
        const_type = max(const_type, expr.rtype)
    if const_value == 0:
        return None
    else:
        return Const(const_type, const_value)


@cython.final
cdef class ExprAdd(Expr):
    def __init__(self, exprs, ExprConst const_):
        cdef uint32_t rtype = 0
        if const_ is not None:
            if const_.value == 0:
                const_ = None
            else:
                rtype = max(rtype, const_.rtype)
        cdef Expr x
        for x in exprs:
            rtype = max(rtype, x.rtype)
        if len(exprs) + <int>(const_ is not None) > 1:
            rtype = max(rtype, 31)  # C type-promotion rule
        self.rtype = rtype
        self.exprs = tuple([(<Expr>x).optimize() for x in exprs])
        self.konst = const_

    def __str__(self):
        res = ' + '.join(map(str, self.exprs))
        cdef ExprConst const_ = self.konst
        cdef uint32_t const_rtype
        if const_ is not None:
            left_rtype = max([(<Expr>expr).rtype for expr in self.exprs])
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

    cdef tuple children(self):
        cdef ExprConst const_ = self.konst
        if const_ is not None:
            return self.exprs + (const_,)
        else:
            return self.exprs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Expr do_optimize(self):
        exprs = []
        const_exprs = []

        if self.konst:
            const_exprs.append(self.konst)

        cdef Expr expr
        cdef PyTypeObject* expr_type
        for expr in self.exprs:
            expr = expr.optimize()
            expr_type = Py_TYPE(expr)
            if expr_type == <PyTypeObject*>ExprAdd:
                exprs.extend((<ExprAdd>expr).exprs)
                if (<ExprAdd>expr).konst is not None:
                    const_exprs.append((<ExprAdd>expr).konst)
            elif expr_type == <PyTypeObject*>ExprConst:
                const_exprs.append(expr)
            else:
                exprs.append(expr)

        cdef ExprConst const_ = CombineConsts(const_exprs)

        # (a ? c1 : c2) + c3 ==> (a ? c1 + c3 : c2 + c3)
        if const_ is not None:
            const_value = const_.value
            for i, expr in enumerate(exprs):
                if type(expr) is ExprCond and \
                        type((<ExprCond>expr).exprT) is ExprConst and \
                        type((<ExprCond>expr).exprF) is ExprConst:
                    expr = ExprCond(
                        (<ExprCond>expr).cond,
                        Const(self.rtype,
                              (<ExprConst>(<ExprCond>expr).exprT).value +
                              const_value),
                        Const(self.rtype,
                              (<ExprConst>(<ExprCond>expr).exprF).value +
                              const_value))
                    exprs[i] = expr.optimize()
                    const_ = None
                    break

        if len(exprs) == 1 and const_ is None:
            return exprs[0]

        if len(exprs) == 0:
            if const_ is not None:
                return const_
            else:
                return Const(self.rtype, 0)

        cdef Py_ssize_t n_exprs
        cdef Py_ssize_t j
        if const_ is self.konst:
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

        return ExprAdd(exprs, const_).force_optimized()

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        exprs = []
        for expr in self.exprs:
            expr.extract_subexprs(threshold, callback, allow_extract_table)
            expr = callback(expr, allow_extract_table,
                            expr._complicated(threshold))
            exprs.append(expr)
        self.exprs = tuple(exprs)
        # Don't bother to do callback on self.konst; it's never required

    cdef uint32_t overhead(self, uint32_t multiply):
        cdef int n = len(self.exprs)
        if self.konst is not None:
            n += 1
        return (n - 1) * 2 * multiply


cdef class ExprBinary(Expr):
    def __init__(self, Expr left, Expr right, uint32_t rtype=0):
        self.left = left.optimize()
        self.right = right.optimize()
        self.rtype = rtype or max(31, self.left.rtype, self.right.rtype)

    cdef tuple children(self):
        return self.left, self.right

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.left = callback(self.left, allow_extract_table,
                             self.left._complicated(threshold))
        self.right = callback(self.right, allow_extract_table,
                              self.right._complicated(threshold))

    cdef uint32_t overhead(self, uint32_t multiply):
        return 2 * multiply


cdef class ExprShift(ExprBinary):
    def __init__(self, Expr left, Expr right):
        super().__init__(left, right, max(31, left.rtype))


@cython.final
cdef class ExprLShift(ExprShift):
    def __str__(self):
        # Avoid the spurious 'u' after the constant
        cdef Expr right = self.right
        if type(right) is ExprConst:
            right_value = (<ExprConst>right).value
            if right_value in (1, 2, 3):
                return '{} * {}'.format(self.left, 1 << right_value)
            return '({} << {})'.format(self.left, right_value)
        else:
            return '({} << {})'.format(self.left, right)

    cdef Expr do_optimize(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        cdef Expr expr

        cdef PyTypeObject* left_type = Py_TYPE(left)
        cdef PyTypeObject* right_type = Py_TYPE(right)
        cdef c_bool right_const = (right_type == <PyTypeObject*>ExprConst)

        if right_const and left_type == <PyTypeObject*>ExprConst:
            return Const(self.rtype,
                         (<ExprConst>left).value << (<ExprConst>right).value)

        # "(a & c1) << c2" ==> (a << c2) & (c1 << c2) (where c2 <= 3)
        # This takes advantage of x86's LEA instruction
        if right_const and (<ExprConst>right).value <= 3 and \
                type(left) is ExprAnd and \
                type((<ExprAnd>left).right) is ExprConst:
            expr_left = ExprLShift((<ExprAnd>left).left, right)
            expr_right = Const(
                (<ExprConst>(<ExprAnd>left).right).rtype,
                (<ExprConst>(<ExprAnd>left).right).value <<
                    (<ExprConst>right).value)
            return ExprAnd(expr_left, expr_right).optimize()

        # PREDICATE << c  ==>  PREDICATE ? (1 << c) : 0
        if right_const and left.is_predicate():
            expr = ExprCond(left,
                            Const(self.rtype, 1 << (<ExprConst>right).value),
                            Const(self.rtype, 0))
            return expr.optimize()

        # (cond ? c1 : c2) << c3 ==> (cond ? c1 << c3 : c2 << c3)
        if right_const and \
                type(left) is ExprCond and \
                type((<ExprCond>left).exprT) is ExprConst and \
                type((<ExprCond>left).exprF) is ExprConst:
            expr = ExprCond((<ExprCond>left).cond,
                            ExprLShift((<ExprCond>left).exprT, right),
                            ExprLShift((<ExprCond>left).exprF, right))
            return expr.optimize()

        # (a >> c1) << c2
        # (a << (c2 - c1)) & ~((1 << c2) - 1)   (c2 > c1)
        # (a >> (c1 - c2)) & ~((1 << c2) - 1)   (c2 <= c1)
        if right_const and \
                type(left) is ExprRShift and \
                type((<ExprRShift>left).right) is ExprConst:
            c2 = (<ExprConst>right).value
            c1 = (<ExprConst>(<ExprRShift>left).right).value
            if c2 > c1:
                expr = ExprLShift((<ExprRShift>left).left, Const(32, c2 - c1))
            elif c2 == c1:
                expr = (<ExprRShift>left).left
            else:
                expr = ExprRShift((<ExprRShift>left).left, Const(32, c1 - c2))
            and_value = ((1 << c2) - 1) ^ ((2ull << (expr.rtype - 1)) - 1)
            expr = ExprAnd(expr, Const(expr.rtype, and_value))
            return expr.optimize()

        # "(a + c1) << c2" ==> (a << c2) + (c1 << c2)
        if right_const and \
                type(left) is ExprAdd and len((<ExprAdd>left).exprs) == 1 and \
                (<ExprAdd>left).konst is not None:
            expr_left = ExprLShift((<ExprAdd>left).exprs[0], right)
            expr_right = Const((<ExprAdd>left).konst.rtype,
                               (<ExprAdd>left).konst.value <<
                                (<ExprConst>right).value)
            return ExprAdd((expr_left,), expr_right).optimize()

        return self


cdef ExprLShift LShift(a, b):
    return ExprLShift(exprize(a), exprize(b))


@cython.final
cdef class ExprRShift(ExprShift):
    def __init__(self, Expr left, Expr right):
        # Always logical shift
        if left.rtype < 32 or left.rtype == 63:
            left = ExprCast(max(32, left.rtype + 1), left)
        super().__init__(left, right)

    def __str__(self):
        # Avoid the spurious 'u' after the constant
        right = self.right
        if type(right) is ExprConst:
            right_s = str((<ExprConst>right).value)
        else:
            right_s = str(right)

        return '({} >> {})'.format(self.left, right_s)

    cdef Expr do_optimize(self):
        cdef Expr left = self.left
        cdef Expr right = self.right

        cdef c_bool right_const = type(right) is ExprConst

        # (a + c1) >> c2
        # Convert to ((a + c1 % (1 << c2)) >> c2) + (c1 >> c2).
        if right_const and type(left) is ExprAdd and (<ExprAdd>left).konst:
            ctype = (<ExprAdd>left).konst.rtype
            c1 = (<ExprAdd>left).konst.value
            c2 = (<ExprConst>right).value

            if c1 >> c2:

                compensation = c1 >> c2
                remainder = c1 - (compensation << c2)
                if remainder < 0:
                    compensation += 1
                    remainder -= 1 << c2

                expr = ExprAdd((<ExprAdd>left).exprs, Const(ctype, remainder))
                expr = ExprRShift(expr, Const(32, c2))
                expr = ExprAdd((expr,), Const(ctype, compensation))
                return expr.optimize()

        # (a >> c1) >> c2 ==> a >> (c1 + c2)
        if right_const and \
                type(left) is ExprRShift and \
                type((<ExprRShift>left).right) is ExprConst:
            right = Add(right, (<ExprRShift>left).right).optimize()
            left = (<ExprRShift>left).left
            return ExprRShift(left, right).optimize()

        return self


cdef ExprRShift RShift(a, b):
    return ExprRShift(exprize(a), exprize(b))


@cython.final
cdef class ExprMul(ExprBinary):
    def __str__(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        if left.rtype >= right.rtype:
            return f'({left} * {right:U})'
        else:
            return f'({left} * {right})'

    cdef Expr do_optimize(self):
        cdef Expr expr
        cdef Expr left = self.left
        cdef Expr right = self.right
        cdef c_bool right_const = type(right) is ExprConst
        cdef c_bool left_const = type(left) is ExprConst

        # Both constants
        if right_const and left_const:
            return Const(
                self.rtype,
                (<ExprConst>left).value * (<ExprConst>right).value)

        # Put constant on the right side
        if not right_const and left_const:
            return ExprMul(right, left).optimize()

        if right_const:
            # Strength reduction (* => <<)
            rv = (<ExprConst>right).value
            if rv == 0:
                return Const(32, 0)
            elif rv == 1:
                return left
            elif rv > 0 and is_pow2(rv):
                expr = ExprLShift(left, Const(32, bit_length(rv) - 1))
                return expr.optimize()

            # (a + c1) * c2 ==> (a * c2 + c1 * c2)
            if type(left) is ExprAdd and \
                    len((<ExprAdd>left).exprs) == 1 and \
                    (<ExprAdd>left).konst:
                expr_left = ExprMul((<ExprAdd>left).exprs[0], right)
                expr_right = ExprMul((<ExprAdd>left).konst, right)
                return ExprAdd((expr_left, expr_right), None).optimize()

            # (cond ? c1 : c2) * c3 ==> (cond ? c1 * c3 : c2 * c3)
            if type(left) is ExprCond and \
                    type((<ExprCond>left).exprT) is ExprConst and \
                    type((<ExprCond>left).exprF) is ExprConst:
                expr = ExprCond((<ExprCond>left).cond,
                                ExprMul((<ExprCond>left).exprT, right),
                                ExprMul((<ExprCond>left).exprF, right))
                return expr.optimize()

            # PREDICATE * c ==> PREDICATE ? c : 0
            if left.is_predicate():
                return ExprCond(left, right, Const(self.rtype, 0)).optimize()

        return self

    cdef uint32_t overhead(self, uint32_t multiply):
        if type(self.left) is ExprConst or type(self.right) is ExprConst:
            return 2 * multiply
        else:
            return 4 * multiply


cdef ExprMul Mul(a, b):
    return ExprMul(exprize(a), exprize(b))


@cython.final
cdef class ExprDiv(ExprBinary):
    def __init__(self, Expr left, Expr right):
        if left.rtype < 32 or left.rtype == 63:
            left = ExprCast(max(32, left.rtype + 1), left)
        super().__init__(left, right)

    def __str__(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        if left.rtype >= right.rtype:
            return f'({left} / {right:U})'
        else:
            return f'({left} / {right})'

    cdef Expr do_optimize(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        if type(right) is ExprConst:
            rv = (<ExprConst>right).value
            if rv == 0:
                raise ZeroDivisionError
            elif rv == 1:
                return left
            elif rv > 0 and is_pow2(rv):
                expr = ExprRShift(left, Const(32, bit_length(rv) - 1))
                return expr.optimize()

        return self

    cdef uint32_t overhead(self, uint32_t multiply):
        return 7 * multiply


cdef ExprDiv Div(a, b):
    return ExprDiv(exprize(a), exprize(b))


@cython.final
cdef class ExprMod(ExprBinary):
    def __str__(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        if left.rtype >= right.rtype:
            return f'({left} % {right:U})'
        else:
            return f'({left} % {right})'

    cdef Expr do_optimize(self):
        cdef Expr right = self.right
        if type(right) is ExprConst:
            value = (<ExprConst>right).value
            if value and (value & (value - 1)) == 0:
                return ExprAnd(self.left,
                               Const(right.rtype, value - 1)).optimize()
        return self

    cdef uint32_t overhead(self, uint32_t multiply):
        return 7 * multiply


@cython.final
cdef class ExprAnd(ExprBinary):
    def __init__(self, Expr left, Expr right):
        if type(right) is ExprConst and (<ExprConst>right).value == 0:
            raise ValueError(f'{left} & {right}')
        super().__init__(left, right)

    def __str__(self):
        cdef Expr left = self.left
        cdef Expr right = self.right
        if left.rtype >= right.rtype:
            return f'({left} & {right:U})'
        else:
            return f'({left} & {right})'

    cdef Expr do_optimize(self):
        cdef Expr left = self.left
        cdef Expr right = self.right

        cdef c_bool right_const = type(right) is ExprConst
        right_value = None

        if right_const:
            right_value = (<ExprConst>right).value

        # (a + c1) & c2 ==> (a + c1') & c2
        # where c1' = c1 with high bits cleared
        if right_const and right_value and \
                type(left) is ExprAdd and (<ExprAdd>left).konst:
            rv = right_value
            bt = bit_length(rv) + 1
            c1 = (<ExprAdd>left).konst.value
            c1p = c1 & ((1 << bt) - 1)
            # If its high bit is set, make it negative
            if c1p & (1 << (bt - 1)):
                c1p |= ~((1 << bt) - 1)
            if c1p != c1:
                left = ExprAdd((<ExprAdd>left).exprs,
                               Const((<ExprAdd>left).konst.rtype, c1p))
                return ExprAnd(left, right).optimize()

        # (a & c1) & c2 ==> a & (c1 & c2)
        if right_const and \
                type(left) is ExprAnd and \
                type((<ExprAnd>left).right) is ExprConst:
            c1 = (<ExprConst>(<ExprAnd>left).right).value
            c2 = right_value
            expr = ExprAnd((<ExprAnd>left).left,
                           Const(max((<ExprAnd>left).right.rtype, right.rtype),
                               c1 & c2))
            return expr.optimize()

        # (a & 0xff) ==> (uint8_t)a
        # (a & 0xffff) ==> (uint16_t)a
        if right_const:
            # Must cast back
            if right_value == 0xff:
                expr = ExprCast(self.rtype, ExprCast(8, left))
                return expr.optimize()
            elif right_value == 0xffff:
                expr = ExprCast(self.rtype, ExprCast(16, left))
                return expr.optimize()

        return self

    cdef c_bool is_predicate(self):
        return (type(self.right) is ExprConst and \
                (<ExprConst>self.right).value == 1)

    cdef uint32_t overhead(self, uint32_t multiply):
        if type(self.left) is ExprRShift and \
                type(self.right) is ExprConst and \
                (<ExprConst>self.right).value == 1:
            # Bit-test, reduce overhead counting
            return 0
        return 2 * multiply


cdef ExprAnd And(a, b):
    return ExprAnd(exprize(a), exprize(b))


# We depend on the order. (op ^ 1) is the opposite of op
DEF COMP_EQ = 0
DEF COMP_NE = 1
DEF COMP_GT = 2
DEF COMP_LE = 3
DEF COMP_GE = 4
DEF COMP_LT = 5

cdef COMP_STR = ['==', '!=', '>', '<=', '>=', '<']


@cython.profile(False)
cdef inline uint8_t comp_negate(uint8_t op) nogil:
    return op ^ 1


@cython.final
cdef class ExprCompare(ExprBinary):
    cdef c_bool is_predicate(self):
        return False

    def __cinit__(self, *args):
        self._negated = None

    def __init__(self, Expr left, uint8_t compare, Expr right):
        super().__init__(left, right, 31)
        self.compare = compare

    def __str__(self):
        return f'({self.left:U} {COMP_STR[self.compare]} {self.right:U})'

    cdef ExprCompare negated(self):
        cdef ExprCompare neg = self._negated
        if neg is None:
            neg = ExprCompare(self.left, comp_negate(self.compare), self.right)
            neg._negated = self
            self._negated = neg
        return neg

    cdef Expr do_optimize(self):
        cdef Expr left = self.left
        cdef Expr right = self.right

        cdef c_bool right_const = type(right) is ExprConst

        # (a >> c1) == c2
        # a >= (c2 << c1) && a < ((c2 + 1) << c1)
        # unsinged(a - (c2 << c1)) < (1 << c1)
        if right_const and self.compare == COMP_EQ and \
                type(left) is ExprRShift and \
                (<ExprRShift>left).left.rtype == 32 and \
                type((<ExprRShift>left).right) is ExprConst and \
                right.rtype == 32:
            c1 = (<ExprConst>(<ExprRShift>left).right).value
            c2 = (<ExprConst>right).value
            if ((c2 + 1) << c1) <= 2**32:
                expr = ExprAdd(((<ExprRShift>left).left,),
                               Const(32, -(c2 << c1)))
                expr = ExprCompare(expr, COMP_LT, Const(32, 1 << c1))
                return expr.optimize()

        # (a >> c1) < c2
        # a < (c2 << c1)
        if right_const and self.compare == COMP_LT and \
                type(left) is ExprRShift and \
                (<ExprRShift>left).left.rtype == 32 and \
                type((<ExprRShift>left).right) is ExprConst and \
                right.rtype == 32:
            c1 = (<ExprConst>(<ExprRShift>left).right).value
            c2 = (<ExprConst>right).value
            if (c2 << c1) < 2**32:
                expr = ExprCompare(
                    (<ExprRShift>left).left, COMP_LT, Const(32, c2 << c1))
                return expr.optimize()

        return self


cdef ExprCompare Eq(a, b): return ExprCompare(exprize(a), COMP_EQ, exprize(b))
cdef ExprCompare Ne(a, b): return ExprCompare(exprize(a), COMP_NE, exprize(b))
cdef ExprCompare Gt(a, b): return ExprCompare(exprize(a), COMP_GT, exprize(b))
cdef ExprCompare Ge(a, b): return ExprCompare(exprize(a), COMP_GE, exprize(b))
cdef ExprCompare Lt(a, b): return ExprCompare(exprize(a), COMP_LT, exprize(b))
cdef ExprCompare Le(a, b): return ExprCompare(exprize(a), COMP_LE, exprize(b))


cdef class ExprUnary(Expr):
    @cython.final
    cdef tuple children(self):
        return self.value,

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.value = callback(self.value, allow_extract_table,
                              self.value._complicated(threshold))


@cython.final
cdef class ExprCast(ExprUnary):
    def __cinit__(self, uint32_t type, Expr value):
        self.rtype = type
        self.value = value.optimize()

    def __str__(self):
        return '{}({})'.format(type_name(self.rtype),
                               utils.trim_brackets(str(self.value)))

    cdef Expr do_optimize(self):
        cdef uint32_t rtype = self.rtype
        cdef Expr value = self.value
        cdef Expr res

        if value.rtype == rtype:
            res = value.optimize()
        elif type(value) is ExprCast and rtype <= value.rtype:
            res = ExprCast(rtype, (<ExprCast>value).value).optimize()
        else:
            res = self

        return res

    cdef c_bool is_predicate(self):
        return self.value.is_predicate()

    # overhead = 0


cdef ExprCast Cast(uint32_t type, value):
    return ExprCast.__new__(ExprCast, type, exprize(value))


@cython.final
cdef class ExprNeg(ExprUnary):
    def __cinit__(self, Expr value):
        self.rtype = max(31, value.rtype)
        self.value = value.optimize()

    def __str__(self):
        value_s = utils.trim_brackets(str(self.value))
        return f'-({value_s})'

    cdef Expr do_optimize(self):
        cdef Expr value = self.value
        cdef Expr res
        if type(value) is ExprNeg:
            res = (<ExprNeg>value).value.optimize()
        elif type(value) is ExprConst:
            res = (<ExprConst>value).negated().optimize()
        else:
            res = self
        return res

    # overhead = 0


cdef ExprNeg Neg(expr):
    return ExprNeg.__new__(ExprNeg, exprize(expr))


@cython.final
cdef class ExprCond(Expr):
    def __cinit__(self, Expr cond, Expr exprT, Expr exprF):
        self.cond = cond.optimize()
        self.exprT = exprT.optimize()
        self.exprF = exprF.optimize()
        self.rtype = max(31, self.exprT.rtype, self.exprF.rtype)

    def __str__(self):
        return '({} ? {} : {})'.format(self.cond, self.exprT, self.exprF)

    cdef tuple children(self):
        return self.cond, self.exprT, self.exprF

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        if not isinstance(self.cond, ExprVar):
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

    cdef Expr do_optimize(self):
        cdef Expr cond = self.cond
        cdef Expr exprT = self.exprT
        cdef Expr exprF = self.exprF

        # cast(PREDICATE) ? T : F ==> (PREDICATE) ? T : F
        while type(cond) is ExprCast and (<ExprCast>cond).value.is_predicate():
            cond = (<ExprCast>cond).value
        if cond is not self.cond:
            return ExprCond(cond, exprT, exprF).optimize()

        # PREDICATE ? 1 : 0 ==> Cast(PREDICATE)
        if (cond.is_predicate() and \
                type(exprT) is ExprConst and \
                type(exprF) is ExprConst and
                (<ExprConst>exprT).value == 1 and
                (<ExprConst>exprF).value == 0):
            return ExprCast(self.rtype, cond).optimize()

        # cond is (** != 0) or (** == 0)
        cdef ExprCompare cond_comp
        if type(cond) is ExprCompare:
            cond_comp = <ExprCompare>cond
        if type(cond) is ExprCompare and \
                type((<ExprCompare>cond_comp).right) is ExprConst and \
                (<ExprConst>(<ExprCompare>cond_comp).right).value == 0:
            if cond_comp.compare == COMP_NE:
                return ExprCond(cond_comp.left, exprT, exprF).optimize()
            elif cond_comp.compare == COMP_EQ:
                return ExprCond(cond_comp.left, exprF, exprT).optimize()

        # Sometimes we might want to negate thecomparator
        if type(cond) is ExprCompare:
            # (1) If both exprT and exprF are constants, swap smaller ones
            #     to right to increase readability.
            # (2) If one of exprT is constant and the other is not, swap
            #     the constant to left to increase readability.
            if (
                    (type(exprT) is ExprConst is type(exprF) and
                     (<ExprConst>exprT).value < (<ExprConst>exprF).value) or
                    (type(exprF) is ExprConst is not type(exprT))):
                return ExprCond(cond_comp.negated(),
                                exprF, exprT).optimize()

        return self

    cdef uint32_t overhead(self, uint32_t multiply):
        return 3 * multiply


cdef ExprCond Cond(cond, exprT, exprF):
    return ExprCond.__new__(ExprCond,
                            exprize(cond), exprize(exprT), exprize(exprF))


@cython.final
cdef class ExprTable(Expr):
    def __cinit__(self, uint32_t type, str name, ndarray values, Expr var,
                  int32_t offset):
        self._has_table = 1
        self.rtype = type
        self.name = name
        self.values = values
        self.var = var.optimize()
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

    cdef statics(self, vs):
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

    cdef tuple children(self):
        return self.var,

    cdef Expr do_optimize(self):
        cdef Expr var = self.var
        # If var contains a cast, it's usually unnecessary
        while type(var) is ExprCast and \
                (<ExprCast>var).value.rtype < var.rtype:
            var = (<ExprCast>var).value.optimize()
        if var is not self.var:
            return ExprTable(self.rtype, self.name, self.values, var,
                             self.offset).optimize()

        # Absorb constants into offset
        if type(var) is ExprAdd and (<ExprAdd>var).konst:
            offset = self.offset - (<ExprAdd>var).konst.value
            var = ExprAdd((<ExprAdd>var).exprs, None).optimize()
            # Occasionally we get very big or small offset.
            # Just cast it to int64_t and truncate to int32_t
            return ExprTable(self.rtype, self.name, self.values, var,
                             <int32_t><int64_t>offset).optimize()

        return self

    def extract_subexprs(self, threshold, callback, allow_extract_table):
        super().extract_subexprs(threshold, callback, allow_extract_table)
        self.var = callback(self.var, allow_extract_table,
                            self.var._complicated(threshold))

    def _complicated(self, _threshold):
        return True

    # Bytes taken by the table is independently calculated
    cdef uint32_t overhead(self, uint32_t multiply):
        return 2 * multiply + self.values.size * type_bytes(self.rtype)


### Factory functions
@cython.nonecheck(False)
cdef Expr exprize(expr):
    '''Convert int to ExprConst'''
    if type(expr) is int:
        return Const(32, expr)
    else:
        return expr


cdef Expr AddMany(in_exprs):
    exprs = []
    const_exprs = []

    cdef PyTypeObject* etype

    cdef Expr expr
    for x in in_exprs:
        expr = exprize(x)
        etype = Py_TYPE(expr)
        if etype == <PyTypeObject*>ExprConst:
            const_exprs.append(expr)
        elif etype == <PyTypeObject*>ExprAdd:
            exprs.extend((<ExprAdd>expr).exprs)
            if (<ExprAdd>expr).konst:
                const_exprs.append((<ExprAdd>expr).konst)
        else:
            exprs.append(expr)

    const_expr = CombineConsts(const_exprs)
    if len(exprs) == 0:
        return const_expr or Const(32, 0)
    elif len(exprs) == 1 and not const_expr:
        return exprs[0]
    else:
        return ExprAdd(exprs, const_expr)


cdef Expr Add(expr0, expr1):
    return AddMany((expr0, expr1))


cdef Expr Sub(expr0, expr1):
    return Add(expr0, Neg(expr1))
