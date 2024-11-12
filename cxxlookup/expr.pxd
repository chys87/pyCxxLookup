# Copyright (c) 2014-2024, chys <admin@CHYS.INFO>
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


cimport cython
from cpython.object cimport PyObject
from libc.stdint cimport *
from libcpp cimport bool as c_bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector


cpdef str type_name(uint32_t type)
cdef uint32_t const_type(uint64_t value) nogil
cdef uint64_t type_max(uint32_t type) nogil


cdef class Expr:
    cdef uint32_t rtype

    cdef int8_t _has_table
    cdef Expr _optimized

    @cython.final
    cdef c_bool has_table(self)
    @cython.final
    cdef Expr optimize(self)
    cdef Expr do_optimize(self)
    @cython.final
    cdef force_optimized(self)
    cdef tuple children(self)
    cdef statics(self, vs)
    cdef c_bool is_predicate(self)
    cdef uint32_t overhead(self, uint32_t multiply)

    @cython.final
    cdef vector[PyObject*] walk_dedup_fast(self)


cdef class ExprVar(Expr):
    pass


cdef class ExprFixedVar(ExprVar):
    cdef str name


cdef class ExprTempVar(ExprVar):
    cdef uint32_t var


cdef str get_temp_var_name(uint32_t var)


cdef class ExprConst(Expr):
    # Store a Python int so that int64_t and uint64_t are both handled
    # properly.  Improve in the future
    cdef object value
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef ExprConst negated(self)
    cdef __str(self, c_bool omit_type)


cdef ExprConst Const(uint32_t type, value)


cdef class ExprAdd(Expr):
    cdef tuple exprs
    cdef ExprConst konst

    cdef tuple children(self)
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef Expr AddMany(exprs)
cdef Expr Add(expr0, expr1)


cdef class ExprBinary(Expr):
    cdef Expr left
    cdef Expr right

    cdef tuple children(self)
    cdef uint32_t overhead(self, uint32_t multiply)


cdef class ExprShift(ExprBinary):
    pass


cdef class ExprLShift(ExprShift):
    cdef Expr do_optimize(self)


cdef ExprLShift LShift(a, b)


cdef class ExprRShift(ExprShift):
    cdef Expr do_optimize(self)


cdef ExprRShift RShift(a, b)


cdef class ExprMul(ExprBinary):
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef ExprMul Mul(a, b)


cdef class ExprDiv(ExprBinary):
    cdef uint64_t _max_dividend

    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef ExprDiv Div(a, b, uint64_t max_dividend = ?)


cdef class ExprMod(ExprBinary):
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef class ExprAnd(ExprBinary):
    cdef c_bool is_predicate(self)
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef ExprAnd And(a, b)


cdef class ExprCompare(ExprBinary):
    cdef ExprCompare _negated
    cdef uint8_t compare

    cdef c_bool is_predicate(self)
    cdef Expr do_optimize(self)
    cdef ExprCompare negated(self)


cdef ExprCompare Eq(a, b)
cdef ExprCompare Ne(a, b)
cdef ExprCompare Gt(a, b)
cdef ExprCompare Ge(a, b)
cdef ExprCompare Lt(a, b)
cdef ExprCompare Le(a, b)


cdef class ExprUnary(Expr):
    cdef Expr value
    cdef tuple children(self)


cdef class ExprCast(ExprUnary):
    cdef Expr do_optimize(self)
    cdef c_bool is_predicate(self)


cdef ExprCast Cast(uint32_t type, value)


cdef class ExprNeg(ExprUnary):
    cdef Expr do_optimize(self)


cdef ExprNeg Neg(expr)


cdef class ExprCond(Expr):
    cdef Expr cond
    cdef Expr exprT
    cdef Expr exprF

    cdef tuple children(self)
    cdef uint32_t overhead(self, uint32_t multiply)
    cdef Expr do_optimize(self)


cdef ExprCond Cond(cond, exprT, exprF)


cdef class ExprTable(Expr):
    cdef str name
    cdef object values
    cdef Expr var
    cdef int32_t offset

    cdef Expr do_optimize(self)
    cdef uint32_t overhead(self, uint32_t multiply)


cdef Expr Sub(expr0, expr1)
