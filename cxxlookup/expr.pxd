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


from cpython.object cimport PyObject
from libc.stdint cimport int8_t, int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector


cdef str type_name(uint32_t type)
cdef uint32_t const_type(uint64_t value) nogil
cdef uint64_t type_max(uint32_t type) nogil


cdef class Expr:
    cdef public uint32_t rtype

    cdef int8_t _has_table
    cdef Expr _optimized

    cdef c_bool has_table(self)
    cdef Expr optimize(self)
    cdef Expr do_optimize(self)
    cdef force_optimized(self)
    cdef children(self)
    cdef statics(self, vs)
    cdef c_bool is_predicate(self)
    cdef uint32_t overhead(self)
    cdef uint32_t static_bytes(self)

    cdef vector[PyObject*] walk_dedup_fast(self)


cdef class ExprVar(Expr):
    pass


cdef class ExprFixedVar(ExprVar):
    cdef public str name


cdef class ExprTempVar(ExprVar):
    cdef public uint32_t var


cdef str get_temp_var_name(uint32_t var)


cdef class ExprConst(Expr):
    # Store a Python int so that int64_t and uint64_t are both handled
    # properly.  Improve in the future
    cdef public object value
    cdef uint32_t overhead(self)
    cdef ExprConst negated(self)
    cdef __str(self, c_bool omit_type)


cdef Const(uint32_t type, value)


cdef class ExprAdd(Expr):
    cdef public object exprs
    cdef public ExprConst konst

    cdef children(self)
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef Expr AddMany(exprs)
cdef Expr Add(expr0, expr1)


cdef class ExprBinary(Expr):
    cdef public Expr left
    cdef public Expr right

    cdef children(self)
    cdef uint32_t overhead(self)


cdef class ExprShift(ExprBinary):
    pass


cdef class ExprLShift(ExprShift):
    cdef Expr do_optimize(self)


cdef ExprLShift LShift(a, b)


cdef class ExprRShift(ExprShift):
    cdef Expr do_optimize(self)


cdef ExprRShift RShift(a, b)


cdef class ExprMul(ExprBinary):
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef ExprMul Mul(a, b)


cdef class ExprDiv(ExprBinary):
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef ExprDiv Div(a, b)


cdef class ExprMod(ExprBinary):
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef class ExprAnd(ExprBinary):
    cdef c_bool is_predicate(self)
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef ExprAnd And(a, b)


cdef class ExprCompare(ExprBinary):
    cdef public str compare
    cdef ExprCompare _negated

    cdef c_bool is_predicate(self)
    cdef Expr do_optimize(self)
    cdef ExprCompare negated(self)


cdef ExprCompare Eq(a, b)
cdef ExprCompare Ne(a, b)
cdef ExprCompare Gt(a, b)
cdef ExprCompare Ge(a, b)
cdef ExprCompare Lt(a, b)
cdef ExprCompare Le(a, b)


cdef class ExprCast(Expr):
    cdef public Expr value

    cdef children(self)
    cdef Expr do_optimize(self)
    cdef c_bool is_predicate(self)


cdef ExprCast Cast(uint32_t type, value)


cdef class ExprNeg(Expr):
    cdef public Expr value

    cdef children(self)
    cdef Expr do_optimize(self)


cdef ExprNeg Neg(expr)


cdef class ExprCond(Expr):
    cdef public Expr cond
    cdef public Expr exprT
    cdef public Expr exprF

    cdef children(self)
    cdef uint32_t overhead(self)
    cdef Expr do_optimize(self)


cdef ExprCond Cond(cond, exprT, exprF)


cdef class ExprTable(Expr):
    cdef public str name
    cdef object values
    cdef public Expr var
    cdef int32_t offset

    cdef Expr do_optimize(self)
    cdef uint32_t overhead(self)
    cdef uint32_t static_bytes(self)


cdef Expr Sub(expr0, expr1)
