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


cimport cython
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector
from numpy cimport ndarray

from .pyx_helpers cimport Frac, make_frac, make_frac_fast

@cython.profile(False)
cdef inline c_bool is_pow2(int64_t v) nogil:
    return v > 0 and not (v & (v - 1))


cdef float linregress_slope(uint32_t[::1] y)

# Frac operations
cdef Frac double_as_frac(double)
cdef Frac limit_denominator(Frac, uint64_t max_denominator) nogil


cdef class LinearReduceResult:
    cdef ndarray reduced_values
    cdef int64_t offset


cdef LinearReduceResult linear_reduce(uint32_t[::1] values, Frac slope)


cdef class PrepareStrideResult:
    cdef ndarray base_values
    cdef ndarray delta
    cdef uint32_t delta_max
    cdef uint32_t delta_nonzeros


cdef PrepareStrideResult prepare_strides(uint32_t[::1] values, uint32_t stride)
