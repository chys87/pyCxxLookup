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


from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp.utility cimport pair


cdef extern from "<cmath>" nogil:
    float abs(float)
    double abs(double)
    long long int abs(long long int)
    long int abs(long int)
    int abs(int)


cdef extern from "pyx_helpers.h" namespace "cxxlookup" nogil:
    uint32_t bit_length(uint64_t)

    uint64_t gcd(uint64_t, uint64_t)

    cdef cppclass Frac:
        int64_t numerator
        uint64_t denominator
        double to_double()

    Frac make_frac(int64_t, uint64_t)
    Frac make_frac_fast(int64_t, uint64_t)


cdef extern from "<absl/container/flat_hash_set.h>" namespace "absl" nogil:
    cdef cppclass flat_hash_set[T,HASH=*,PRED=*,ALLOCATOR=*]:
        ctypedef T value_type
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass
        flat_hash_set() except +
        flat_hash_set(flat_hash_set&) except +
        #flat_hash_set(key_compare&)
        #flat_hash_set& operator=(flat_hash_set&)
        iterator begin()
        const_iterator const_begin "begin"()
        void clear()
        size_t count(const T&)
        bint contains(const T&)
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        pair[iterator, iterator] equal_range(T&)
        pair[const_iterator, const_iterator] const_equal_range "equal_range"(T&)
        iterator erase(iterator)
        iterator erase(iterator, iterator)
        size_t erase(T&)
        iterator find(T&)
        const_iterator const_find "find"(T&)
        pair[iterator, bint] insert(T&)
        iterator insert(iterator, T&)
        #key_compare key_comp()
        iterator insert(iterator, iterator)
        iterator lower_bound(T&)
        const_iterator const_lower_bound "lower_bound"(T&)
        size_t max_size()
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        size_t size()
        void swap(flat_hash_set&)
        iterator upper_bound(T&)
        const_iterator const_upper_bound "upper_bound"(T&)
        #value_compare value_comp()
        void max_load_factor(float)
        float max_load_factor()
        void rehash(size_t)
        void reserve(size_t)
        size_t bucket_count()
        size_t max_bucket_count()
        size_t bucket_size(size_t)
        size_t bucket(const T&)
