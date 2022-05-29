from libc.stdint cimport int64_t, uint32_t, uint64_t


cdef str type_name(uint32_t type)
cdef uint32_t const_type(uint64_t value) nogil
cdef uint64_t type_max(uint32_t type) nogil
