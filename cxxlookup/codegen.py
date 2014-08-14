#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80

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

import string

import numpy as np


COMMON_HEADERS = \
    "#include <stdint.h>\n"


def make_numpy_array(values):
    return np.array(values, dtype=np.uint32)


def most_common_element(arr):
    """Return the most common element of a numpy array"""
    u, indices = np.unique(arr, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def make_code(base, values, hole):
    # FIXME: The current version is unusable in practice.
    code = 'inline uint32_t lookup(uint32_t c) noexcept {\n'
    code += '\tswitch (c) {\n'
    for i, v in enumerate(values):
        code += '\tcase {}: return {};\n'.format(base + i, v)
    code += '\tdefault: return {};\n'.format(hole)
    code += '\t}\n'
    code += '}\n'
    return code


WRAP_TEMPLATE = string.Template(r'''namespace {
namespace CxxLookup_$func_name {

$code

} // namespace CxxLookup_$func_name
} // namespace

uint32_t $func_name(uint32_t c) noexcept {
    return CxxLookup_$func_name::lookup(c);
}''')


def wrap_code(func_name, code):
    return WRAP_TEMPLATE.substitute(func_name=func_name, code=code)
