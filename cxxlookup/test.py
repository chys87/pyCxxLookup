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

import os
from string import Template
import subprocess
import tempfile

import numpy as np


class TestError(Exception):
    pass


CODE_TEMPLATE = Template(r'''$headers
#include <stdio.h>
#include <stdlib.h>

namespace {

$code

} // namespace

int main(void) {
    uint32_t *out = new uint32_t[$hi - $lo];
    for (uint32_t i = $lo; i < $hi; ++i)
        out[i - $lo] = $func_name(i);
    fwrite(out, sizeof(uint32_t), $hi - $lo, stdout);
    delete[] out;
    return 0;
}
''')


def check_result(exe_name, cwd, base, values):
    test_proc = subprocess.Popen([os.path.join('.', exe_name)],
                                 stdout=subprocess.PIPE,
                                 cwd=cwd)
    test_output = test_proc.stdout.read()
    rc = test_proc.wait()
    if rc != 0:
        if rc > 0:
            raise TestError("Process returned with code {}".format(rc))
        else:
            raise TestError("Process terminated by signal {}".format(-rc))

    if len(test_output) != values.size * 4:
        raise TestError("Incorrect number of data in test result "
                        "(Expected {} bytes; Got {} bytes)".format(
                            values.size * 4, len(test_output)))

    got_values = np.fromstring(test_output, dtype=np.uint32)

    unequal = (values != got_values)

    if np.any(unequal):
        k = unequal.tostring().find(b'\x01')
        raise TestError("[{}]: Expected {}; Got {}".format(base + k,
                                                           values[k],
                                                           got_values[k]))


def _run_test(func_name, base, values, hole, headers, code,
              cwd, src_name, exe_name):

    src = CODE_TEMPLATE.substitute(headers=headers,
                                   code=code,
                                   func_name=func_name,
                                   lo=base,
                                   hi=base + values.size)

    with open(os.path.join(cwd, src_name), 'w') as f:
        f.write(src)

    rc = subprocess.call(['g++', '-O2', '-std=gnu++11', '-s',
                          '-o', exe_name, src_name],
                         cwd=cwd)
    if rc != 0:
        raise TestError("Compilation failed")

    check_result(exe_name, cwd, base, values)


def run_test(func_name, base, values, hole, headers, code, cxx_name=None):
    if cxx_name:
        tmpdir = tempfile.gettempdir()
        exe_name = os.path.join(
            tmpdir, 'pyCxxLookup-test.{}.exe'.format(os.getlogin()))
        try:
            _run_test(func_name, base, values, hole, headers, code,
                      tmpdir, cxx_name, exe_name)
        finally:
            try:
                os.unlink(exe_name)
            except FileNotFoundError:
                pass
    else:
        with tempfile.TemporaryDirectory(prefix='pyCxxLookup') as dirname:
            _run_test(func_name, base, values, hole, headers, code,
                      dirname, 'test.cpp', 'test.exe')
