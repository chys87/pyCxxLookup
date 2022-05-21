#!/usr/bin/env python3
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

# Copyright (c) 2014, 2016, chys <admin@CHYS.INFO>
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

from typing import Sequence

from . import codegen
from .codegen import COMMON_HEADERS
from .options import Options, OPT_DEFAULT
from .test import run_test, TestError
from . import utils


__all__ = ['TestError', 'COMMON_HEADERS', 'CxxLookup', 'make']


class CxxLookup:
    def __init__(self, func_name: str, base: int, values: Sequence[int],
                 hole: int | None = None, opt: Options = OPT_DEFAULT):
        self._func_name = func_name
        self._base = base
        self._values = utils.make_numpy_array(values)
        if hole is None:
            hole = int(utils.most_common_element(self._values))
        self._hole = hole
        self._opt = opt

    def make_code(self) -> str:
        return self.code

    @utils.cached_property
    @utils.profiling
    def code(self) -> str:
        return codegen.make_code(self._func_name, self._base, self._values,
                                 self._hole, self._opt)

    def test(self, cxx_name : str | None = None) -> None:
        run_test(self._func_name, self._base, self._values, self._hole,
                 COMMON_HEADERS, self.make_code(),
                 cxx_name=cxx_name)


def make(func_name: str, base: int, values: Sequence[int],
         hole: int | None = None, opt: Options = OPT_DEFAULT) -> str:
    obj = CxxLookup(func_name, base, values, hole, opt)
    obj.test()
    return obj.make_code()
