#!/usr/bin/env python3
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

from dataclasses import dataclass, replace


__all__ = ['Options', 'OPT_Os', 'OPT_O2', 'OPT_O3', 'OPT_O4', 'OPT_DEFAULT']


@dataclass
class Options:
    linear_threshold: int
    const_threshold: int
    hole_threshold: int
    group_threshold: int
    overhead_multiply: int = 2

    def allow_complex_data_manipulation(self) -> 'Options':
        '''
        Allow complex data manipulation without increasing the number of
        branches.'''
        return replace(self, overhead_multiply=2)


# Prefere smaller code size
OPT_Os = Options(64, 32, 24, 3, 2)

# Prefer a balance
OPT_O2 = Options(128, 96, 64, 3, 8)

# Agressively optimize for performance
OPT_O3 = Options(256, 192, 128, 3, 16)

# Even more aggressive
OPT_O4 = Options(512, 384, 256, 3, 32)

OPT_DEFAULT = OPT_O2
