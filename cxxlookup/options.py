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


class Options:
    __slots__ = ('linear_threshold', 'const_threshold', 'hole_threshold',
                 'split_threshold', 'group_threshold', 'overhead_multiply')
    def __init__(self,
                 linear_threshold,
                 const_threshold,
                 hole_threshold,
                 split_threshold,
                 group_threshold,
                 overhead_multiply):
        self.linear_threshold = linear_threshold
        self.const_threshold = const_threshold
        self.hole_threshold = hole_threshold
        self.split_threshold = split_threshold
        self.group_threshold = group_threshold
        self.overhead_multiply = overhead_multiply

    def __str__(self):
        return 'cxxlookup.Options({})'.format(', '.join(map(str, [
            self.linear_threshold,
            self.const_threshold,
            self.hole_threshold,
            self.split_threshold,
            self.group_threshold,
            self.overhead_multiply])))

    __repr__ = __str__


# Prefere smaller code size
OPT_Os = Options(64, 32, 24, 128, 3, 4)

# Prefer a balance
OPT_O2 = Options(256, 96, 64, 512, 3, 8)

# Agressively optimize for performance
OPT_O3 = Options(512, 192, 128, 1024, 3, 16)

# Even more aggressive
OPT_O4 = Options(1024, 384, 256, 2048, 3, 32)

OPT_DEFAULT = OPT_O2
