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

import argparse
import os
import random
import subprocess
import sys
import tempfile

from cxxlookup import CxxLookup, TestError


def static_check():
    # Pyflakes
    rc_pyflakes = subprocess.call(['pyflakes', '.'])

    # PEP-8 check
    rc_pep8 = subprocess.call(['pep8', '.'])

    if rc_pyflakes != 0 or rc_pep8 != 0:
        sys.exit(1)


class Tester:
    def __init__(self, tempdir):
        self._tempdir = tempdir or os.path.join(
            tempfile.gettempdir(), 'pyCxxLookupTest.{}'.format(os.getlogin()))
        try:
            os.mkdir(self._tempdir)
        except FileExistsError:
            pass

    def __call__(self, name, values, base=0, hole=None):
        print('Running test', name)
        cxx_name = os.path.join(self._tempdir, name + '.cpp')
        cl = CxxLookup('test_func', base, values, hole)

        try:
            cl.test(cxx_name=cxx_name)

        except TestError as e:
            print('Failed:', e, file=sys.stderr)


def test_wcwidth(tester):
    import unicodedata
    values = [int(unicodedata.east_asian_width(chr(c)) in 'WF') for c
              in range(0x10000)]
    tester('wcwidth', values)


def test_misc1(tester):
    # Linear
    values = [x * 32 + 170 for x in range(1024)]
    values.extend([0] * 1000)
    values.extend(100000 - 5 * x for x in range(1000))

    # 2 unique values: [a,a,a,a,a,a,b,b,b,b,b,b]
    values.extend([42] * 50)
    values.extend([54] * 54)

    # 2 unique values: [a,a,a,a,a,a,b,b,b,b,b,b,a,a,a,a,a]
    values.extend([0] * 1000)
    values.extend([42] * 5)
    values.extend([54] * 4)
    values.extend([42] * 5)

    # 2 unique values: bit test
    values.extend([0] * 1000)
    values.extend(random.choice((25, 54)) for _ in range(64))

    # Pakced into one single 64-bit integer
    values.extend([0] * 1000)
    values.extend(random.randint(0, 255) for _ in range(8))

    # Mostly linear, but with some outliers
    # (This one should NOT be optimized with "mostly linear" method)
    values.extend([0] * 1000)
    tmp = [x * 32 for x in range(1024)]
    for _ in range(32):
        i = random.randrange(len(tmp))
        tmp[i] = random.randint(0, 2**32 - 1)
    values.extend(tmp)

    # Mostly linear, but with some outliers
    # (This one SHOULD be optimized with "mostly linear" method)
    values.extend([0] * 1000)
    tmp = [x * 2**20 for x in range(1024)]
    for _ in range(32):
        i = random.randrange(len(tmp))
        tmp[i] += random.randint(0, 16383)
    values.extend(tmp)

    # Two level lookup.
    values.extend([0] * 1000)
    tmp = [random.randrange(2**32) for _ in range(100)]
    values.extend(random.choice(tmp) for _ in range(1000))

    # "Compression"
    values.extend([0] * 1000)
    values.extend(random.randrange(2) + 20 for _ in range(200))
    values.extend([0] * 1000)
    values.extend(random.randrange(4) + 20 for _ in range(200))
    values.extend([0] * 1000)
    values.extend(random.randrange(16) + 30 for _ in range(200))

    # "GCD reduce"
    values.extend([0] * 1000)
    values.extend(random.randrange(256) * 2557 + 2554 for _ in range(300))

    # Lo/Hi split
    tmp_values = [random.randrange(65536) for _ in range(256)]
    lo_values = [random.choice(tmp_values) for _ in range(1024)]
    hi_values = [random.choice(tmp_values) for _ in range(1024)]
    values.extend([0] * 1000)
    values.extend(lo + hi * 65536 for (lo, hi) in zip(lo_values, hi_values))

    tester('misc1', values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--static',
                        help='Enable static analysis.',
                        action='store_true', default=False)
    parser.add_argument('--no-unit-tests',
                        help='Don\'t run unit tests.',
                        dest='unit_tests', action='store_false', default=True)
    parser.add_argument('--tempdir',
                        help='Specify temp dir to keep the files')
    args = parser.parse_args()

    if args.static:
        static_check()
    if args.unit_tests:
        random.seed(0)  # Use fixed seed to get repeatable results
        tester = Tester(args.tempdir)
        test_wcwidth(tester)
        test_misc1(tester)


if __name__ == '__main__':
    main()
