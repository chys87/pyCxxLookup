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

import cxxlookup


def static_check():
    # Pyflakes
    rc_pyflakes = subprocess.call(['pyflakes', '.'])

    # PEP-8 check
    rc_pep8 = subprocess.call(['pep8', '.'])

    if rc_pyflakes != 0 or rc_pep8 != 0:
        sys.exit(1)


class Tester:
    def __init__(self, args):
        self._tempdir = args.tempdir or os.path.join(
            tempfile.gettempdir(), 'pyCxxLookupTest.{}'.format(os.getlogin()))
        try:
            os.mkdir(self._tempdir)
        except FileExistsError:
            pass

        self._diff = args.diff

    def run(self, name, test_func):
        print('Preparing test', name, '...')

        res = test_func()
        base = 0
        hole = None
        opt = cxxlookup.OPT_DEFAULT
        if isinstance(res, dict):
            values = res['values']
            base = res.get('base', base)
            hole = res.get('hole', hole)
            opt = res.get('opt', opt)

        else:
            values = res

        cxx_name = os.path.join(self._tempdir, name + '.cpp')
        bak_name = None

        if os.path.exists(cxx_name):
            bak_name = cxx_name + '.bak'
            try:
                os.unlink(bak_name)
            except FileNotFoundError:
                pass
            os.rename(cxx_name, bak_name)

        print('Running test', name, '...')

        cl = cxxlookup.CxxLookup('test_func', base, values, hole=hole, opt=opt)

        try:
            cl.test(cxx_name=cxx_name)

        except cxxlookup.TestError as e:
            print('Failed:', e, file=sys.stderr)
            return

        if self._diff and bak_name:
            subprocess.call(['diff', '-u', bak_name, cxx_name])


TEST_LIST = []


def Testing(func):
    test_name = func.__name__
    if test_name.startswith('test_'):
        test_name = test_name[5:]
    TEST_LIST.append((test_name, func))
    return func


@Testing
def test_wcwidth():
    import unicodedata
    values = [int(unicodedata.east_asian_width(chr(c)) in 'WF') for c
              in range(0x10000)]
    return values


@Testing
def test_misc1():
    random.seed(0)

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

    return values


@Testing
def test_togb18030():
    N = 0x110000
    values = [0] * N
    for k in range(N):
        try:
            gb = chr(k).encode('gb18030')
        except UnicodeEncodeError:
            gb = b''
        gbv = 0
        for c in gb:
            gbv = gbv * 256 + c
        values[k] = gbv

    return {'values': values, 'opt': cxxlookup.OPT_Os}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--static',
                        help='Enable static analysis.',
                        action='store_true', default=False)
    parser.add_argument('-l', '--list-only',
                        help='List tests only.',
                        action='store_true', default=False)
    parser.add_argument('--no-tests',
                        help='Don\'t run tests.',
                        dest='tests', action='store_false', default=True)
    parser.add_argument('-d', '--diff',
                        help='Show diff from last version if possible.',
                        default=False, action='store_true')
    parser.add_argument('--tempdir',
                        help='Specify temp dir to keep the files')
    parser.add_argument('-p', '--profiling', default=False,
                        action='store_true',
                        help='Run Python profiler')
    parser.add_argument('test_names', nargs='*', metavar='TEST_NAME',
                        help='Only run the specified tests.')
    args = parser.parse_args()

    if args.profiling:
        os.environ['pyCxxLookup_Profiling'] = '1'
    else:
        os.environ.pop('pyCxxLookup_Profiling', None)

    if args.static:
        static_check()

    if args.list_only:
        for test_name, test_func in TEST_LIST:
            print(test_name)
        return

    if args.tests:
        if args.test_names:
            test_list = []
            for test_name, test_func in TEST_LIST:
                if test_name in args.test_names:
                    test_list.append((test_name, test_func))

        else:
            test_list = TEST_LIST

        tester = Tester(args)

        for test_name, test_func in test_list:
            tester.run(test_name, test_func)


if __name__ == '__main__':
    main()
