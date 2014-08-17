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
        tester = Tester(args.tempdir)
        test_wcwidth(tester)


if __name__ == '__main__':
    main()
