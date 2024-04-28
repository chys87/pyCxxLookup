#!/usr/bin/env python3
# coding: utf-8
# vim: set ts=4 sts=4 sw=4 expandtab cc=80:

# Copyright (c) 2014-2024, chys <admin@CHYS.INFO>
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

from pathlib import Path
from setuptools import setup, Extension
import shutil
import os
import sys

from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np


def main():
    if 'linux' in sys.platform:
        DEFAULT_CFLAGS = \
            '-O3 -march=native -pthread '\
            '-fvisibility-inlines-hidden -fvisibility=hidden -flto'
        os.environ.setdefault('CFLAGS', DEFAULT_CFLAGS)
        os.environ.setdefault('CXXFLAGS', DEFAULT_CFLAGS + ' -std=gnu++20')

        if shutil.which('clang++') and shutil.which('clang'):
            os.environ['CC'] = 'clang'
            os.environ['CXX'] = 'clang++'
            os.environ['LDSHARED'] = 'clang++ -shared'


    ext_modules = [
        Extension('cxxlookup._speedups', ['cxxlookup/_speedups.cpp'],
                  include_dirs=[np.get_include()]),
        Extension('cxxlookup.cutils', ['cxxlookup/cutils.pyx'],
                  include_dirs=[np.get_include()]),
        Extension('cxxlookup.expr', ['cxxlookup/expr.pyx'],
                  include_dirs=[np.get_include()]),
        Extension('cxxlookup.codegen', ['cxxlookup/codegen.pyx'],
                  include_dirs=[np.get_include()]),
    ]

    Options.error_on_unknown_names = True
    Options.cache_builtins = True
    Options.buffer_max_dims = 2

    setup(name='cxxlookup',
          version='1.0',
          author='chys',
          author_email='admin@CHYS.INFO',
          url='https://github.com/chys87/pyCxxLookup',
          license='BSD',
          description='Generate C++ lookup functions with Python 3.',
          packages=['cxxlookup'],
          ext_modules=cythonize(ext_modules,
                                language_level=3,
                                # Don't build in tree
                                build_dir=Path("cython_build")),
          package_data={'cxxlookup': ['py.typed']},
          include_package_data=True,
          zip_safe=False,
          install_requires=[
              'numpy',
          ])


if __name__ == '__main__':
    main()
