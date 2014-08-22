/*
Copyright (c) 2014, chys <admin@CHYS.INFO>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of chys <admin@CHYS.INFO> nor the names of other
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

static uint32_t do_gcd_many(const uint32_t *p, size_t n) {
	uint32_t res = 0;
	for (; n; --n) {
		uint32_t v = *p++;
		if (res < v) {
			uint32_t t = v;
			v = res;
			res = t;
		}
		while (v) {
			uint32_t t = res % v;
			res = v;
			v = t;
		}
		if (res == 1)
			break;
	}
	return res;
}

// _speedups.gcd_many(array.tostring())
static PyObject *gcd_many(PyObject *self, PyObject *args) {
	Py_buffer buf;
	if (!PyArg_ParseTuple(args, "y*", &buf))
		return NULL;

	const uint32_t *p = (const uint32_t *)buf.buf;
	Py_ssize_t len = buf.len;

	return PyLong_FromLong(do_gcd_many(p, (size_t)len / 4));
}

static PyMethodDef speedups_methods[] = {
	{"gcd_many",  &gcd_many, METH_VARARGS,
		"Calculate greatest common divisor (GCD) of a uint32_t array"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef speedups_module = {
	PyModuleDef_HEAD_INIT,
	"_speedups",
	NULL,
	0,
	speedups_methods
};

PyMODINIT_FUNC
PyInit__speedups(void) {
	return PyModule_Create(&speedups_module);
}
