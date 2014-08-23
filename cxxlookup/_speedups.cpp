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
#include <algorithm>
#include <memory>
#include <type_traits>
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {

uint32_t do_gcd_many(const uint32_t *p, size_t n) {
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

int compare_uint32(const void *a, const void *b) {
	uint32_t A = *(const uint32_t *)a;
	uint32_t B = *(const uint32_t *)b;
	if (A < B)
		return -1;
	else if (A == B)
		return 0;
	else
		return 1;
}

size_t do_unique(uint32_t *out, const uint32_t *in, size_t n) {
	uint32_t *po = out;
	if (n >= 64) {
		// Let's try to eliminate some (but not all) duplicates with
		// some simple tricks
		uint64_t hashtable_valid = 0;
		constexpr uint32_t HASHTABLE_SIZE = 61;
		uint32_t hashtable[HASHTABLE_SIZE];

		for (size_t k = n; k; --k) {
			uint32_t v = *in++;
			uint32_t h = v % HASHTABLE_SIZE;
			if (!(hashtable_valid & (1ull << h))) {
				hashtable_valid |= 1ull << h;
				hashtable[h] = v;
				*po++ = v;
			} else if (hashtable[h] != v) {
				hashtable[h] = v;
				*po++ = v;
			}
		}
	} else {
		memcpy(out, in, n * sizeof(uint32_t));
		po = out + n;
	}

	std::sort(out, po);

	return std::unique(out, po) - out;
}

template <typename T>
std::pair<T, size_t> do_mode_cnt(const T *p, size_t n) {

	static_assert(std::is_unsigned<T>::value, "Unsigned integer type only.");

	struct Item {
		T v;
		ssize_t cnt;
	};

	size_t table_size = n * 2;
	if (sizeof(size_t) > 32)
		table_size |= table_size >> 32;
	table_size |= table_size >> 16;
	table_size |= table_size >> 8;
	table_size |= table_size >> 4;
	table_size |= table_size >> 2;
	table_size |= table_size >> 1;

	T N = table_size;

	std::unique_ptr<Item[]> item(new Item[N]);
	memset(&item[0], -1, N * sizeof(Item));

	uint32_t maxval = 0;
	size_t maxcnt = 0;

	for (size_t k = n; k; --k) {
		T v = *p++;
		T h = v % N;

		while (item[h].cnt >= 0 && item[h].v != v) {
			++h;
			if (h >= N)
				h = 0;
		}
		if (item[h].cnt >= 0) {
			item[h].cnt++;
		} else {
			item[h].v = v;
			item[h].cnt = 1;
		}
		if (item[h].cnt > maxcnt) {
			maxcnt = item[h].cnt;
			maxval = v;
		}
	}

	return std::make_pair(maxval, maxcnt);
}

// _speedups.gcd_many(array.tostring())
PyObject *gcd_many(PyObject *self, PyObject *args) {
	Py_buffer buf;
	if (!PyArg_ParseTuple(args, "y*", &buf))
		return NULL;

	const uint32_t *p = (const uint32_t *)buf.buf;
	Py_ssize_t len = buf.len;

	return PyLong_FromLong(do_gcd_many(p, (size_t)len / 4));
}

// np.fromstring(_speedups.unique(array.tostring()))
PyObject *unique(PyObject *self, PyObject *args) {
	Py_buffer buf;
	if (!PyArg_ParseTuple(args, "y*", &buf))
		return NULL;

	const uint32_t *p = (const uint32_t *)buf.buf;
	size_t n = (size_t)buf.len / 4;

	if (n == 0)
		return PyBytes_FromStringAndSize("", 0);

	uint32_t *out = (uint32_t *)malloc(n * sizeof(uint32_t));
	if (out == NULL)
		return PyErr_NoMemory();

	n = do_unique(out, p, n);

	PyObject *res = PyBytes_FromStringAndSize((const char *)out, n * sizeof(uint32_t));
	free(out);
	return res;
}

// _speedups.unique(array.tostring())
PyObject *mode_cnt(PyObject *self, PyObject *args) {
	Py_buffer buf;
	Py_ssize_t n;
	if (!PyArg_ParseTuple(args, "y*n", &buf, &n))
		return NULL;

	if (n == 0)
		Py_RETURN_NONE;

	const uint32_t *p = (const uint32_t *)buf.buf;

	if (buf.len == n * 4) {
		auto res = do_mode_cnt(reinterpret_cast<const uint32_t *>(buf.buf), n);
		return Py_BuildValue("(In)", unsigned(res.first), Py_ssize_t(res.second));
	} else if (buf.len == n * 8) {
		auto res = do_mode_cnt(reinterpret_cast<const uint64_t *>(buf.buf), n);
		return Py_BuildValue("(Ln)", static_cast<PY_LONG_LONG>(int64_t(res.first)), Py_ssize_t(res.second));
	} else {
		Py_RETURN_NONE;
	}
}

PyMethodDef speedups_methods[] = {
	{"gcd_many",  &gcd_many, METH_VARARGS,
		"Calculate greatest common divisor (GCD) of a uint32_t array"},
	{"unique", &unique, METH_VARARGS,
		"Accelerated version of np.unique for uint32_t arrays"},
	{"mode_cnt", &mode_cnt, METH_VARARGS,
		"Compute mode and its corresponding count for uint32_t/int64_t arrays"},
	{NULL, NULL, 0, NULL}
};

struct PyModuleDef speedups_module = {
	PyModuleDef_HEAD_INIT,
	"_speedups",
	NULL,
	0,
	speedups_methods
};

} // namespace

#ifdef __GNUC__
PyMODINIT_FUNC PyInit__speedups(void) __attribute__((__visibility__("default")));
#endif

PyMODINIT_FUNC
PyInit__speedups(void) {
	return PyModule_Create(&speedups_module);
}
