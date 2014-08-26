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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <algorithm>
#include <map>
#include <memory>
#include <type_traits>
#include <assert.h>
#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>

namespace {

template <typename T>
class NumpyVectorIterator {
public:
	constexpr NumpyVectorIterator(const T *ptr, ptrdiff_t stride) :
		ptr_(ptr), stride_(stride) {}

	const T &operator *() const {
		return *ptr_;
	}

	NumpyVectorIterator &operator ++() {
		ptr_ = reinterpret_cast<const T *>(reinterpret_cast<intptr_t>(ptr_) + stride_);
		return *this;
	}

	NumpyVectorIterator operator ++(int) {
		NumpyVectorIterator copy = *this;
		++*this;
		return copy;
	}

	bool operator == (const NumpyVectorIterator &other) const { return ptr_ == other.ptr_; }
	bool operator != (const NumpyVectorIterator &other) const { return ptr_ != other.ptr_; }

private:
	const T *ptr_;
	ptrdiff_t stride_;
};

template <typename T>
class NumpyVectorView {
public:
	typedef NumpyVectorIterator<T> iterator, const_iterator;

public:
	explicit NumpyVectorView(PyArrayObject *obj) :
		ptr_(static_cast<const T *>(PyArray_DATA(obj))),
		stride_(PyArray_STRIDES(obj)[0]),
		n_(PyArray_DIMS(obj)[0]) {
	}

	iterator begin() const {
		return {ptr_, stride_};
	}

	iterator end() const {
		const T *p = reinterpret_cast<const T *>(reinterpret_cast<intptr_t>(ptr_) + stride_ * n_);
		return {p, stride_};
	}

private:
	const T *ptr_;
	ptrdiff_t stride_;
	size_t n_;
};

template <typename T>
T do_gcd_many(const NumpyVectorView<T> &data) {
	typedef typename std::make_unsigned<T>::type UT;
	UT res = 0;
	for (T u: data) {
		if (u < 0)
			u = -u;
		UT v = u;
		if (res < v) {
			UT t = v;
			v = res;
			res = t;
		}
		while (v) {
			UT t = res % v;
			res = v;
			v = t;
		}
		if (res == 1)
			break;
	}
	return res;
}

template <typename T>
size_t do_unique(T *out, const T *in, size_t n) {

	typedef typename std::make_unsigned<T>::type UT;

	T *po = out;
	{
		// Let's try to eliminate some (but not all) duplicates with
		// some simple tricks
		constexpr uint32_t HASHTABLE_SIZE = 61;
		T hashtable[HASHTABLE_SIZE] = {1, 0};

		for (size_t k = n; k; --k) {
			T v = *in++;
			size_t h = UT(v) % HASHTABLE_SIZE;
			if (hashtable[h] != v) {
				hashtable[h] = v;
				*po++ = v;
			}
		}
	}

	std::sort(out, po);

	return std::unique(out, po) - out;
}

template <typename T>
std::pair<T, size_t> do_mode_cnt(const T *p, size_t n) {
	std::map<T, size_t> cntmap;
	uint32_t maxval = 0;
	size_t maxcnt = 0;
	for (size_t k = n; k; --k) {
		T v = *p++;
		size_t cnt = ++cntmap[v];
		if (cnt > maxcnt) {
			maxcnt = cnt;
			maxval = v;
		}
	}
	return std::make_pair(maxval, maxcnt);
}

template <int mode, typename T>
T do_min_max(const T *p, size_t n) {
	assert(n > 0);

	if (mode == 0) {
		T m = p[0];
		for (size_t i = 1; i < n; ++i)
			if (p[i] < m)
				m = p[i];
		return m;
	} else if (mode == 1) {
		T M = p[0];
		for (size_t i = 1; i < n; ++i)
			if (p[i] > M)
				M = p[i];
		return M;
	} else {
		T m = p[0];
		T M = p[0];
		for (size_t i = 1; i < n; ++i) {
			if (p[i] < m)
				m = p[i];
			if (p[i] > M)
				M = p[i];
		}
		return M - m;
	}
}

PyObject *gcd_many(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;

	int type = PyArray_TYPE(array);

	if (type == NPY_UINT32)
		return PyLong_FromLong(do_gcd_many(NumpyVectorView<uint32_t>(array)));
	else
		Py_RETURN_NONE;
}

// np.fromstring(utils._array_for_speedups(array))
PyObject *unique(PyObject *self, PyObject *args) {
	Py_buffer buf;
	int type;
	if (!PyArg_ParseTuple(args, "(y*i)", &buf, &type))
		return NULL;

	size_t bytes = buf.len;

	if (type == 32) {
		// uint32
		size_t n = bytes / 4;
		std::unique_ptr<uint32_t[]> out(new uint32_t[n]);
		n = do_unique(out.get(), reinterpret_cast<const uint32_t *>(buf.buf), n);
		return PyBytes_FromStringAndSize((const char *)out.get(), n * sizeof(uint32_t));
	} else if (type == 63) {
		// int64
		size_t n = bytes / 8;
		std::unique_ptr<int64_t[]> out(new int64_t[n]);
		n = do_unique(out.get(), reinterpret_cast<const int64_t *>(buf.buf), n);
		return PyBytes_FromStringAndSize((const char *)out.get(), n * sizeof(int64_t));
	} else {
		Py_RETURN_NONE;
	}
}

// _speedups.unique(utils._array_for_speedups(array))
PyObject *mode_cnt(PyObject *self, PyObject *args) {
	Py_buffer buf;
	int type;
	if (!PyArg_ParseTuple(args, "(y*i)", &buf, &type))
		return NULL;

	size_t bytes = buf.len;

	if (type == 32 && bytes >= 4) {
		size_t n = bytes / 4;
		auto res = do_mode_cnt(reinterpret_cast<const uint32_t *>(buf.buf), n);
		return Py_BuildValue("(In)", unsigned(res.first), Py_ssize_t(res.second));
	} else if (type == 63 && bytes >= 8) {
		size_t n = bytes / 8;
		auto res = do_mode_cnt(reinterpret_cast<const uint64_t *>(buf.buf), n);
		return Py_BuildValue("(Ln)", static_cast<PY_LONG_LONG>(int64_t(res.first)), Py_ssize_t(res.second));
	} else {
		Py_RETURN_NONE;
	}
}

template <int mode>
PyObject *min_max_mode(const void *ptr, size_t bytes, int type) {
	if (type == 32 && bytes >= 4) {
		const uint32_t *p = static_cast<const uint32_t *>(ptr);
		size_t n = bytes / 4;
		return PyLong_FromLong(do_min_max<mode>(p, n));
	} else if (type == 63 && bytes >= 8) {
		const int64_t *p = static_cast<const int64_t *>(ptr);
		size_t n = bytes / 8;
		return PyLong_FromLongLong(do_min_max<mode>(p, n));
	} else {
		Py_RETURN_NONE;
	}
}

PyObject *min_max(PyObject *self, PyObject *args) {
	Py_buffer buf;
	int type;
	int mode;
	if (!PyArg_ParseTuple(args, "(y*i)i", &buf, &type, &mode))
		return NULL;

	const void *ptr = buf.buf;
	size_t bytes = buf.len;

	switch (uint32_t(mode)) {
		case 0:
			return min_max_mode<0>(ptr, bytes, type);
		case 1:
			return min_max_mode<1>(ptr, bytes, type);
		case 2:
			return min_max_mode<2>(ptr, bytes, type);
		default:
			Py_RETURN_NONE;
	}
}

PyMethodDef speedups_methods[] = {
	{"gcd_many",  &gcd_many, METH_VARARGS,
		"Calculate greatest common divisor (GCD) of a uint32_t array"},
	{"unique", &unique, METH_VARARGS,
		"Accelerated version of np.unique for uint32_t/int64_t arrays"},
	{"mode_cnt", &mode_cnt, METH_VARARGS,
		"Compute mode and its corresponding count for uint32_t/int64_t arrays"},
	{"min_max", &min_max, METH_VARARGS,
		"Compute min/max of an array"},
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
	PyObject *module = PyModule_Create(&speedups_module);
	import_array();
	return module;
}
