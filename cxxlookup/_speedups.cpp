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

// Including "0x"
// Eqv. to "len(hex(v))" in Python
template <typename T>
inline unsigned hex_len(T v) {
	unsigned r = 3;
	while ((v >>= 4) != 0)
		++r;
	return r;
}

inline char *copy_string(char *dst, const char *src) {
	size_t l = strlen(src);
	return static_cast<char *>(memcpy(dst, src, l)) + l;
}

template <typename T>
inline char *write_hex(char *dst, T v, unsigned len) {
	while (len--) {
		*dst++ = "0123456789abcdef"[(v >> (len * 4)) & 15u];
	}
	return dst;
}

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
PyObject *do_unique(PyArrayObject *array) {

	typedef typename std::make_unsigned<T>::type UT;

	size_t n = PyArray_DIMS(array)[0];

	npy_intp dims[1] = {npy_intp(n)};

	PyArray_Descr *descr = PyArray_DESCR(array);
	Py_INCREF(descr);
	PyArrayObject *out_obj = reinterpret_cast<PyArrayObject *>(
			PyArray_Empty(1, dims, descr, 0));
	if (out_obj == NULL)
		return NULL;

	T *out = static_cast<T *>(PyArray_DATA(out_obj));
	T *po = out;
	{
		// Let's try to eliminate some (but not all) duplicates with
		// some simple tricks
		constexpr uint32_t HASHTABLE_SIZE = 61;
		T hashtable[HASHTABLE_SIZE] = {1, 0};

		for (T v: NumpyVectorView<T>(array)) {
			size_t h = UT(v) % HASHTABLE_SIZE;
			if (hashtable[h] != v) {
				hashtable[h] = v;
				*po++ = v;
			}
		}
	}

	std::sort(out, po);
	size_t out_size = std::unique(out, po) - out;

	if (out_size != n) {
		dims[0] = out_size;

		PyArray_Dims newshape = {
			dims,
			1
		};

		PyObject *res = PyArray_Resize(out_obj, &newshape, 1, NPY_CORDER);
		if (res == NULL)
			return NULL;
		Py_DECREF(res);
	}
	return reinterpret_cast<PyObject *>(out_obj);
}

template <typename T>
std::pair<T, size_t> do_mode_cnt(const NumpyVectorView<T> &data) {
	std::map<T, size_t> cntmap;
	uint32_t maxval = 0;
	size_t maxcnt = 0;
	for (T v: data) {
		size_t cnt = ++cntmap[v];
		if (cnt > maxcnt) {
			maxcnt = cnt;
			maxval = v;
		}
	}
	return std::make_pair(maxval, maxcnt);
}

template <int mode, typename T>
T do_min_max(const NumpyVectorView<T> &data) {
	auto it = data.begin();
	auto end = data.end();

	assert (it != end);

	if (mode == 0) {
		T m = *it;
		while (++it != end)
			if (*it < m)
				m = *it;
		return m;
	} else if (mode == 1) {
		T M = *it;
		while (++it != end)
			if (*it > M)
				M = *it;
		return M;
	} else {
		T m = *it;
		T M = *it;
		while (++it != end) {
			T v = *it;
			if (v < m)
				m = v;
			if (v > M)
				M = v;
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

PyObject *unique(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;

	switch (PyArray_TYPE(array)) {
		case NPY_UINT32:
			return do_unique<uint32_t>(array);
		case NPY_INT64:
			return do_unique<int64_t>(array);
		default:
			Py_RETURN_NONE;
	}
}

PyObject *mode_cnt(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;
	if (PyArray_DIMS(array)[0] == 0)
		Py_RETURN_NONE;

	int type = PyArray_TYPE(array);

	if (type == NPY_UINT32) {
		auto res = do_mode_cnt(NumpyVectorView<uint32_t>(array));
		return Py_BuildValue("(In)", unsigned(res.first), Py_ssize_t(res.second));
	} else if (type == NPY_INT64) {
		auto res = do_mode_cnt(NumpyVectorView<int64_t>(array));
		return Py_BuildValue("(Ln)", static_cast<PY_LONG_LONG>(int64_t(res.first)), Py_ssize_t(res.second));
	} else {
		Py_RETURN_NONE;
	}
}

template <int mode>
PyObject *min_max_mode(PyArrayObject *array) {
	switch (PyArray_TYPE(array)) {
		case NPY_UINT32:
			return PyLong_FromLong(do_min_max<mode>(NumpyVectorView<uint32_t>(array)));
		case NPY_INT64:
			return PyLong_FromLongLong(do_min_max<mode>(NumpyVectorView<int64_t>(array)));
		default:
			Py_RETURN_NONE;
	}
}

PyObject *min_max(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	int mode;
	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &array, &mode))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;
	if (PyArray_DIMS(array)[0] == 0)
		Py_RETURN_NONE;

	switch (uint32_t(mode)) {
		case 0:
			return min_max_mode<0>(array);
		case 1:
			return min_max_mode<1>(array);
		case 2:
			return min_max_mode<2>(array);
		default:
			Py_RETURN_NONE;
	}
}

template <typename T>
bool do_is_linear(const NumpyVectorView<T> &data) {
	auto it = data.begin();
	auto end = data.end();

	T first = *it;
	T second = *++it;
	T slope = second - first;

	T v = second;
	while (++it != end) {
		v += slope;
		if (v != *it)
			return false;
	}
	return true;
}

PyObject *is_linear(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;
	if (PyArray_DIMS(array)[0] < 3)
		Py_RETURN_TRUE;

	switch (PyArray_TYPE(array)) {
		case NPY_UINT32:
			return PyBool_FromLong(do_is_linear(NumpyVectorView<uint32_t>(array)));
		default:
			Py_RETURN_NONE;
	}
}

template <typename T>
bool do_is_const(const NumpyVectorView<T> &data) {
	auto it = data.begin();
	auto end = data.end();

	T first = *it;

	while (++it != end) {
		if (first != *it)
			return false;
	}
	return true;
}

PyObject *is_const(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;
	if (PyArray_DIMS(array)[0] < 2)
		Py_RETURN_TRUE;

	switch (PyArray_TYPE(array)) {
		case NPY_UINT32:
			return PyBool_FromLong(do_is_const(NumpyVectorView<uint32_t>(array)));
		default:
			Py_RETURN_NONE;
	}
}

template <typename R, int NPR, typename T>
PyObject *do_slope_array(PyArrayObject *array) {
	size_t n = PyArray_DIMS(array)[0];
	if (n < 2)
		Py_RETURN_NONE;

	npy_intp dims[1] = {npy_intp(n - 1)};

	PyArrayObject *out_obj = reinterpret_cast<PyArrayObject *>(
			PyArray_EMPTY(1, dims, NPR, 0));
	if (out_obj == NULL)
		return NULL;

	R *out = static_cast<R *>(PyArray_DATA(out_obj));

	NumpyVectorView<T> view(array);
	auto it = view.begin();
	auto end = view.end();

	R prev = *it;
	while (++it != end) {
		R cur = *it;
		*out++ = cur - prev;
		prev = cur;
	}

	return reinterpret_cast<PyObject *>(out_obj);
}

PyObject *slope_array(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	PyTypeObject *dtype;
	if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &array, &dtype))
		return NULL;

	// Note that dtype may not necessarily be a real PyTypeObject object

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;

	switch (PyArray_TYPE(array)) {
		case NPY_UINT32:
			if (dtype == &PyInt64ArrType_Type)
				return do_slope_array<int64_t, NPY_INT64, uint32_t>(array);
			else if (dtype == &PyUInt32ArrType_Type)
				return do_slope_array<uint32_t, NPY_UINT32, uint32_t>(array);
			else
				Py_RETURN_NONE;
		default:
			Py_RETURN_NONE;
	}
}

// format_c_array(array, type_str, name_str)
PyObject *format_c_array(PyObject *self, PyObject *args) {
	PyArrayObject *array;
	const char *type;
	Py_ssize_t type_len;
	const char *name;
	Py_ssize_t name_len;

	if (!PyArg_ParseTuple(args, "O!s#s#", &PyArray_Type, &array, &type, &type_len, &name, &name_len))
		return NULL;

	if (PyArray_NDIM(array) != 1)
		Py_RETURN_NONE;
	if (PyArray_TYPE(array) != NPY_UINT32)
		Py_RETURN_NONE;

	size_t n = PyArray_DIMS(array)[0];
	unsigned n_len = hex_len(n);

	uint32_t max = do_min_max<1>(NumpyVectorView<uint32_t>(array));
	unsigned max_len = hex_len(max);

	size_t mem_upper_bound = (max_len + 4) * n + (n_len + 8) * (n / 8) + type_len + name_len + 128;
	std::unique_ptr<char[]> buf(new char[mem_upper_bound]);
	char *w = buf.get();

	w += snprintf(w, mem_upper_bound, "const %.*s %.*s[%#zx] = {",
			int(type_len), type, int(name_len), name, n);

	size_t i = 0;
	for (uint32_t v: NumpyVectorView<uint32_t>(array)) {
		if (i % 8 == 0) {
			w = copy_string(w, "\n\t/* 0x");
			w = write_hex(w, i, n_len - 2);
			w = copy_string(w, " */");
		}
		w = copy_string(w, " 0x");
		w = write_hex(w, v, max_len - 2);
		*w++ = ',';
		++i;
	}
	if (*(w - 1) == ',')
		--w;
	w = copy_string(w, "\n};\n\n");

	return Py_BuildValue("s#", buf.get(), Py_ssize_t(w - buf.get()));
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
	{"is_linear", &is_linear, METH_VARARGS,
		"Return whether an array is linear"},
	{"is_const", &is_const, METH_VARARGS,
		"Return whether an array is constant"},
	{"slope_array", &slope_array, METH_VARARGS,
		"Create the \"slope array\" of a given array"},
	{"format_c_array", &format_c_array, METH_VARARGS,
		"Format a NumPy string as a C array"},
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
