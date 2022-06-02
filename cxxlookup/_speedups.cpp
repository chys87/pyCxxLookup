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
#include <Python.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if __has_include(<x86intrin.h>)
#include <x86intrin.h>
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include <numpy/arrayobject.h>

namespace {

constexpr size_t kReleaseGILThreshold = 1024;

class ReleaseGIL {
 public:
  ReleaseGIL() { Py_UNBLOCK_THREADS; }
  explicit ReleaseGIL(size_t size) {
    if (size >= kReleaseGILThreshold) {
      Py_UNBLOCK_THREADS;
    }
  }

  ReleaseGIL(const ReleaseGIL&) = delete;
  ReleaseGIL& operator=(const ReleaseGIL&) = delete;

  ~ReleaseGIL() {
    if (_save) {
      Py_BLOCK_THREADS;
    }
  }

 private:
  PyThreadState* _save = nullptr;
};

template <typename T>
class VectorView {
 public:
  // stride_ can be 0
  explicit VectorView(PyArrayObject* obj)
      : ptr_(static_cast<const T*>(PyArray_DATA(obj))),
        stride_(PyArray_STRIDES(obj)[0]),
        remaining_(PyArray_DIMS(obj)[0]) {}
  VectorView(const VectorView&) = default;

  bool more() const { return remaining_; }
  explicit operator bool() const { return more(); }

  T operator[](size_t k) const {
    return *reinterpret_cast<const T*>(intptr_t(ptr_) + stride_ * k);
  }

  T next() {
    T v = *ptr_;
    ptr_ =
        reinterpret_cast<const T*>(reinterpret_cast<intptr_t>(ptr_) + stride_);
    --remaining_;
    return v;
  }

  const T* data() const { return ptr_; }
  ptrdiff_t stride() const { return stride_; }
  unsigned size() const { return remaining_; }

 private:
  const T* ptr_;
  ptrdiff_t stride_;
  unsigned remaining_;
};

// Including "0x"
// Eqv. to "len(hex(v))" in Python
template <typename T>
inline unsigned hex_len(T v) {
  unsigned r = 3;
  while ((v >>= 4) != 0) ++r;
  return r;
}

inline char* copy_string(char* dst, const char* src) {
  size_t l = strlen(src);
  return static_cast<char*>(memcpy(dst, src, l)) + l;
}

template <typename T>
inline char* write_hex(char* dst, T v, unsigned len) {
  while (len--) {
    *dst++ = "0123456789abcdef"[(v >> (len * 4)) & 15u];
  }
  return dst;
}

template <typename T>
inline constexpr
    typename std::enable_if<std::is_signed<T>::value,
                            typename std::make_unsigned<T>::type>::type
    do_abs(T u) {
  return (u < 0) ? -u : u;
}

template <typename T>
inline constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
do_abs(T u) {
  return u;
}

template <typename T>
typename std::make_unsigned<T>::type do_gcd_many(VectorView<T> data) {
  typedef typename std::make_unsigned<T>::type UT;

  if (!data) return 0;

  UT prev = do_abs(data.next());
  UT res = prev;
  while ((res != 1) && data) {
    UT v = do_abs(data.next());
    if (v == prev) continue;
    prev = v;
    if (res < v) std::swap(res, v);
    while (v) {
      UT t = res % v;
      res = v;
      v = t;
    }
  }
  return res;
}

template <typename T>
PyObject* do_unique(PyArrayObject* array) {
  typedef typename std::make_unsigned<T>::type UT;

  size_t n = PyArray_DIMS(array)[0];

  npy_intp dims[1] = {npy_intp(n)};

  PyArray_Descr* descr = PyArray_DESCR(array);
  Py_INCREF(descr);
  PyArrayObject* out_obj =
      reinterpret_cast<PyArrayObject*>(PyArray_Empty(1, dims, descr, 0));
  if (out_obj == NULL) return NULL;

  T* out = static_cast<T*>(PyArray_DATA(out_obj));
  T* po = out;
  size_t out_size;
  {
    VectorView<T> view(array);
    ReleaseGIL release_gil(view.size());
    {
      // Let's try to eliminate some (but not all) duplicates with
      // some simple tricks
      constexpr uint32_t HASHTABLE_SIZE = 61;
      T hashtable[HASHTABLE_SIZE] = {1, 0};

      while (view) {
        T v = view.next();
        size_t h = UT(v) % HASHTABLE_SIZE;
        if (hashtable[h] != v) {
          hashtable[h] = v;
          *po++ = v;
        }
      }
    }

    std::sort(out, po);
    out_size = std::unique(out, po) - out;
  }

  if (out_size != n) {
    dims[0] = out_size;

    PyArray_Dims newshape = {dims, 1};

    PyObject* res = PyArray_Resize(out_obj, &newshape, 1, NPY_CORDER);
    if (res == NULL) return NULL;
    Py_DECREF(res);
  }
  return reinterpret_cast<PyObject*>(out_obj);
}

template <typename T>
std::pair<T, size_t> do_mode_cnt(VectorView<T> data) {
  assert(data.more());

  T prev = data.next();
  T maxval = prev;
  size_t maxcnt = 1;

  std::map<T, size_t> cntmap;
  auto pair = cntmap.insert(std::make_pair(maxval, 1));
  size_t* prevp = &pair.first->second;

  while (data) {
    T v = data.next();
    if (v != prev) {
      prev = v;
      prevp = &cntmap[v];
    }
    size_t cnt = ++*prevp;
    if (cnt > maxcnt) {
      maxcnt = cnt;
      maxval = v;
    }
  }
  return std::make_pair(maxval, maxcnt);
}

template <int mode, typename T>
T do_min_max(VectorView<T> data) {
  assert(data.more());

  if (mode == 0) {
    T m = data.next();
    while (data) {
      T v = data.next();
      if (v < m) m = v;
    }
    return m;
  } else if (mode == 1) {
    T M = data.next();
    while (data) {
      T v = data.next();
      if (v > M) M = v;
    }
    return M;
  } else {
    T m = data.next();
    T M = m;
    while (data) {
      T v = data.next();
      if (v < m) m = v;
      if (v > M) M = v;
    }
    return M - m;
  }
}

PyObject* gcd_many(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;

  int type = PyArray_TYPE(array);

  if (type == NPY_UINT32) {
    VectorView<uint32_t> view(array);
    auto res = (ReleaseGIL(view.size()), do_gcd_many(view));
    return PyLong_FromLong(res);
  } else {
    Py_RETURN_NONE;
  }
}

PyObject* unique(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;

  switch (PyArray_TYPE(array)) {
    case NPY_UINT32:
      return do_unique<uint32_t>(array);
    case NPY_INT64:
      return do_unique<int64_t>(array);
    default:
      Py_RETURN_NONE;
  }
}

PyObject* mode_cnt(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  if (PyArray_DIMS(array)[0] == 0) Py_RETURN_NONE;

  int type = PyArray_TYPE(array);

  if (type == NPY_UINT32) {
    VectorView<uint32_t> view(array);
    auto res = (ReleaseGIL(view.size() * 3), do_mode_cnt(view));
    return Py_BuildValue("(In)", unsigned(res.first), Py_ssize_t(res.second));
  } else if (type == NPY_INT64) {
    VectorView<int64_t> view(array);
    auto res = (ReleaseGIL(view.size() * 3), do_mode_cnt(view));
    return Py_BuildValue("(Ln)", static_cast<PY_LONG_LONG>(int64_t(res.first)),
                         Py_ssize_t(res.second));
  } else {
    Py_RETURN_NONE;
  }
}

template <int mode>
PyObject* min_max_mode(PyArrayObject* array) {
  switch (PyArray_TYPE(array)) {
    case NPY_UINT32: {
      VectorView<uint32_t> view(array);
      auto res = (ReleaseGIL(view.size()), do_min_max<mode>(view));
      return PyLong_FromLong(res);
    }
    case NPY_INT64: {
      VectorView<int64_t> view(array);
      auto res = (ReleaseGIL(view.size()), do_min_max<mode>(view));
      return PyLong_FromLongLong(res);
    }
    default:
      Py_RETURN_NONE;
  }
}

PyObject* min_max(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  int mode;
  if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &array, &mode)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  if (PyArray_DIMS(array)[0] == 0) Py_RETURN_NONE;

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
bool do_is_linear(VectorView<T> data) {
  T first = data.next();
  T second = data.next();
  T slope = second - first;

  T v = second;
  while (data) {
    v += slope;
    if (v != data.next()) return false;
  }
  return true;
}

PyObject* is_linear(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  if (PyArray_DIMS(array)[0] < 3) Py_RETURN_TRUE;

  switch (PyArray_TYPE(array)) {
    case NPY_UINT32:
      return PyBool_FromLong(do_is_linear(VectorView<uint32_t>(array)));
    default:
      Py_RETURN_NONE;
  }
}

template <typename T>
bool do_is_const(VectorView<T> data) {
  T first = data.next();

  while (data) {
    if (first != data.next()) return false;
  }
  return true;
}

PyObject* is_const(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  if (PyArray_DIMS(array)[0] < 2) Py_RETURN_TRUE;

  switch (PyArray_TYPE(array)) {
    case NPY_UINT32:
      return PyBool_FromLong(do_is_const(VectorView<uint32_t>(array)));
    default:
      Py_RETURN_NONE;
  }
}

template <typename T>
long do_const_range(VectorView<T> data) {
  T first = data.next();
  long cnt = 1;

  while (data) {
    if (first != data.next()) break;
    ++cnt;
  }
  return cnt;
}

PyObject* const_range(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  size_t n = PyArray_DIMS(array)[0];
  if (n < 2) return PyLong_FromLong(n);

  switch (PyArray_TYPE(array)) {
    case NPY_UINT32:
      return PyLong_FromLong(do_const_range(VectorView<uint32_t>(array)));
    default:
      Py_RETURN_NONE;
  }
}

template <typename R, int NPR, typename T>
PyObject* do_slope_array(PyArrayObject* array) {
  size_t n = PyArray_DIMS(array)[0];
  if (n < 2) Py_RETURN_NONE;

  npy_intp dims[1] = {npy_intp(n - 1)};

  PyArrayObject* out_obj =
      reinterpret_cast<PyArrayObject*>(PyArray_EMPTY(1, dims, NPR, 0));
  if (out_obj == NULL) return NULL;

  R* out = static_cast<R*>(PyArray_DATA(out_obj));

  VectorView<T> view(array);
  R prev = view.next();
  while (view) {
    R cur = view.next();
    *out++ = cur - prev;
    prev = cur;
  }

  return reinterpret_cast<PyObject*>(out_obj);
}

PyObject* slope_array(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  PyTypeObject* dtype;
  if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &array, &dtype))
    return NULL;

  // Note that dtype may not necessarily be a real PyTypeObject object

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;

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

template <typename T>
bool do_array_equal(PyArrayObject* x, PyArrayObject* y, size_t n) {
  VectorView<T> vx(x);
  VectorView<T> vy(y);
  for (; n; --n) {
    if (vx.next() != vy.next()) return false;
  }
  return true;
}

PyObject* array_equal(PyObject* self, PyObject* args) {
  PyArrayObject *x, *y;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &y))
    return NULL;

  int type = PyArray_TYPE(x);
  if (type != PyArray_TYPE(y)) Py_RETURN_NONE;

  if (PyArray_NDIM(x) != 1 || PyArray_NDIM(y) != 1) Py_RETURN_NONE;

  size_t n = PyArray_DIMS(x)[0];
  if (n != PyArray_DIMS(y)[0]) Py_RETURN_NONE;

  bool r;
  switch (type) {
    case NPY_UINT32:
      r = do_array_equal<uint32_t>(x, y, n);
      break;
    default:
      Py_RETURN_NONE;
  }
  return PyBool_FromLong(r);
}

[[gnu::always_inline]] inline bool array_range_equal(
    const VectorView<uint32_t>& array, size_t a, size_t b, size_t n) {
  auto stride = array.stride();
  const uint32_t* p =
      reinterpret_cast<const uint32_t*>(intptr_t(array.data()) + a * stride);
  const uint32_t* q =
      reinterpret_cast<const uint32_t*>(intptr_t(array.data()) + b * stride);

  if (stride == sizeof(uint32_t)) {
    // This is the common case
#ifdef __SSE4_1__
    if (n >= 4) {
      __m128i X = _mm_loadu_si128((const __m128i*)p) ^
                  _mm_loadu_si128((const __m128i*)q);
      if (!_mm_testz_si128(X, X)) return false;
      p += 4;
      q += 4;
      n -= 4;
    }
#endif
    return memcmp(p, q, n * sizeof(uint32_t)) == 0;
  } else {
    for (; n; --n) {
      if (*p != *q) return false;
      p = reinterpret_cast<const uint32_t*>(uintptr_t(p) + stride);
      q = reinterpret_cast<const uint32_t*>(uintptr_t(q) + stride);
    }
    return true;
  }
}

PyObject* array_cycle(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  unsigned max_cycle;
  if (!PyArg_ParseTuple(args, "O!I", &PyArray_Type, &array, &max_cycle))
    return nullptr;
  if (PyArray_TYPE(array) != NPY_UINT32) Py_RETURN_NONE;
  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;

  size_t nn = PyArray_DIMS(array)[0];
  uint32_t n = nn;
  if (nn < 2) return PyLong_FromLong(0);
  if (nn != n) Py_RETURN_NONE;
  VectorView<uint32_t> array_view(array);

  long res = 0;

  do {
    ReleaseGIL release_gil(nn);

    // Typical Linux stack size is 8 MiB
    constexpr size_t kStackAlloc = 1024 * 1024 / sizeof(uint32_t);
    uint32_t indices_stack_alloc[kStackAlloc];
    std::unique_ptr<uint32_t[]> indices_new(n <= kStackAlloc ? nullptr
                                                             : new uint32_t[n]);
    uint32_t* indices =
        (n <= kStackAlloc) ? indices_stack_alloc : indices_new.get();
    uint32_t ind_n = 0;

    {
      VectorView<uint32_t> view(array_view);
      uint32_t first = view[0];
      uint32_t i = 0;
      while (view) {
        if (view.next() == first) indices[ind_n++] = i;
        ++i;
      }
    }

    // Special case: constant array
    if (ind_n == n) {
      res = 1;
      break;
    }

    for (uint32_t i = 1; i < ind_n; ++i) {
      uint32_t k = indices[i];
      if (k > max_cycle) break;

      // Check whether indices are likely correct
      {
        bool ok = true;
        for (uint32_t j = i * 2; j < ind_n; j += i) {
          if (indices[j] != indices[j - i] + k) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
      }

      // Compare array slices
      bool ok = true;
      for (uint32_t j = k; j < n; j += k) {
        if (!array_range_equal(array_view, 0, j, std::min(n - j, k))) {
          ok = false;
          break;
        }
      }
      if (ok) {
        res = k;
        break;
      }
    }
  } while (false);
  return PyLong_FromLong(res);
}


// format_c_array(array, type, name_str)
PyObject* format_c_array(PyObject* self, PyObject* args) {
  PyArrayObject* array;
  unsigned type;
  const char* name;
  Py_ssize_t name_len;

  if (!PyArg_ParseTuple(args, "O!Is#", &PyArray_Type, &array, &type, &name,
                        &name_len))
    return NULL;

  if (PyArray_NDIM(array) != 1) Py_RETURN_NONE;
  if (PyArray_TYPE(array) != NPY_UINT32) Py_RETURN_NONE;

  size_t n = PyArray_DIMS(array)[0];
  VectorView<uint32_t> view(array);

  std::unique_ptr<char[]> buf;
  char* w;

  VectorView<uint32_t> array_view(array);

  {
    ReleaseGIL release_gil(array_view.size());

    unsigned n_len = hex_len(n);

    uint32_t max = do_min_max<1>(array_view);
    unsigned max_len = hex_len(max);

    size_t mem_upper_bound =
        (max_len + 4) * n + (n_len + 8) * (n / 8) + name_len + 128;
    buf.reset(w = new char[mem_upper_bound]);

    w += snprintf(
        w, mem_upper_bound, "alignas(%sint%u_t) const %sint%u_t %.*s[%#zx] = {",
        (type & 1) ? "" : "u", (type + 1) & ~1u, (type & 1) ? "" : "u",
        (type + 1) & ~1u, int(name_len), name, n);

    size_t i = 0;
    while (view) {
      uint32_t v = view.next();
      if (i % 8 == 0) {
        w = copy_string(w, "\n  /* 0x");
        w = write_hex(w, i, n_len - 2);
        w = copy_string(w, " */");
      }
      w = copy_string(w, " 0x");
      w = write_hex(w, v, max_len - 2);
      *w++ = ',';
      ++i;
    }
    if (*(w - 1) == ',') --w;
    w = copy_string(w, "\n};\n\n");
  }

  return Py_BuildValue("s#", buf.get(), Py_ssize_t(w - buf.get()));
}

PyMethodDef speedups_methods[] = {
    {"gcd_many", &gcd_many, METH_VARARGS,
     "Calculate greatest common divisor (GCD) of a uint32_t array"},
    {"unique", &unique, METH_VARARGS,
     "Accelerated version of np.unique for uint32_t/int64_t arrays"},
    {"mode_cnt", &mode_cnt, METH_VARARGS,
     "Compute mode and its corresponding count for uint32_t/int64_t arrays"},
    {"min_max", &min_max, METH_VARARGS, "Compute min/max of an array"},
    {"is_linear", &is_linear, METH_VARARGS,
     "Return whether an array is linear"},
    {"is_const", &is_const, METH_VARARGS,
     "Return whether an array is constant"},
    {"const_range", &const_range, METH_VARARGS,
     "Return the length of prefix that is constant"},
    {"slope_array", &slope_array, METH_VARARGS,
     "Create the \"slope array\" of a given array"},
    {"array_equal", &array_equal, METH_VARARGS,
     "Return whether two arrays are equal"},
    {"array_cycle", &array_cycle, METH_VARARGS,
     "Find minimum positive cycle of an array"},
    {"format_c_array", &format_c_array, METH_VARARGS,
     "Format a NumPy string as a C array"},
    {NULL, NULL, 0, NULL}};

struct PyModuleDef speedups_module = {PyModuleDef_HEAD_INIT, "_speedups", NULL,
                                      0, speedups_methods};

}  // namespace

#ifdef __GNUC__
PyMODINIT_FUNC PyInit__speedups(void)
    __attribute__((__visibility__("default")));
#endif

PyMODINIT_FUNC PyInit__speedups(void) {
  PyObject* module = PyModule_Create(&speedups_module);
  import_array();
  return module;
}
