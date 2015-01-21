#if !(defined MPL_VECTOR_HPP)

#define MPL_VECTOR_HPP

#include <cstddef>

namespace mpl {

  template<typename T, unsigned int n>
  class vector {
    T data[n];
  public:
    typedef T value_type;
    typedef T * pointer;
    typedef T & reference;
    typedef const T & const_reference;
    typedef unsigned int size_type;
    typedef ptrdiff_t difference_type;
    typedef T * iterator;
    typedef const T * const_iterator;
    vector(const T &d0) {
      static_assert(n==1, "wrong number of arguments");
      data[0]=d0;
    }
    vector(const T &d0, const T &d1) {
      static_assert(n==2, "wrong number of arguments");
      data[0]=d0;
      data[1]=d1;
    }
    vector(const T &d0, const T &d1, const T &d2) {
      static_assert(n==3, "wrong number of arguments");
      data[0]=d0;
      data[1]=d1;
      data[2]=d2;
    }
    vector(const T &d0, const T &d1, const T &d2, const T &d3) {
      static_assert(n==4, "wrong number of arguments");
      data[0]=d0;
      data[1]=d1;
      data[2]=d2;
      data[3]=d3;
    }
    reference operator[](size_type i) {
      return data[i];
    }
    const_reference operator[](size_type i) const {
      return data[i];
    }
    iterator begin() {
      return data;
    }
    const_iterator begin() const {
      return data;
    }
    iterator end() {
      return data+n;
    }
    const_iterator end() const {
      return data+n;
    }
  };

}

#endif
