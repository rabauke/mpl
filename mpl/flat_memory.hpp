#if !(defined MPL_FLAT_MEMORY_HPP)

#define MPL_FLAT_MEMORY_HPP

#include <mpi.h>
#include <cstddef>
#include <vector>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <mpl/utility.hpp>

namespace mpl {

  namespace detail {

    template<typename T, typename I>
    class flat_memory_in {
    public:
      using size_type = std::ptrdiff_t;

    private:
      size_type n;
      T *first;

    public:
      flat_memory_in(I i1, I i2) : n(std::distance(i1, i2)), first(new T[n]) {
        std::copy(i1, i2, first);
      }

      ~flat_memory_in() { delete[] first; }

      size_type size() const { return n; }

      const T *data() const { return first; }

      const T &operator[](std::size_t i) const { return data()[i]; }
    };

    template<typename T>
    class flat_memory_in<T, T *> {
    public:
      using size_type = std::ptrdiff_t;

    private:
      size_type n;
      const T *first;

    public:
      flat_memory_in(const T *i1, const T *i2) : n(std::distance(i1, i2)), first(i1) {}

      ~flat_memory_in() = default;

      size_type size() const { return n; }

      const T *data() const { return first; }

      const T &operator[](std::size_t i) const { return data()[i]; }
    };

    // a more general implementation would be possible with
    // contiguous_iterator_tag of C++17
    template<typename T>
    class flat_memory_in<T, typename std::vector<T>::iterator> {
    public:
      using size_type = std::ptrdiff_t;

    private:
      using iterator = typename std::vector<T>::iterator;
      size_type n;
      iterator first;

    public:
      flat_memory_in(iterator i1, iterator i2) : n(std::distance(i1, i2)), first(i1) {}

      ~flat_memory_in() = default;

      size_type size() const { return n; }

      const T *data() const { return &(*first); }

      const T &operator[](std::size_t i) const { return data()[i]; }
    };

    template<typename T>
    class flat_memory_in<T, typename std::vector<T>::const_iterator> {
    public:
      using size_type = std::ptrdiff_t;

    private:
      using const_iterator = typename std::vector<T>::const_iterator;
      size_type n;
      const_iterator first;

    public:
      flat_memory_in(const_iterator i1, const_iterator i2) : n(std::distance(i1, i2)), first(i1) {}

      ~flat_memory_in() = default;

      size_type size() const { return n; }

      const T *data() const { return &(*first); }

      const T &operator[](std::size_t i) const { return data()[i]; }
    };

    //--------------------------------------------------------------------

    template<typename T, typename I>
    class flat_memory_out {
    public:
      using size_type = std::ptrdiff_t;

    private:
      size_type n;
      I first_out;
      T *first;

    public:
      flat_memory_out(size_type n, I first_out) : n(n), first_out(first_out), first(new T[n]) {}

      ~flat_memory_out() { delete[] first; }

      size_type size() const { return n; }

      const T *data() const { return first; }

      T *data() { return first; }

      const T &operator[](std::size_t i) const { return data()[i]; }

      T &operator[](std::size_t i) { return data()[i]; }

      I copy_back(size_type m) const {
        return std::copy(first, first + std::min(m, n), first_out);
      }
    };

    template<typename T>
    class flat_memory_out<T, T *> {
    public:
      using size_type = std::ptrdiff_t;

    private:
      size_type n;
      T *first_out;
      T *first;

    public:
      flat_memory_out(size_type n, T *first_out)
          : n(n), first_out(first_out), first(first_out) {}

      ~flat_memory_out() = default;

      size_type size() const { return n; }

      const T *data() const { return first; }

      T *data() { return first; }

      const T &operator[](std::size_t i) const { return data()[i]; }

      T &operator[](std::size_t i) { return data()[i]; }

      T *copy_back(size_type m) const { return first_out + std::min(m, n); }
    };

    // a more general implementation would be possible with
    // contiguous_iterator_tag of C++17
    template<typename T>
    class flat_memory_out<T, typename std::vector<T>::iterator> {
    public:
      using size_type = std::ptrdiff_t;

    private:
      using iterator = typename std::vector<T>::iterator;
      size_type n;
      iterator first_out;
      iterator first;

    public:
      flat_memory_out(size_type n, iterator first_out)
          : n(n), first_out(first_out), first(first_out) {}

      ~flat_memory_out() = default;

      size_type size() const { return n; }

      const T *data() const { return &(*first); }

      T *data() { return &(*first); }

      const T &operator[](std::size_t i) const { return data()[i]; }

      T &operator[](std::size_t i) { return data()[i]; }

      iterator copy_back(size_type m) const { return first_out + std::min(m, n); }
    };

  }  // namespace detail

}  // namespace mpl

#endif
