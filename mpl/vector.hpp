#if !(defined MPL_VECTOR_HPP)

#define MPL_VECTOR_HPP

#include <cstdlib>
#include <iterator>

namespace mpl {

  namespace detail {

    struct uninitialized {};

    template<typename T>
    class vector {
    public:
      using value_type = T;
      using pointer = T *;
      using const_pointer = const T *;
      using reference = T &;
      using const_reference = const T &;
      using iterator = T *;
      using const_iterator = const T *;
      using size_type = std::size_t;

    private:
      size_type size_{0};
      T *data_{nullptr};

    public:
      explicit vector(size_type size)
          : size_{size}, data_{reinterpret_cast<pointer>(operator new(size_ * sizeof(T)))} {
        for (size_type i{0}; i < size; ++i)
          new (&data_[i]) value_type();
      }

      explicit vector(size_type size, uninitialized)
          : size_{size}, data_{reinterpret_cast<pointer>(operator new(size_ * sizeof(T)))} {
        if (not std::is_trivially_copyable<value_type>::value)
          for (size_type i{0}; i < size; ++i)
            new (&data_[i]) value_type();
      }

      template<typename IterT>
      explicit vector(size_type size, IterT iter)
          : size_{size}, data_{reinterpret_cast<pointer>(operator new(size_ * sizeof(T)))} {
        for (size_type i{0}; i < size; ++i) {
          new (&data_[i]) value_type(*iter);
          ++iter;
        }
      }

      vector(const vector &) = delete;
      vector &operator=(const vector &) = delete;

      size_type size() const { return size_; }
      bool empty() const { return size_ == 0; }
      pointer data() { return data_; }
      const_pointer data() const { return data_; }

      reference operator[](size_type i) { return data_[i]; }
      const_reference operator[](size_type i) const { return data_[i]; }

      iterator begin() { return data_; }
      const_iterator begin() const { return data_; }
      iterator end() { return data_ + size_; }
      const_iterator end() const { return data_ + size_; }

      ~vector() {
        for (auto &val : *this)
          val.~T();
        operator delete(data_);
      }
    };

  }  // namespace detail

}  // namespace mpl

#endif  // MPL_VECTOR_HPP
