#if !(defined MPL_TEST_HELPER_HPP)
#define MPL_TEST_HELPER_HPP

#include <type_traits>
#include <mpl/mpl.hpp>

template<typename, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(&T::size)>> : std::true_type {};


template<typename, typename = void>
struct has_resize : std::false_type {};

template<typename T>
struct has_resize<T, std::void_t<decltype(T().resize(1))>> : std::true_type {};


template<typename, typename = void>
struct has_begin_end : std::false_type {};

template<typename T>
struct has_begin_end<
    T, std::void_t<std::common_type_t<decltype(T().begin()), decltype(T().end())>>>
    : std::true_type {};


struct tuple {
  int a{0};
  double b{0};
  tuple &operator++() {
    ++a;
    ++b;
    return *this;
  }
};

inline bool operator==(const tuple &t1, const tuple &t2) {
  return t1.a == t2.a and t1.b == t2.b;
}

inline tuple operator+(const tuple &t1, const tuple &t2) {
  return tuple{t1.a + t2.a, t1.b + t2.b};
}

MPL_REFLECTION(tuple, a, b)


template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) const { return a + b; }
};


enum class use_non_root_overload { no, yes };

#endif  // MPL_TEST_HELPER_HPP
