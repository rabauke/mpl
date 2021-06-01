#if !(defined MPL_TEST_HELPER_HPP)
#define MPL_TEST_HELPER_HPP

#include <type_traits>

template<typename, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(T().size())>> : std::true_type {};


template<typename, typename = void>
struct has_resize : std::false_type {};

template<typename T>
struct has_resize<T, std::void_t<decltype(T().resize(1))>> : std::true_type {};

#endif  // MPL_TEST_HELPER_HPP
