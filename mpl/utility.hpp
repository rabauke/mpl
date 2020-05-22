#if !(defined MPL_UTILITY_HPP)

#define MPL_UTILITY_HPP

#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>
#include <valarray>

namespace mpl {

  namespace detail {

    template<typename T>
    struct iterator_traits : public std::iterator_traits<T> {
      using insert_type = typename std::iterator_traits<T>::value_type;
    };

    template<typename T>
    struct iterator_traits<std::back_insert_iterator<T>>
        : public std::iterator_traits<std::back_insert_iterator<T>> {
      using insert_type = typename T::value_type;
    };

    template<typename from_type, typename to_type>
    struct is_narrowing {
      static constexpr bool value = std::numeric_limits<from_type>::max() >
                                    std::numeric_limits<to_type>::max();
    };

    template<typename T, bool is_enum = std::is_enum<T>::value>
    struct underlying_type;

    template<typename T>
    struct underlying_type<T, true> {
      using type = typename std::underlying_type<T>::type;

      static constexpr int value(const T &v) { return static_cast<int>(v); }
    };

    template<typename T>
    struct underlying_type<T, false> {
      using type = T;

      static constexpr int value(const T &v) { return static_cast<int>(v); }
    };

    template<typename T>
    struct is_valid_tag {
      static constexpr bool value =
          (std::is_enum<T>::value) and
          (not is_narrowing<typename underlying_type<T>::type, int>::value);
    };

    template<typename T>
    struct is_valid_color {
      static constexpr bool value =
          (std::is_integral<T>::value or std::is_enum<T>::value) and
          (not is_narrowing<typename underlying_type<T>::type, int>::value);
    };

    template<typename T>
    struct is_valid_key {
      static constexpr bool value =
          (std::is_integral<T>::value or std::is_enum<T>::value) and
          (not is_narrowing<typename underlying_type<T>::type, int>::value);
    };

    //------------------------------------------------------------------

    template<typename T, typename Enable = void>
    struct is_contiguous_iterator : public std::false_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, typename std::enable_if<std::is_same<
               T, typename std::vector<typename T::value_type>::iterator>::value>::type>
        : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, typename std::enable_if<std::is_same<
               T, typename std::vector<typename T::value_type>::const_iterator>::value>::type>
        : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, typename std::enable_if<std::is_same<
               T, typename std::basic_string<typename T::value_type>::const_iterator>::value>::
               type> : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<T, typename std::enable_if<std::is_pointer<T>::value>::type>
        : public std::true_type {};

    //------------------------------------------------------------------

    template<typename T>
    struct remove_const_from_members {
      using type = T;
    };

    template<typename T1, typename T2>
    struct remove_const_from_members<std::pair<T1, T2>> {
      using type =
          std::pair<typename std::remove_const<T1>::type, typename std::remove_const<T2>::type>;
    };

  }  // namespace detail

}  // namespace mpl

#endif
