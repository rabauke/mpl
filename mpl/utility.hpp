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
    struct is_not_narrowing
        : public std::integral_constant<bool, std::is_integral_v<from_type> and
                                                  std::is_integral_v<to_type> and
                                                  (std::numeric_limits<to_type>::min() <=
                                                   std::numeric_limits<from_type>::min()) and
                                                  (std::numeric_limits<from_type>::max() <=
                                                   std::numeric_limits<to_type>::max())> {};

    template<typename from_type, typename to_type>
    inline constexpr bool is_not_narrowing_v = is_not_narrowing<from_type, to_type>::value;

    template<typename T, bool is_enum = std::is_enum_v<T>>
    struct underlying_type;

    template<typename T>
    struct underlying_type<T, true> {
      using type = std::underlying_type_t<T>;

      static constexpr int value(const T &v) { return static_cast<int>(v); }
    };

    template<typename T>
    struct underlying_type<T, false> {
      using type = T;

      static constexpr int value(const T &v) { return static_cast<int>(v); }
    };

    template<typename T>
    using underlying_type_t = typename underlying_type<T>::type;

    template<typename T>
    struct is_valid_tag
        : public std::integral_constant<
              bool, std::is_enum_v<T> and is_not_narrowing_v<underlying_type_t<T>, int>> {};

    template<typename T>
    inline constexpr bool is_valid_tag_v = is_valid_tag<T>::value;

    template<typename T>
    struct is_valid_color
        : public std::integral_constant<
              bool, (std::is_integral_v<T> or std::is_enum_v<T>) and is_not_narrowing_v<underlying_type_t<T>, int>> {};

    template<typename T>
    inline constexpr bool is_valid_color_v = is_valid_color<T>::value;

    template<typename T>
    struct is_valid_key
        : public std::integral_constant<
              bool, (std::is_integral_v<T> or std::is_enum_v<T>) and is_not_narrowing_v<underlying_type_t<T>, int>> {};

    template<typename T>
    inline constexpr bool is_valid_key_v = is_valid_key<T>::value;

    //------------------------------------------------------------------

    template<typename T, typename Enable = void>
    struct is_contiguous_iterator : public std::false_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, std::enable_if_t<
               std::is_same_v<T, typename std::vector<typename T::value_type>::iterator>>>
        : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, std::enable_if_t<
               std::is_same_v<T, typename std::vector<typename T::value_type>::const_iterator>>>
        : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<
        T, std::enable_if_t<std::is_same_v<
               T, typename std::basic_string<typename T::value_type>::const_iterator>>>
        : public std::true_type {};

    template<typename T>
    struct is_contiguous_iterator<T, std::enable_if_t<std::is_pointer_v<T>>>
        : public std::true_type {};

    template<typename T>
    inline constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<T>::value;

    //------------------------------------------------------------------

    template<typename T>
    struct remove_const_from_members {
      using type = T;
    };

    template<typename T1, typename T2>
    struct remove_const_from_members<std::pair<T1, T2>> {
      using type = std::pair<std::remove_const_t<T1>, std::remove_const_t<T2>>;
    };

    template<typename T>
    using remove_const_from_members_t = typename remove_const_from_members<T>::type;

  }  // namespace detail

}  // namespace mpl

#endif
