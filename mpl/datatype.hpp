#if !(defined MPL_DATATYPE_HPP)

#define MPL_DATATYPE_HPP

#include <mpi.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <valarray>
#include <complex>
#include <utility>
#include <tuple>
#include <array>
#include <cstddef>
#include <type_traits>

namespace mpl {

  namespace detail {

    struct unsupported_type {};
    struct basic_or_fixed_size_type {};
    struct stl_container {};
    struct contigous_const_stl_container : public stl_container {};
    struct contigous_stl_container : public contigous_const_stl_container {};

  }  // namespace detail

  //--- forward declarations -------------------------------------------

  template<typename T>
  class datatype_traits;

  namespace detail {

    template<typename T, typename E>
    class datatype_traits_impl;

  }

  template<typename T>
  class base_struct_builder;

  template<typename T>
  class struct_builder {
  public:
    using data_type_category = detail::unsupported_type;
  };

  //--------------------------------------------------------------------

  template<typename S>
  class struct_layout {
    template<typename T>
    inline std::size_t size(T) {
      return sizeof(T);
    }

    template<typename T>
    inline std::size_t size(T *) {
      return sizeof(T);
    }

    template<typename T>
    inline MPI_Datatype get_datatype(T) {
      return datatype_traits<T>().get_datatype();
    }

    template<typename T>
    inline MPI_Datatype get_datatype(T *) {
      return datatype_traits<T>().get_datatype();
    }

    MPI_Aint base;
    std::vector<int> blocklengths;
    std::vector<MPI_Aint> displacements;
    std::vector<MPI_Datatype> datatypes;

  public:
    struct_layout &register_struct(const S &x) {
      MPI_Get_address(const_cast<S *>(&x), &base);
      return *this;
    }

    template<typename T>
    struct_layout &register_element(T &x) {
      static_assert(not std::is_const<T>::value, "type must not be const");
      blocklengths.push_back(sizeof(x) / size(x));
      MPI_Aint address;
      MPI_Get_address(&x, &address);
      displacements.push_back(address - base);
      datatypes.push_back(get_datatype(x));
      return *this;
    }

    template<typename T>
    struct_layout &register_vector(T *x, std::ptrdiff_t N) {
      static_assert(not std::is_const<T>::value, "type must not be const");
      blocklengths.push_back(N);
      MPI_Aint address;
      MPI_Get_address(x, &address);
      displacements.push_back(address - base);
      datatypes.push_back(get_datatype(x));
      return *this;
    }

    friend class base_struct_builder<S>;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class base_struct_builder {
  private:
    MPI_Datatype type;

  public:
    void define_struct(const struct_layout<T> &str) {
      MPI_Datatype temp_type;
      MPI_Type_create_struct(str.blocklengths.size(), str.blocklengths.data(),
                             str.displacements.data(), str.datatypes.data(), &temp_type);
      MPI_Type_commit(&temp_type);
      MPI_Type_create_resized(temp_type, 0, sizeof(T), &type);
      MPI_Type_commit(&type);
      MPI_Type_free(&temp_type);
    }

    base_struct_builder() = default;

    base_struct_builder(const base_struct_builder &) = delete;

    void operator=(const base_struct_builder &) = delete;

    ~base_struct_builder() { MPI_Type_free(&type); }

    using data_type_category = detail::basic_or_fixed_size_type;

    friend class detail::datatype_traits_impl<T, void>;
  };

  //--------------------------------------------------------------------

  template<typename T1, typename T2>
  class struct_builder<std::pair<T1, T2>> : public base_struct_builder<std::pair<T1, T2>> {
    using base = base_struct_builder<std::pair<T1, T2>>;
    struct_layout<std::pair<T1, T2>> layout;

  public:
    struct_builder() {
      std::pair<T1, T2> pair;
      layout.register_struct(pair);
      layout.register_element(pair.first);
      layout.register_element(pair.second);
      base::define_struct(layout);
    }
  };

  //--------------------------------------------------------------------

  namespace detail {

    template<typename F, typename T, std::size_t n>
    class apply_n {
      F &f;

    public:
      explicit apply_n(F &f) : f(f) {}

      void operator()(T &x) const {
        apply_n<F, T, n - 1> next(f);
        next(x);
        f(std::get<n - 1>(x));
      }
    };

    template<typename F, typename T>
    struct apply_n<F, T, 1> {
      F &f;

    public:
      explicit apply_n(F &f) : f(f) {}

      void operator()(T &x) const { f(std::get<0>(x)); }
    };

    template<typename F, typename... Args>
    void apply(std::tuple<Args...> &t, F &f) {
      apply_n<F, std::tuple<Args...>, std::tuple_size<std::tuple<Args...>>::value> app(f);
      app(t);
    }

    template<typename... Ts>
    class register_element {
      struct_layout<std::tuple<Ts...>> &layout;

    public:
      explicit register_element(struct_layout<std::tuple<Ts...>> &layout) : layout(layout) {}

      template<typename T>
      void operator()(T &x) const {
        layout.register_element(x);
      }
    };

  }  // namespace detail

  template<typename... Ts>
  class struct_builder<std::tuple<Ts...>> : public base_struct_builder<std::tuple<Ts...>> {
    using base = base_struct_builder<std::tuple<Ts...>>;
    struct_layout<std::tuple<Ts...>> layout;

  public:
    struct_builder() {
      std::tuple<Ts...> tuple;
      layout.register_struct(tuple);
      base::define_struct(layout);
      detail::register_element<Ts...> reg(layout);
      detail::apply<detail::register_element<Ts...>>(tuple, reg);
      base::define_struct(layout);
    }
  };

  //--------------------------------------------------------------------

  template<typename T, std::size_t N0>
  class struct_builder<T[N0]> : public base_struct_builder<T[N0]> {
    using base = base_struct_builder<T[N0]>;
    struct_layout<T[N0]> layout;

  public:
    struct_builder() {
      T array[N0];
      layout.register_struct(array);
      layout.register_vector(&array[0], N0);
      base::define_struct(layout);
    }
  };

  template<typename T, std::size_t N0, std::size_t N1>
  class struct_builder<T[N0][N1]> : public base_struct_builder<T[N0][N1]> {
    using base = base_struct_builder<T[N0][N1]>;
    struct_layout<T[N0][N1]> layout;

  public:
    struct_builder() {
      T array[N0][N1];
      layout.register_struct(array);
      layout.register_vector(&array[0][0], N0 * N1);
      base::define_struct(layout);
    }
  };

  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2>
  class struct_builder<T[N0][N1][N2]> : public base_struct_builder<T[N0][N1][N2]> {
    using base = base_struct_builder<T[N0][N1][N2]>;
    struct_layout<T[N0][N1][N2]> layout;

  public:
    struct_builder() {
      T array[N0][N1][N2];
      layout.register_struct(array);
      layout.register_vector(&array[0][0][0], N0 * N1 * N2);
      base::define_struct(layout);
    }
  };

  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
  class struct_builder<T[N0][N1][N2][N3]> : public base_struct_builder<T[N0][N1][N2][N3]> {
    using base = base_struct_builder<T[N0][N1][N2][N3]>;
    struct_layout<T[N0][N1][N2][N3]> layout;

  public:
    struct_builder() {
      T array[N0][N1][N2][N3];
      layout.register_struct(array);
      layout.register_vector(&array[0][0][0][0], N0 * N1 * N2 * N3);
      base::define_struct(layout);
    }
  };

  //--------------------------------------------------------------------

  template<typename T, std::size_t N>
  class struct_builder<std::array<T, N>> : public base_struct_builder<std::array<T, N>> {
    using base = base_struct_builder<std::array<T, N>>;
    struct_layout<std::array<T, N>> layout;

  public:
    struct_builder() {
      std::array<T, N> array;
      layout.register_struct(array);
      layout.register_vector(array.data(), N);
      base::define_struct(layout);
    }
  };

  //--------------------------------------------------------------------

  namespace detail {

    template<typename T, typename Enable = void>
    class datatype_traits_impl {
    public:
      static MPI_Datatype get_datatype() {
        static struct_builder<T> builder;
        return builder.type;
      }
      using data_type_category = typename struct_builder<T>::data_type_category;
    };

    template<typename T>
    class datatype_traits_impl<T, typename std::enable_if<std::is_enum<T>::value>::type> {
      using underlying = typename std::underlying_type<T>::type;

    public:
      static MPI_Datatype get_datatype() { return datatype_traits<underlying>::get_datatype(); }
      using data_type_category = typename datatype_traits<underlying>::data_type_category;
    };

#if defined MPL_HOMOGENEOUS
    template<typename T>
    class datatype_traits_impl<
        T, typename std::enable_if<
               std::is_trivially_copyable<T>::value and std::is_copy_assignable<T>::value and
               not std::is_enum<T>::value and not std::is_array<T>::value>::type> {
    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits_impl<unsigned char[sizeof(T)]>::get_datatype();
      }
      using data_type_category = typename datatype_traits_impl<T>::data_type_category;
    };
#endif

  }  // namespace detail

  template<typename T>
  class datatype_traits {
  public:
    static MPI_Datatype get_datatype() {
      return detail::datatype_traits_impl<T>::get_datatype();
    }
    using data_type_category = typename detail::datatype_traits_impl<T>::data_type_category;
  };

#define MPL_DATATYPE_TRAITS(type, mpi_type)                      \
  template<>                                                     \
  class datatype_traits<type> {                                  \
  public:                                                        \
    static MPI_Datatype get_datatype() { return mpi_type; }      \
    using data_type_category = detail::basic_or_fixed_size_type; \
  }

  MPL_DATATYPE_TRAITS(char, MPI_CHAR);

  MPL_DATATYPE_TRAITS(signed char, MPI_SIGNED_CHAR);

  MPL_DATATYPE_TRAITS(unsigned char, MPI_UNSIGNED_CHAR);

  MPL_DATATYPE_TRAITS(wchar_t, MPI_WCHAR);

  MPL_DATATYPE_TRAITS(signed short int, MPI_SHORT);

  MPL_DATATYPE_TRAITS(unsigned short int, MPI_UNSIGNED_SHORT);

  MPL_DATATYPE_TRAITS(signed int, MPI_INT);

  MPL_DATATYPE_TRAITS(unsigned int, MPI_UNSIGNED);

  MPL_DATATYPE_TRAITS(signed long, MPI_LONG);

  MPL_DATATYPE_TRAITS(unsigned long, MPI_UNSIGNED_LONG);

  MPL_DATATYPE_TRAITS(signed long long, MPI_LONG_LONG);

  MPL_DATATYPE_TRAITS(unsigned long long, MPI_UNSIGNED_LONG_LONG);

  MPL_DATATYPE_TRAITS(bool, MPI_CXX_BOOL);

  MPL_DATATYPE_TRAITS(float, MPI_FLOAT);

  MPL_DATATYPE_TRAITS(double, MPI_DOUBLE);

  MPL_DATATYPE_TRAITS(long double, MPI_LONG_DOUBLE);

#if __cplusplus >= 201703L
  MPL_DATATYPE_TRAITS(std::byte, MPI_BYTE);
#endif

  MPL_DATATYPE_TRAITS(std::complex<float>, MPI_CXX_FLOAT_COMPLEX);

  MPL_DATATYPE_TRAITS(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);

  MPL_DATATYPE_TRAITS(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);

#undef MPL_DATATYPE_TRAITS

  template<>
  class datatype_traits<char16_t> {
  public:
    static MPI_Datatype get_datatype() {
      return datatype_traits<std::uint_least16_t>::get_datatype();
    }
    using data_type_category = detail::basic_or_fixed_size_type;
  };

  template<>
  class datatype_traits<char32_t> {
  public:
    static MPI_Datatype get_datatype() {
      return datatype_traits<std::uint_least32_t>::get_datatype();
    }
    using data_type_category = detail::basic_or_fixed_size_type;
  };

  template<typename T, typename A>
  class datatype_traits<std::vector<T, A>> {
  public:
    using data_type_category = detail::contigous_stl_container;
  };

  template<typename A>
  class datatype_traits<std::vector<bool, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename A>
  class datatype_traits<std::deque<T, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename A>
  class datatype_traits<std::forward_list<T, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename A>
  class datatype_traits<std::list<T, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::set<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::map<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::multiset<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::multimap<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::unordered_set<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::unordered_map<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::unordered_multiset<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename C, typename A>
  class datatype_traits<std::unordered_multimap<T, C, A>> {
  public:
    using data_type_category = detail::stl_container;
  };

  template<typename T, typename Trait, typename Char>
  class datatype_traits<std::basic_string<T, Trait, Char>> {
  public:
    using data_type_category = detail::contigous_const_stl_container;
  };

  template<typename T>
  class datatype_traits<std::valarray<T>> {
  public:
    using data_type_category = detail::contigous_stl_container;
  };

}  // namespace mpl

#define MPL_GET_NTH_ARG(                                                                      \
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, \
    _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, \
    _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, \
    _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, \
    _92, _93, _94, _95, _96, _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107,   \
    _108, _109, _110, _111, _112, _113, _114, _115, _116, _117, _118, _119, N, ...)           \
  N

#define MPL_FE_0(MPL_CALL, ...)
#define MPL_FE_1(MPL_CALL, x) MPL_CALL(x)
#define MPL_FE_2(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_1(MPL_CALL, __VA_ARGS__)
#define MPL_FE_3(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_2(MPL_CALL, __VA_ARGS__)
#define MPL_FE_4(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_3(MPL_CALL, __VA_ARGS__)
#define MPL_FE_5(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_4(MPL_CALL, __VA_ARGS__)
#define MPL_FE_6(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_5(MPL_CALL, __VA_ARGS__)
#define MPL_FE_7(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_6(MPL_CALL, __VA_ARGS__)
#define MPL_FE_8(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_7(MPL_CALL, __VA_ARGS__)
#define MPL_FE_9(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_8(MPL_CALL, __VA_ARGS__)
#define MPL_FE_10(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_9(MPL_CALL, __VA_ARGS__)
#define MPL_FE_11(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_10(MPL_CALL, __VA_ARGS__)
#define MPL_FE_12(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_11(MPL_CALL, __VA_ARGS__)
#define MPL_FE_13(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_12(MPL_CALL, __VA_ARGS__)
#define MPL_FE_14(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_13(MPL_CALL, __VA_ARGS__)
#define MPL_FE_15(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_14(MPL_CALL, __VA_ARGS__)
#define MPL_FE_16(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_15(MPL_CALL, __VA_ARGS__)
#define MPL_FE_17(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_16(MPL_CALL, __VA_ARGS__)
#define MPL_FE_18(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_17(MPL_CALL, __VA_ARGS__)
#define MPL_FE_19(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_18(MPL_CALL, __VA_ARGS__)
#define MPL_FE_20(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_19(MPL_CALL, __VA_ARGS__)
#define MPL_FE_21(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_20(MPL_CALL, __VA_ARGS__)
#define MPL_FE_22(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_21(MPL_CALL, __VA_ARGS__)
#define MPL_FE_23(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_22(MPL_CALL, __VA_ARGS__)
#define MPL_FE_24(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_23(MPL_CALL, __VA_ARGS__)
#define MPL_FE_25(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_24(MPL_CALL, __VA_ARGS__)
#define MPL_FE_26(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_25(MPL_CALL, __VA_ARGS__)
#define MPL_FE_27(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_26(MPL_CALL, __VA_ARGS__)
#define MPL_FE_28(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_27(MPL_CALL, __VA_ARGS__)
#define MPL_FE_29(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_28(MPL_CALL, __VA_ARGS__)
#define MPL_FE_30(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_29(MPL_CALL, __VA_ARGS__)
#define MPL_FE_31(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_30(MPL_CALL, __VA_ARGS__)
#define MPL_FE_32(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_31(MPL_CALL, __VA_ARGS__)
#define MPL_FE_33(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_32(MPL_CALL, __VA_ARGS__)
#define MPL_FE_34(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_33(MPL_CALL, __VA_ARGS__)
#define MPL_FE_35(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_34(MPL_CALL, __VA_ARGS__)
#define MPL_FE_36(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_35(MPL_CALL, __VA_ARGS__)
#define MPL_FE_37(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_36(MPL_CALL, __VA_ARGS__)
#define MPL_FE_38(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_37(MPL_CALL, __VA_ARGS__)
#define MPL_FE_39(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_38(MPL_CALL, __VA_ARGS__)
#define MPL_FE_40(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_39(MPL_CALL, __VA_ARGS__)
#define MPL_FE_41(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_40(MPL_CALL, __VA_ARGS__)
#define MPL_FE_42(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_41(MPL_CALL, __VA_ARGS__)
#define MPL_FE_43(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_42(MPL_CALL, __VA_ARGS__)
#define MPL_FE_44(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_43(MPL_CALL, __VA_ARGS__)
#define MPL_FE_45(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_44(MPL_CALL, __VA_ARGS__)
#define MPL_FE_46(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_45(MPL_CALL, __VA_ARGS__)
#define MPL_FE_47(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_46(MPL_CALL, __VA_ARGS__)
#define MPL_FE_48(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_47(MPL_CALL, __VA_ARGS__)
#define MPL_FE_49(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_48(MPL_CALL, __VA_ARGS__)
#define MPL_FE_50(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_49(MPL_CALL, __VA_ARGS__)
#define MPL_FE_51(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_50(MPL_CALL, __VA_ARGS__)
#define MPL_FE_52(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_51(MPL_CALL, __VA_ARGS__)
#define MPL_FE_53(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_52(MPL_CALL, __VA_ARGS__)
#define MPL_FE_54(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_53(MPL_CALL, __VA_ARGS__)
#define MPL_FE_55(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_54(MPL_CALL, __VA_ARGS__)
#define MPL_FE_56(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_55(MPL_CALL, __VA_ARGS__)
#define MPL_FE_57(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_56(MPL_CALL, __VA_ARGS__)
#define MPL_FE_58(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_57(MPL_CALL, __VA_ARGS__)
#define MPL_FE_59(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_58(MPL_CALL, __VA_ARGS__)
#define MPL_FE_60(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_59(MPL_CALL, __VA_ARGS__)
#define MPL_FE_61(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_60(MPL_CALL, __VA_ARGS__)
#define MPL_FE_62(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_61(MPL_CALL, __VA_ARGS__)
#define MPL_FE_63(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_62(MPL_CALL, __VA_ARGS__)
#define MPL_FE_64(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_63(MPL_CALL, __VA_ARGS__)
#define MPL_FE_65(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_64(MPL_CALL, __VA_ARGS__)
#define MPL_FE_66(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_65(MPL_CALL, __VA_ARGS__)
#define MPL_FE_67(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_66(MPL_CALL, __VA_ARGS__)
#define MPL_FE_68(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_67(MPL_CALL, __VA_ARGS__)
#define MPL_FE_69(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_68(MPL_CALL, __VA_ARGS__)
#define MPL_FE_70(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_69(MPL_CALL, __VA_ARGS__)
#define MPL_FE_71(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_70(MPL_CALL, __VA_ARGS__)
#define MPL_FE_72(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_71(MPL_CALL, __VA_ARGS__)
#define MPL_FE_73(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_72(MPL_CALL, __VA_ARGS__)
#define MPL_FE_74(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_73(MPL_CALL, __VA_ARGS__)
#define MPL_FE_75(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_74(MPL_CALL, __VA_ARGS__)
#define MPL_FE_76(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_75(MPL_CALL, __VA_ARGS__)
#define MPL_FE_77(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_76(MPL_CALL, __VA_ARGS__)
#define MPL_FE_78(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_77(MPL_CALL, __VA_ARGS__)
#define MPL_FE_79(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_78(MPL_CALL, __VA_ARGS__)
#define MPL_FE_80(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_79(MPL_CALL, __VA_ARGS__)
#define MPL_FE_81(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_80(MPL_CALL, __VA_ARGS__)
#define MPL_FE_82(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_81(MPL_CALL, __VA_ARGS__)
#define MPL_FE_83(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_82(MPL_CALL, __VA_ARGS__)
#define MPL_FE_84(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_83(MPL_CALL, __VA_ARGS__)
#define MPL_FE_85(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_84(MPL_CALL, __VA_ARGS__)
#define MPL_FE_86(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_85(MPL_CALL, __VA_ARGS__)
#define MPL_FE_87(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_86(MPL_CALL, __VA_ARGS__)
#define MPL_FE_88(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_87(MPL_CALL, __VA_ARGS__)
#define MPL_FE_89(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_88(MPL_CALL, __VA_ARGS__)
#define MPL_FE_90(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_89(MPL_CALL, __VA_ARGS__)
#define MPL_FE_91(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_90(MPL_CALL, __VA_ARGS__)
#define MPL_FE_92(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_91(MPL_CALL, __VA_ARGS__)
#define MPL_FE_93(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_92(MPL_CALL, __VA_ARGS__)
#define MPL_FE_94(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_93(MPL_CALL, __VA_ARGS__)
#define MPL_FE_95(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_94(MPL_CALL, __VA_ARGS__)
#define MPL_FE_96(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_95(MPL_CALL, __VA_ARGS__)
#define MPL_FE_97(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_96(MPL_CALL, __VA_ARGS__)
#define MPL_FE_98(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_97(MPL_CALL, __VA_ARGS__)
#define MPL_FE_99(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_98(MPL_CALL, __VA_ARGS__)
#define MPL_FE_100(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_99(MPL_CALL, __VA_ARGS__)
#define MPL_FE_101(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_100(MPL_CALL, __VA_ARGS__)
#define MPL_FE_102(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_101(MPL_CALL, __VA_ARGS__)
#define MPL_FE_103(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_102(MPL_CALL, __VA_ARGS__)
#define MPL_FE_104(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_103(MPL_CALL, __VA_ARGS__)
#define MPL_FE_105(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_104(MPL_CALL, __VA_ARGS__)
#define MPL_FE_106(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_105(MPL_CALL, __VA_ARGS__)
#define MPL_FE_107(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_106(MPL_CALL, __VA_ARGS__)
#define MPL_FE_108(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_107(MPL_CALL, __VA_ARGS__)
#define MPL_FE_109(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_108(MPL_CALL, __VA_ARGS__)
#define MPL_FE_110(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_109(MPL_CALL, __VA_ARGS__)
#define MPL_FE_111(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_110(MPL_CALL, __VA_ARGS__)
#define MPL_FE_112(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_111(MPL_CALL, __VA_ARGS__)
#define MPL_FE_113(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_112(MPL_CALL, __VA_ARGS__)
#define MPL_FE_114(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_113(MPL_CALL, __VA_ARGS__)
#define MPL_FE_115(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_114(MPL_CALL, __VA_ARGS__)
#define MPL_FE_116(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_115(MPL_CALL, __VA_ARGS__)
#define MPL_FE_117(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_116(MPL_CALL, __VA_ARGS__)
#define MPL_FE_118(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_117(MPL_CALL, __VA_ARGS__)
#define MPL_FE_119(MPL_CALL, x, ...) MPL_CALL(x) MPL_FE_118(MPL_CALL, __VA_ARGS__)
#define MPL_CALL_MACRO_X_FOR_EACH(x, ...)                                                      \
  MPL_GET_NTH_ARG(                                                                             \
      "ignored", ##__VA_ARGS__, MPL_FE_119, MPL_FE_118, MPL_FE_117, MPL_FE_116, MPL_FE_115,    \
      MPL_FE_114, MPL_FE_113, MPL_FE_112, MPL_FE_111, MPL_FE_110, MPL_FE_109, MPL_FE_108,      \
      MPL_FE_107, MPL_FE_106, MPL_FE_105, MPL_FE_104, MPL_FE_103, MPL_FE_102, MPL_FE_101,      \
      MPL_FE_100, MPL_FE_99, MPL_FE_98, MPL_FE_97, MPL_FE_96, MPL_FE_95, MPL_FE_94, MPL_FE_93, \
      MPL_FE_92, MPL_FE_91, MPL_FE_90, MPL_FE_89, MPL_FE_88, MPL_FE_87, MPL_FE_86, MPL_FE_85,  \
      MPL_FE_84, MPL_FE_83, MPL_FE_82, MPL_FE_81, MPL_FE_80, MPL_FE_79, MPL_FE_78, MPL_FE_77,  \
      MPL_FE_76, MPL_FE_75, MPL_FE_74, MPL_FE_73, MPL_FE_72, MPL_FE_71, MPL_FE_70, MPL_FE_69,  \
      MPL_FE_68, MPL_FE_67, MPL_FE_66, MPL_FE_65, MPL_FE_64, MPL_FE_63, MPL_FE_62, MPL_FE_61,  \
      MPL_FE_60, MPL_FE_59, MPL_FE_58, MPL_FE_57, MPL_FE_56, MPL_FE_55, MPL_FE_54, MPL_FE_53,  \
      MPL_FE_52, MPL_FE_51, MPL_FE_50, MPL_FE_49, MPL_FE_48, MPL_FE_47, MPL_FE_46, MPL_FE_45,  \
      MPL_FE_44, MPL_FE_43, MPL_FE_42, MPL_FE_41, MPL_FE_40, MPL_FE_39, MPL_FE_38, MPL_FE_37,  \
      MPL_FE_36, MPL_FE_35, MPL_FE_34, MPL_FE_33, MPL_FE_32, MPL_FE_31, MPL_FE_30, MPL_FE_29,  \
      MPL_FE_28, MPL_FE_27, MPL_FE_26, MPL_FE_25, MPL_FE_24, MPL_FE_23, MPL_FE_22, MPL_FE_21,  \
      MPL_FE_20, MPL_FE_19, MPL_FE_18, MPL_FE_17, MPL_FE_16, MPL_FE_15, MPL_FE_14, MPL_FE_13,  \
      MPL_FE_12, MPL_FE_11, MPL_FE_10, MPL_FE_9, MPL_FE_8, MPL_FE_7, MPL_FE_6, MPL_FE_5,       \
      MPL_FE_4, MPL_FE_3, MPL_FE_2, MPL_FE_1, MPL_FE_0)                                        \
  (x, ##__VA_ARGS__)

#define MPL_REGISTER(element) layout.register_element(str.element);

#define MPL_REFLECTION(STRUCT, ...)                                     \
  namespace mpl {                                                       \
    template<>                                                          \
    class struct_builder<STRUCT> : public base_struct_builder<STRUCT> { \
      struct_layout<STRUCT> layout;                                     \
                                                                        \
    public:                                                             \
      struct_builder() {                                                \
        STRUCT str;                                                     \
        layout.register_struct(str);                                    \
        MPL_CALL_MACRO_X_FOR_EACH(MPL_REGISTER, __VA_ARGS__)            \
        define_struct(layout);                                          \
      }                                                                 \
    };                                                                  \
  }

#endif
