#if !(defined MPL_DATATYPE_HPP)

#define MPL_DATATYPE_HPP

#include <mpi.h>
#include <cstddef>
#include <vector>
#include <complex>
#include <utility>
#include <tuple>
#include <array>
#include <type_traits>

namespace mpl {

  //--- forward declarations -------------------------------------------

  template<typename T>
  struct datatype_traits;

  namespace detail {
    
    template<typename T, typename E>
    struct datatype_traits_impl;

  }

  template<typename T>
  class base_struct_builder;

  template<typename T>
  class struct_builder;

  //--------------------------------------------------------------------

  template<typename S>
  class struct_layout {
    template <typename T>
    inline std::size_t size(T) { 
      return sizeof(T); 
    }
    template <typename T>
    inline std::size_t size(T *) { 
      return sizeof(T); 
    }
    template <typename T>
    inline MPI_Datatype get_datatype(T) {
      return datatype_traits<T>().get_datatype(); 
    }
    template <typename T>
    inline MPI_Datatype get_datatype(T *) {
      return datatype_traits<T>().get_datatype(); 
    }
    MPI_Aint base;
    std::vector<int> blocklengths;
    std::vector<MPI_Aint> displacements;
    std::vector<MPI_Datatype> datatypes;
  public:
    struct_layout & register_struct(const S &x) {
      MPI_Get_address(const_cast<S *>(&x), &base);
      return *this;
    }
    template<typename T>
    struct_layout & register_element(T &x) {
      static_assert(not std::is_const<T>::value, "type must not be const");
      blocklengths.push_back(sizeof(x)/size(x));
      MPI_Aint address;
      MPI_Get_address(&x, &address);
      displacements.push_back(address-base);
      datatypes.push_back(get_datatype(x));
      return *this;
    }
    template<typename T>
    struct_layout & register_vector(T *x, std::ptrdiff_t N) {
      static_assert(not std::is_const<T>::value, "type must not be const");
      blocklengths.push_back(N);
      MPI_Aint address;
      MPI_Get_address(x, &address);
      displacements.push_back(address-base);
      datatypes.push_back(get_datatype(*x));
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
      MPI_Type_create_struct(str.blocklengths.size(),
			     str.blocklengths.data(),
 			     str.displacements.data(),
			     str.datatypes.data(), &temp_type);
      MPI_Type_commit(&temp_type);
      MPI_Type_create_resized(temp_type, 0, sizeof(T), &type);
      MPI_Type_commit(&type);
      MPI_Type_free(&temp_type);
    }
    base_struct_builder() {
    }
    base_struct_builder(const base_struct_builder &)=delete;
    base_struct_builder & operator=(const base_struct_builder &)=delete;
    ~base_struct_builder() {
      MPI_Type_free(&type);
    }
    friend class detail::datatype_traits_impl<T, void>;
  };

  //--------------------------------------------------------------------

  template<typename T1, typename T2>
  class struct_builder<std::pair<T1, T2>> : public base_struct_builder<std::pair<T1, T2>> {
    typedef base_struct_builder<std::pair<T1, T2>> base;
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
      apply_n(F &f) : f(f) {
      }
      void operator()(T &x) const {
	apply_n<F, T, n-1> next(f);
	next(x);
	f(std::get<n-1>(x));
      }
    };
    
    template<typename F, typename T>
    struct apply_n<F, T, 1> {
      F &f;
    public:
      apply_n(F &f) : f(f) {
      }
      void operator()(T &x) const {
	f(std::get<0>(x));
      }
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
      register_element(struct_layout<std::tuple<Ts...>> &layout) : layout(layout) {
      }
      template<typename T>
      void operator()(T &x) const {
	layout.register_element(x);
      }
    };

  }

  template<typename... Ts>
  class struct_builder<std::tuple<Ts...>> : public base_struct_builder<std::tuple<Ts...>> {
    typedef base_struct_builder<std::tuple<Ts...>> base;
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

  template<typename T, std::size_t N>
  class struct_builder<T[N]> : public base_struct_builder<T[N]> {
    typedef base_struct_builder<T[N]> base;
    struct_layout<T[N]> layout;
  public:
    struct_builder() {
      T array[N];
      layout.register_struct(array);
      layout.register_vector(array, N);
      base::define_struct(layout);
    }
  };

  template<typename T, std::size_t N0, std::size_t N1>
  class struct_builder<T[N0][N1]> : public base_struct_builder<T[N0][N1]> {
    typedef base_struct_builder<T[N0][N1]> base;
    struct_layout<T[N0][N1]> layout;
  public:
    struct_builder()  {
      T array[N0][N1];
      layout.register_struct(array);
      layout.register_vector(array, N0*N1);
      base::define_struct(layout);
    }
  };
  
  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2>
  class struct_builder<T[N0][N1][N2]> : public base_struct_builder<T[N0][N1][N2]> {
    typedef base_struct_builder<T[N0][N1][N2]> base;
    struct_layout<T[N0][N1][N2]> layout;
  public:
    struct_builder()  {
      T array[N0][N1][N2];
      layout.register_struct(array);
      layout.register_vector(array, N0*N1*N2);
      base::define_struct(layout);
    }
  };

  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
  class struct_builder<T[N0][N1][N2][N3]> : public base_struct_builder<T[N0][N1][N2][N3]> {
    typedef base_struct_builder<T[N0][N1][N2][N3]> base;
    struct_layout<T[N0][N1][N2][N3]> layout;
  public:
    struct_builder()  {
      T array[N0][N1][N2][N3];
      layout.register_struct(array);
      layout.register_vector(array, N0*N1*N2*N3);
      base::define_struct(layout);
    }
  };

  //--------------------------------------------------------------------

  template<typename T, std::size_t N>
  class struct_builder<std::array<T, N>> : public base_struct_builder<std::array<T, N>> {
    typedef base_struct_builder<std::array<T, N>> base;
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

    template<typename T, typename Enable=void>
    class datatype_traits_impl {
    public:
      static MPI_Datatype get_datatype() {
	static struct_builder<T> builder; 
	return builder.type;
      }
    };
    
    template<typename T>
    class datatype_traits_impl<T, typename std::enable_if<std::is_enum<T>::value>::type> {
    public:
      static MPI_Datatype get_datatype() {
	return datatype_traits<typename std::underlying_type<T>::type>::get_datatype();
      }
    };
    
#if defined MPL_HOMOGENEOUS
    template<typename T>
    class datatype_traits_impl<T, typename std::enable_if<std::is_trivially_copyable<T>::value 
							  and std::is_copy_assignable<T>::value 
							  and not std::is_enum<T>::value 
							  and not std::is_array<T>::value>::type> {
    public:
      static MPI_Datatype get_datatype() {
	return datatype_traits_impl<unsigned char[sizeof(T)]>::get_datatype();
      }
    };
#endif

  }
    
  template<typename T>
  class datatype_traits {
  public:
    static MPI_Datatype get_datatype() {
      return detail::datatype_traits_impl<T>::get_datatype();
    }
  };
  
#define MPL_DATATYPE_TRAITS(type, mpi_type)          \
  template<>				             \
  struct datatype_traits<type> {         	     \
    static constexpr MPI_Datatype get_datatype() {   \
      return mpi_type;				     \
    }						     \
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

  MPL_DATATYPE_TRAITS(std::complex<float>, MPI_CXX_FLOAT_COMPLEX);
  MPL_DATATYPE_TRAITS(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);
  MPL_DATATYPE_TRAITS(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);
  
#undef MPL_DATATYPE_TRAITS

}

#endif
