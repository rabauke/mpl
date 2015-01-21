#if !(defined MPL_DATATYPE_HPP)

#define MPL_DATATYPE_HPP

#include <mpi.h>
#include <cstddef>
#include <vector>
#include <complex>
#include <utility>
#include <tuple>

namespace mpl {
  
  template<typename T>
  struct datatype_traits;
  template<typename T>
  class base_struct_builder;

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
    template <typename T>
    inline MPI_Datatype get_datatype(const T *) {
      return datatype_traits<T>().get_datatype(); 
    }
    MPI_Aint base;
    std::vector<int> blocklengths;
    std::vector<MPI_Aint> displacements;
    std::vector<MPI_Datatype> datatypes;
  public:
    struct_layout(const S &x) {
      MPI_Get_address(const_cast<S *>(&x), &base);
    }
    template<typename T>
    struct_layout & operator()(const T &x) {
      blocklengths.push_back(sizeof(x)/size(x));
      MPI_Aint address;
      MPI_Get_address(const_cast<T *>(&x), &address);
      displacements.push_back(address-base);
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
      MPI_Type_create_struct(str.blocklengths.size(),
			     const_cast<int *>(&str.blocklengths[0]),
 			     const_cast<MPI_Aint *>(&str.displacements[0]),
			     const_cast<MPI_Datatype *>(&str.datatypes[0]), &temp_type);
      MPI_Type_commit(&temp_type);
      MPI_Type_create_resized(temp_type, 0, sizeof(T), &type);
      MPI_Type_commit(&type);
      MPI_Type_free(&temp_type);
    }
    ~base_struct_builder() {
      MPI_Type_free(&type);
    }
    friend class datatype_traits<T>;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class struct_builder;

  // template<typename T>
  // class struct_builder<std::complex<T> > : public base_struct_builder<std::complex<T> > {
  //   typedef base_struct_builder<std::complex<T> > base;
  //   T complex_number[2];
  //   struct_layout<std::complex<T> > layout;
  // public:
  //   struct_builder() : layout(*reinterpret_cast<std::complex<T> *>(&complex_number[0])) {
  //     layout(complex_number);
  //     base::define_struct(layout);
  //   }
  // };
    
  template<typename T1, typename T2>
  class struct_builder<std::pair<T1, T2> > : public base_struct_builder<std::pair<T1, T2> > {
    typedef base_struct_builder<std::pair<T1, T2> > base;
    std::pair<T1, T2> pair;
    struct_layout<std::pair<T1, T2> > layout;
  public:
    struct_builder() : layout(pair) {
      layout(pair.first);
      layout(pair.second);
      base::define_struct(layout);
    }
  };

  namespace detail {
    
    template<typename F, typename T, std::size_t n>
    class apply_n {
      F &f;
    public:
      apply_n(F &f) : f(f) {
      }
      void operator()(const T &x) {
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
      void operator()(const T &x) {
	f(std::get<0>(x));
      }
    };

    template<typename F, typename... Args>
    void apply(const std::tuple<Args...> &t, F &f) {
      apply_n<F, std::tuple<Args...>, std::tuple_size<std::tuple<Args...> >::value> app(f);
      app(t);
    }
    
    template<typename... Ts>
    class register_element {
      struct_layout<std::tuple<Ts...> > &layout;
    public:
      register_element(struct_layout<std::tuple<Ts...> > &layout) : layout(layout) {
      }
      template<typename T>
      void operator()(const T &x) const {
	layout(x);
      }
    };

  }

  template<typename... Ts>
  class struct_builder<std::tuple<Ts...> > : public base_struct_builder<std::tuple<Ts...> > {
    typedef base_struct_builder<std::tuple<Ts...> > base;
    std::tuple<Ts...> tuple;
    struct_layout<std::tuple<Ts...> > layout;
  public:
    struct_builder() : layout(tuple) {
      base::define_struct(layout);
      detail::register_element<Ts...> reg(layout);
      detail::apply<detail::register_element<Ts...> >(tuple, reg);
      base::define_struct(layout);
    }
  };
  
  //--------------------------------------------------------------------

  template<typename T>
  class datatype_traits {
  public:
    static MPI_Datatype get_datatype() {
      static struct_builder<T> builder; 
      return builder.type;
    }
  };
  
#define MPL_DATATYPE_TRAITS(type, mpi_type)          \
  template<>				             \
  struct datatype_traits<type> {		     \
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

// #if __cplusplus >= 201103L
// #include <cstdint>
//   MPL_DATATYPE_TRAITS(int8_t, MPI_INT8_T);
//   MPL_DATATYPE_TRAITS(uint8_t, MPI_UINT8_T);
//   MPL_DATATYPE_TRAITS(int16_t, MPI_INT16_T);
//   MPL_DATATYPE_TRAITS(uint16_t, MPI_UINT16_T);
//   MPL_DATATYPE_TRAITS(int32_t, MPI_INT32_T);
//   MPL_DATATYPE_TRAITS(uint32_t, MPI_UINT32_T);
//   MPL_DATATYPE_TRAITS(int64_t, MPI_INT64_T);
//   MPL_DATATYPE_TRAITS(uint64_t, MPI_UINT64_T);
// #endif

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
