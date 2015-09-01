#if !(defined MPL_OPERATOR_HPP)

#define MPL_OPERATOR_HPP

#include <mpi.h>
#include <functional>

namespace mpl {

  template<typename T>
  struct max : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return (x<y) ? y : x;
    }
  };

  template<typename T>
  struct min : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return !(y<x) ? x : y;
    }
  };

  template<typename T>
  struct plus : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x+y;
    }
  };

  template<typename T>
  struct multiplies : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x*y;
    }
  };

  template<typename T>
  struct logical_and : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x and y;
    }
  };

  template<typename T>
  struct logical_or : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x or y;
    }
  };

  template<typename T>
  struct logical_xor : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x xor y;
    }
  };

  template<typename T>
  struct bit_and : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x & y;
    }
  };

  template<typename T>
  struct bit_or : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x | y;
    }
  };

  template<typename T>
  struct bit_xor : public std::function<T (T, T)> {
    T operator()(const T &x, const T &y) {
      return x ^ y;
    }
  };
  
  // -------------------------------------------------------------------

  template<typename F>
  struct op_traits {
    static constexpr bool is_commutative() {
      return false;
    }
  };

  template<typename T>
  struct op_traits<max<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<min<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<plus<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<multiplies<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<logical_and<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<logical_or<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<logical_xor<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<bit_and<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<bit_or<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  template<typename T>
  struct op_traits<bit_xor<T> > {
    static constexpr bool is_commutative() {
      return true;
    }
  };

  // -------------------------------------------------------------------

  namespace detail {
 
    template<typename F>
    class op : public F {
    public:
      typedef F functor;
      typedef typename F::first_argument_type first_argument_type;
      typedef typename F::second_argument_type second_argument_type;
      typedef typename F::result_type result_type;
      static F f;
      static void apply(void *invec, void *inoutvec, int *len, 
			MPI_Datatype *datatype) {
	first_argument_type *i1=reinterpret_cast<first_argument_type *>(invec);
	second_argument_type *i2=reinterpret_cast<second_argument_type *>(inoutvec);
	for (int i=0, i_end=*len; i<i_end; ++i, ++i1, ++i2)
	  *i2=f(*i1, *i2);
      }
      MPI_Op mpi_op;
      op() {
	MPI_Op_create(op::apply, op_traits<F>::is_commutative(), &mpi_op);
      }
      ~op() {
	MPI_Op_free(&mpi_op);
      }
      static constexpr bool is_s_commutative() {
	return op_traits<F>::is_commutative();
      }
      op(op const &)=delete;
      op& operator=(op const&)=delete;
    };
    
    template<typename F>
    F op<F>::f;
    
  }
  
}

#endif
