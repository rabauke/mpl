#if !(defined MPL_OPERATOR_HPP)

#define MPL_OPERATOR_HPP

#include <mpi.h>
#include <functional>
#include <type_traits>
#include <memory>
#include <ciso646>

namespace mpl {

  /// Function object for calculating the maximum of two values.
  template<typename T>
  struct max {
    /// \param x first argument
    /// \param y second argument
    /// \return maximum of the two arguments
    T operator()(const T &x, const T &y) const { return (x < y) ? y : x; }
  };

  /// Function object for calculating the minimum of two values.
  template<typename T>
  struct min {
    /// \param x first argument
    /// \param y second argument
    /// \return minimum of the two arguments
    T operator()(const T &x, const T &y) const { return not(y < x) ? x : y; }
  };

  /// Function object for calculating the sum of two values.
  template<typename T>
  struct plus {
    /// \param x first argument
    /// \param y second argument
    /// \return sum of the two arguments
    T operator()(const T &x, const T &y) const { return x + y; }
  };

  /// Function object for calculating the product of two values.
  template<typename T>
  struct multiplies {
    /// \param x first argument
    /// \param y second argument
    /// \return product of the two arguments
    T operator()(const T &x, const T &y) const { return x * y; }
  };

  /// Function object for calculating the logical conjunction of two values.
  template<typename T>
  struct logical_and {
    /// \param x first argument
    /// \param y second argument
    /// \return logical conjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x and y; }
  };

  /// Function object for calculating the logical (inclusive) disjunction of two values.
  template<typename T>
  struct logical_or {
    /// \param x first argument
    /// \param y second argument
    /// \return logical (inclusive) disjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x or y; }
  };

  /// Function object for calculating the logical exclusive disjunction of two values.
  template<typename T>
  struct logical_xor {
    /// \param x first argument
    /// \param y second argument
    /// \return logical exclusive disjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x xor y; }
  };

  /// Function object for calculating the bitwise conjunction of two values.
  template<typename T>
  struct bit_and {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise conjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x & y; }
  };

  /// Function object for calculating the bitwise (inclusive) disjunction of two values.
  template<typename T>
  struct bit_or {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise (inclusive) disjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x | y; }
  };

  /// Function object for calculating the bitwise exclusive disjunction of two values.
  template<typename T>
  struct bit_xor {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise exclusive disjunction of the two arguments
    T operator()(const T &x, const T &y) const { return x ^ y; }
  };

  // -------------------------------------------------------------------

  template<typename F>
  struct op_traits {
    static constexpr bool is_commutative = false;
  };

  template<typename T>
  struct op_traits<max<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<min<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<plus<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<multiplies<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<logical_and<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<logical_or<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<logical_xor<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<bit_and<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<bit_or<T>> {
    static constexpr bool is_commutative = true;
  };

  template<typename T>
  struct op_traits<bit_xor<T>> {
    static constexpr bool is_commutative = true;
  };

  // -------------------------------------------------------------------

  namespace detail {

    template<typename T>
    struct remove_class {};
    template<typename C, typename R, typename... A>
    struct remove_class<R (C::*)(A...)> {
      using type = R(A...);
    };
    template<typename C, typename R, typename... A>
    struct remove_class<R (C::*)(A...) const> {
      using type = R(A...);
    };
    template<typename C, typename R, typename... A>
    struct remove_class<R (C::*)(A...) volatile> {
      using type = R(A...);
    };
    template<typename C, typename R, typename... A>
    struct remove_class<R (C::*)(A...) const volatile> {
      using type = R(A...);
    };

    template<typename T>
    struct get_signature_impl {
      using type =
          typename remove_class<decltype(&std::remove_reference<T>::type::operator())>::type;
    };
    template<typename R, typename... A>
    struct get_signature_impl<R(A...)> {
      using type = R(A...);
    };
    template<typename R, typename... A>
    struct get_signature_impl<R (&)(A...)> {
      using type = R(A...);
    };
    template<typename R, typename... A>
    struct get_signature_impl<R (*)(A...)> {
      using type = R(A...);
    };
    template<typename T>
    using get_signature = typename get_signature_impl<T>::type;

    template<typename T>
    struct get_result_type_impl;
    template<typename R, typename... A>
    struct get_result_type_impl<R(A...)> {
      using type = R;
    };
    template<typename T>
    using get_result_type = typename get_result_type_impl<T>::type;

    template<typename T>
    struct get_first_argument_type_impl;
    template<typename R, typename A1, typename... A>
    struct get_first_argument_type_impl<R(A1, A...)> {
      using type = A1;
    };
    template<typename T>
    using get_first_argument_type = typename get_first_argument_type_impl<T>::type;

    template<typename T>
    struct get_second_argument_type_impl;
    template<typename R, typename A1, typename A2, typename... A>
    struct get_second_argument_type_impl<R(A1, A2, A...)> {
      using type = A2;
    };
    template<typename T>
    using get_second_argument_type = typename get_second_argument_type_impl<T>::type;

    //------------------------------------------------------------------

    template<typename T, typename F>
    class op;

    template<typename T, typename F>
    inline op<T, F> &get_op(F f) {
      static op<T, F> op_(f);
      return op_;
    }

    template<typename T, typename F>
    class op {
    public:
      using functor = F;
      using signature = get_signature<functor>;
      using first_argument_type = typename std::decay<get_first_argument_type<signature>>::type;
      using second_argument_type =
          typename std::decay<get_second_argument_type<signature>>::type;
      using result_type = typename std::decay<get_result_type<signature>>::type;
      static_assert(
          std::is_assignable<typename std::add_lvalue_reference<first_argument_type>::type,
                             T>::value and
              std::is_assignable<typename std::add_lvalue_reference<second_argument_type>::type,
                                 T>::value and
              std::is_assignable<T &, result_type>::value,
          "argument type mismatch");
      static_assert(!std::is_pointer<F>::value, "functor must not be function pointer");

      static constexpr bool is_commutative = op_traits<functor>::is_commutative;
      static std::unique_ptr<functor> f;

      static void apply(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
        T *i1 = reinterpret_cast<T *>(invec);
        T *i2 = reinterpret_cast<T *>(inoutvec);
        for (int i{0}, i_end{*len}; i < i_end; ++i, ++i1, ++i2)
          *i2 = (*f)(*i1, *i2);
      }

      MPI_Op mpi_op{MPI_OP_NULL};

    private:
      explicit op(F f_) {
        f.reset(new F(f_));
        MPI_Op_create(op::apply, is_commutative, &mpi_op);
      }

    public:
      op(op const &) = delete;

      ~op() { MPI_Op_free(&mpi_op); }

      void operator=(op const &) = delete;

      friend op &get_op<>(F);
    };

    template<typename T, typename F>
    std::unique_ptr<F> op<T, F>::f;

  }  // namespace detail

}  // namespace mpl

#endif
