#if !(defined MPL_OPERATOR_HPP)

#define MPL_OPERATOR_HPP

#include <mpi.h>
#include <functional>
#include <type_traits>
#include <memory>
#include <ciso646>


namespace mpl {

  /// Function object for calculating the maximum of two values in reduction operations
  /// as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct max {
    /// \param x first argument
    /// \param y second argument
    /// \return maximum of the two arguments
    T operator()(const T &x, const T &y) const {
      return (x < y) ? y : x;
    }
  };

  /// Function object for calculating the minimum of two values in reduction operations
  /// as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct min {
    /// \param x first argument
    /// \param y second argument
    /// \return minimum of the two arguments
    T operator()(const T &x, const T &y) const {
      return not(y < x) ? x : y;
    }
  };

  /// Function object for calculating the sum of two values in reduction operations as
  /// communicator::reduce.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct plus {
    /// \param x first argument
    /// \param y second argument
    /// \return sum of the two arguments
    T operator()(const T &x, const T &y) const {
      return x + y;
    }
  };

  /// Function object for calculating the product of two values in reduction operations
  /// as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct multiplies {
    /// \param x first argument
    /// \param y second argument
    /// \return product of the two arguments
    T operator()(const T &x, const T &y) const {
      return x * y;
    }
  };

  /// Function object for calculating the logical conjunction of two values in reduction
  /// operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct logical_and {
    /// \param x first argument
    /// \param y second argument
    /// \return logical conjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x and y;
    }
  };

  /// Function object for calculating the logical (inclusive) disjunction of two values
  /// in reduction operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct logical_or {
    /// \param x first argument
    /// \param y second argument
    /// \return logical (inclusive) disjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x or y;
    }
  };

  /// Function object for calculating the logical exclusive disjunction of two values in
  /// reduction operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct logical_xor {
    /// \param x first argument
    /// \param y second argument
    /// \return logical exclusive disjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x xor y;
    }
  };

  /// Function object for calculating the bitwise conjunction of two values in reduction
  /// operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct bit_and {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise conjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x & y;
    }
  };

  /// Function object for calculating the bitwise (inclusive) disjunction of two values
  /// in reduction operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct bit_or {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise (inclusive) disjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x | y;
    }
  };

  /// Function object for calculating the bitwise exclusive disjunction of two values in
  /// reduction operations as <tt>communicator::reduce</tt>.
  /// \tparam T data type of the reduction operation's arguments and its result
  template<typename T>
  struct bit_xor {
    /// \param x first argument
    /// \param y second argument
    /// \return bitwise exclusive disjunction of the two arguments
    T operator()(const T &x, const T &y) const {
      return x ^ y;
    }
  };

  // -------------------------------------------------------------------

  /// Traits class for storing meta information about reduction operations.
  /// \tparam F function object type
  template<typename F>
  struct op_traits {
    /// Is true if reduction operation specified in the template parameter \c F is commutative.
    static constexpr bool is_commutative = false;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c max reduction operation.
  template<typename T>
  struct op_traits<max<T>> {
    /// The <tt>\ref max</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c min reduction operation.
  template<typename T>
  struct op_traits<min<T>> {
    /// The <tt>\ref min</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c plus reduction operation.
  template<typename T>
  struct op_traits<plus<T>> {
    /// The <tt>\ref plus</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c multiplies reduction operation.
  template<typename T>
  struct op_traits<multiplies<T>> {
    /// The <tt>\ref multiplies</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c logical_and reduction operation.
  template<typename T>
  struct op_traits<logical_and<T>> {
    /// The <tt>\ref logical_and</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c logical_or reduction operation.
  template<typename T>
  struct op_traits<logical_or<T>> {
    /// The <tt>\ref logical_or</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c logical_xor reduction operation.
  template<typename T>
  struct op_traits<logical_xor<T>> {
    /// The <tt>\ref logical_xor</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c bit_and reduction operation.
  template<typename T>
  struct op_traits<bit_and<T>> {
    /// The <tt>\ref bit_and</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c bit_or reduction operation.
  template<typename T>
  struct op_traits<bit_or<T>> {
    /// The <tt>\ref bit_or</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  /// Specialization of traits class \c op_traits for storing meta information about
  /// the \c bit_xor reduction operation.
  template<typename T>
  struct op_traits<bit_xor<T>> {
    /// The <tt>\ref bit_xor</tt> reduction operation is commutative.
    static constexpr bool is_commutative = true;
  };

  // -------------------------------------------------------------------

  namespace detail {

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

      static_assert(is_binary_functor<T, F>::value,
                    "reduction operator must be a binary function");
      static_assert(not std::is_pointer_v<F>, "functor must not be function pointer");

      static constexpr bool is_commutative = op_traits<functor>::is_commutative;
      static std::unique_ptr<functor> f;

      static void apply(void *in_vector, void *in_out_vector, int *len,
                        [[maybe_unused]] MPI_Datatype *datatype) {
        auto *i_1{static_cast<T *>(in_vector)};
        auto *i_2{static_cast<T *>(in_out_vector)};
        for (int i{0}, i_end{*len}; i < i_end; ++i, ++i_1, ++i_2)
          *i_2 = (*f)(*i_1, *i_2);
      }

      MPI_Op mpi_op{MPI_OP_NULL};

    private:
      explicit op(F f_) {
        f.reset(new F(f_));
        MPI_Op_create(op::apply, is_commutative, &mpi_op);
      }

    public:
      op(op const &) = delete;

      ~op() {
        MPI_Op_free(&mpi_op);
      }

      void operator=(op const &) = delete;

      friend op &get_op<>(F);
    };

    template<typename T, typename F>
    std::unique_ptr<F> op<T, F>::f;

  }  // namespace detail

}  // namespace mpl

#endif
