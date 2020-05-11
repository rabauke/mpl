#define BOOST_TEST_MODULE reduce

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool reduce_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    comm_world.reduce(add<T>(), 0, x, y);
    return y == T((N * N + N) / 2);
  } else {
    comm_world.reduce(add<T>(), 0, x);
  }
  return true;
}

template<typename T>
bool reduce_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    comm_world.reduce(mpl::plus<T>(), 0, x, y);
    return y == T((N * N + N) / 2);
  } else {
    comm_world.reduce(mpl::plus<T>(), 0, x);
  }
  return true;
}

template<typename T>
bool reduce_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    comm_world.reduce([](T a, T b) { return a + b; }, 0, x, y);
    return y == T((N * N + N) / 2);
  } else {
    comm_world.reduce([](T a, T b) { return a + b; }, 0, x);
  }
  return true;
}

template<typename T>
bool reduce_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.reduce(add<T>{}, 0, x);
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

template<typename T>
bool reduce_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.reduce(mpl::plus<T>(), 0, x);
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

template<typename T>
bool reduce_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.reduce([](T a, T b) { return a + b; }, 0, x);
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

template<typename T>
bool allreduce_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  comm_world.allreduce(add<T>(), x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allreduce_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  comm_world.allreduce(mpl::plus<T>(), x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allreduce_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  comm_world.allreduce([](T a, T b) { return a + b; }, x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allreduce_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.allreduce(add<T>(), x);
  return x == T((N * N + N) / 2);
}

template<typename T>
bool allreduce_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.allreduce(mpl::plus<T>(), x);
  return x == T((N * N + N) / 2);
}

template<typename T>
bool allreduce_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  comm_world.allreduce([](T a, T b) { return a + b; }, x);
  return x == T((N * N + N) / 2);
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(reduce_func_test<double>());
  BOOST_TEST(reduce_op_test<double>());
  BOOST_TEST(reduce_lambda_test<double>());
  BOOST_TEST(reduce_inplace_func_test<double>());
  BOOST_TEST(reduce_inplace_op_test<double>());
  BOOST_TEST(reduce_inplace_lambda_test<double>());
  BOOST_TEST(allreduce_func_test<double>());
  BOOST_TEST(allreduce_op_test<double>());
  BOOST_TEST(allreduce_lambda_test<double>());
  BOOST_TEST(allreduce_inplace_func_test<double>());
  BOOST_TEST(allreduce_inplace_op_test<double>());
  BOOST_TEST(allreduce_inplace_lambda_test<double>());
}
