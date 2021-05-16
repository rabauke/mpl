#define BOOST_TEST_MODULE exscan

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
struct add {
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool exscan_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.exscan(add<T>(), x, y);
  return comm_world.rank() == 0 || y == T((N * N - N) / 2);
}

template<typename T>
bool exscan_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.exscan(mpl::plus<T>(), x, y);
  return comm_world.rank() == 0 || y == T((N * N - N) / 2);
}

template<typename T>
bool exscan_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.exscan([](T a, T b) { return a + b; }, x, y);
  return comm_world.rank() == 0 || y == T((N * N - N) / 2);
}

template<typename T>
bool exscan_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.exscan(add<T>(), x);
  return comm_world.rank() == 0 || x == T((N * N - N) / 2);
}

template<typename T>
bool exscan_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.exscan(mpl::plus<T>(), x);
  return comm_world.rank() == 0 || x == T((N * N - N) / 2);
}

template<typename T>
bool exscan_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.exscan([](T a, T b) { return a + b; }, x);
  return comm_world.rank() == 0 || x == T((N * N - N) / 2);
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(exscan_func_test<double>());
  BOOST_TEST(exscan_op_test<double>());
  BOOST_TEST(exscan_lambda_test<double>());
  BOOST_TEST(exscan_inplace_func_test<double>());
  BOOST_TEST(exscan_inplace_op_test<double>());
  BOOST_TEST(exscan_inplace_lambda_test<double>());
}
