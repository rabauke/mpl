#define BOOST_TEST_MODULE scan

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool scan_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.scan(add<T>(), x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool scan_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.scan(mpl::plus<T>(), x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool scan_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  comm_world.scan([](T a, T b) { return a + b; }, x, y);
  return y == T((N * N + N) / 2);
}

template<typename T>
bool scan_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.scan(add<T>(), x);
  return x == T((N * N + N) / 2);
}

template<typename T>
bool scan_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.scan(mpl::plus<T>(), x);
  return x == T((N * N + N) / 2);
}

template<typename T>
bool scan_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  comm_world.scan([](T a, T b) { return a + b; }, x);
  return x == T((N * N + N) / 2);
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(scan_func_test<double>());
  BOOST_TEST(scan_op_test<double>());
  BOOST_TEST(scan_lambda_test<double>());
  BOOST_TEST(scan_inplace_func_test<double>());
  BOOST_TEST(scan_inplace_op_test<double>());
  BOOST_TEST(scan_inplace_lambda_test<double>());
}
