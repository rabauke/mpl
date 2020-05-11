#define BOOST_TEST_MODULE iscan

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool iscan_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iscan(add<T>(), x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool iscan_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iscan(mpl::plus<T>(), x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool iscan_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iscan([](T a, T b) { return a + b; }, x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool iscan_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iscan(add<T>(), x)};
  r.wait();
  return x == T((N * N + N) / 2);
}

template<typename T>
bool iscan_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iscan(mpl::plus<T>(), x)};
  r.wait();
  return x == T((N * N + N) / 2);
}

template<typename T>
bool iscan_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iscan([](T a, T b) { return a + b; }, x)};
  r.wait();
  return x == T((N * N + N) / 2);
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(iscan_func_test<double>());
  BOOST_TEST(iscan_op_test<double>());
  BOOST_TEST(iscan_lambda_test<double>());
  BOOST_TEST(iscan_inplace_func_test<double>());
  BOOST_TEST(iscan_inplace_op_test<double>());
  BOOST_TEST(iscan_inplace_lambda_test<double>());
}
