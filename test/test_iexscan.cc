#define BOOST_TEST_MODULE iexscan

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool iexscan_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iexscan(add<T>(), x, y)};
  r.wait();
  return comm_world.rank() == 0 ? true : y == T((N * N - N) / 2);
}

template<typename T>
bool iexscan_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iexscan(mpl::plus<T>(), x, y)};
  r.wait();
  return comm_world.rank() == 0 ? true : y == T((N * N - N) / 2);
}

template<typename T>
bool iexscan_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)}, y{};
  auto r{comm_world.iexscan([](T a, T b) { return a + b; }, x, y)};
  r.wait();
  return comm_world.rank() == 0 ? true : y == T((N * N - N) / 2);
}

template<typename T>
bool iexscan_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iexscan(add<T>(), x)};
  r.wait();
  return comm_world.rank() == 0 ? true : x == T((N * N - N) / 2);
}

template<typename T>
bool iexscan_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iexscan(mpl::plus<T>(), x)};
  r.wait();
  return comm_world.rank() == 0 ? true : x == T((N * N - N) / 2);
}

template<typename T>
bool iexscan_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.rank() + 1;
  T x{T(N)};
  auto r{comm_world.iexscan([](T a, T b) { return a + b; }, x)};
  r.wait();
  return comm_world.rank() == 0 ? true : x == T((N * N - N) / 2);
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(iexscan_func_test<double>());
  BOOST_TEST(iexscan_op_test<double>());
  BOOST_TEST(iexscan_lambda_test<double>());
  BOOST_TEST(iexscan_inplace_func_test<double>());
  BOOST_TEST(iexscan_inplace_op_test<double>());
  BOOST_TEST(iexscan_inplace_lambda_test<double>());
}
