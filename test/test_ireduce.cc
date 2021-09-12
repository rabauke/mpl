#define BOOST_TEST_MODULE ireduce

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool allireduce_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  auto r{comm_world.iallreduce(add<T>(), x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allireduce_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  auto r{comm_world.iallreduce(mpl::plus<T>(), x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allireduce_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)}, y{};
  auto r{comm_world.iallreduce([](T a, T b) { return a + b; }, x, y)};
  r.wait();
  return y == T((N * N + N) / 2);
}

template<typename T>
bool allireduce_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.iallreduce(add<T>(), x)};
  r.wait();
  return x == T((N * N + N) / 2);
}

template<typename T>
bool allireduce_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.iallreduce(mpl::plus<T>(), x)};
  r.wait();
  return x == T((N * N + N) / 2);
}

template<typename T>
bool allireduce_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.iallreduce([](T a, T b) { return a + b; }, x)};
  r.wait();
  return x == T((N * N + N) / 2);
}


BOOST_AUTO_TEST_CASE(ireduce) {
  BOOST_TEST(allireduce_func_test<double>());
  BOOST_TEST(allireduce_op_test<double>());
  BOOST_TEST(allireduce_lambda_test<double>());
  BOOST_TEST(allireduce_inplace_func_test<double>());
  BOOST_TEST(allireduce_inplace_op_test<double>());
  BOOST_TEST(allireduce_inplace_lambda_test<double>());
}
