#define BOOST_TEST_MODULE ireduce

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool ireduce_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    auto r{comm_world.ireduce(add<T>(), 0, x, y)};
    r.wait();
    return y == T((N * N + N) / 2);
  } else {
    auto r{comm_world.ireduce(add<T>(), 0, x)};
    r.wait();
  }
  return true;
}

template<typename T>
bool ireduce_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    auto r{comm_world.ireduce(mpl::plus<T>(), 0, x, y)};
    r.wait();
    return y == T((N * N + N) / 2);
  } else {
    auto r{comm_world.ireduce(mpl::plus<T>(), 0, x)};
    r.wait();
  }
  return true;
}

template<typename T>
bool ireduce_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  if (comm_world.rank() == 0) {
    T y{};
    auto r{comm_world.ireduce([](T a, T b) { return a + b; }, 0, x, y)};
    r.wait();
    return y == T((N * N + N) / 2);
  } else {
    auto r{comm_world.ireduce([](T a, T b) { return a + b; }, 0, x)};
    r.wait();
  }
  return true;
}

template<typename T>
bool ireduce_inplace_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.ireduce(add<T>(), 0, x)};
  r.wait();
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

template<typename T>
bool ireduce_inplace_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.ireduce(mpl::plus<T>(), 0, x)};
  r.wait();
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

template<typename T>
bool ireduce_inplace_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  T x{T(comm_world.rank() + 1)};
  auto r{comm_world.ireduce([](T a, T b) { return a + b; }, 0, x)};
  r.wait();
  return x == T((N * N + N) / 2) or comm_world.rank() > 0;
}

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
  BOOST_TEST(ireduce_func_test<double>());
  BOOST_TEST(ireduce_op_test<double>());
  BOOST_TEST(ireduce_lambda_test<double>());
  BOOST_TEST(ireduce_inplace_func_test<double>());
  BOOST_TEST(ireduce_inplace_op_test<double>());
  BOOST_TEST(ireduce_inplace_lambda_test<double>());
  BOOST_TEST(allireduce_func_test<double>());
  BOOST_TEST(allireduce_op_test<double>());
  BOOST_TEST(allireduce_lambda_test<double>());
  BOOST_TEST(allireduce_inplace_func_test<double>());
  BOOST_TEST(allireduce_inplace_op_test<double>());
  BOOST_TEST(allireduce_inplace_lambda_test<double>());
}
