#define BOOST_TEST_MODULE ireduce_scatter

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
bool ireduce_scatter_block_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  std::vector<T> x(N, comm_world.rank() + 1);
  T y{-1};
  auto r{comm_world.ireduce_scatter_block(add<T>(), x.data(), y)};
  r.wait();
  return (N * N + N) / 2 == y;
}

template<typename T>
bool ireduce_scatter_block_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  std::vector<T> x(N, comm_world.rank() + 1);
  T y{-1};
  auto r{comm_world.ireduce_scatter_block(mpl::plus<T>(), x.data(), y)};
  r.wait();
  return (N * N + N) / 2 == y;
}

template<typename T>
bool ireduce_scatter_block_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  std::vector<T> x(N, comm_world.rank() + 1);
  T y{-1};
  auto r{comm_world.ireduce_scatter_block([](T a, T b) { return a + b; }, x.data(), y)};
  r.wait();
  return (N * N + N) / 2 == y;
}

template<typename T>
bool ireduce_scatter_func_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  mpl::contiguous_layouts<T> l;
  for (int i = 1; i <= N; ++i)
    l.push_back(mpl::contiguous_layout<T>(i));
  std::vector<T> x;
  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= i; ++j)
      x.push_back(T(j));
  std::vector<T> y(comm_world.rank() + 1);
  auto r{comm_world.ireduce_scatter(add<T>(), x.data(), y.data(), l)};
  r.wait();
  for (int i = 0; i <= comm_world.rank(); ++i)
    if (y[i] != N * (i + 1))
      return false;
  return true;
}

template<typename T>
bool ireduce_scatter_op_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  mpl::contiguous_layouts<T> l;
  for (int i = 1; i <= N; ++i)
    l.push_back(mpl::contiguous_layout<T>(i));
  std::vector<T> x;
  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= i; ++j)
      x.push_back(T(j));
  std::vector<T> y(comm_world.rank() + 1);
  auto r{comm_world.ireduce_scatter(mpl::plus<T>(), x.data(), y.data(), l)};
  r.wait();
  for (int i = 0; i <= comm_world.rank(); ++i)
    if (y[i] != N * (i + 1))
      return false;
  return true;
}

template<typename T>
bool ireduce_scatter_lambda_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int N = comm_world.size();
  mpl::contiguous_layouts<T> l;
  for (int i = 1; i <= N; ++i)
    l.push_back(mpl::contiguous_layout<T>(i));
  std::vector<T> x;
  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= i; ++j)
      x.push_back(T(j));
  std::vector<T> y(comm_world.rank() + 1);
  auto r{comm_world.ireduce_scatter([](T a, T b) { return a + b; }, x.data(), y.data(), l)};
  r.wait();
  for (int i = 0; i <= comm_world.rank(); ++i)
    if (y[i] != N * (i + 1))
      return false;
  return true;
}

BOOST_AUTO_TEST_CASE(ireduce_scatter) {
  BOOST_TEST(ireduce_scatter_block_func_test<double>());
  BOOST_TEST(ireduce_scatter_block_op_test<double>());
  BOOST_TEST(ireduce_scatter_block_lambda_test<double>());
  BOOST_TEST(ireduce_scatter_func_test<double>());
  BOOST_TEST(ireduce_scatter_op_test<double>());
  BOOST_TEST(ireduce_scatter_lambda_test<double>());
}
