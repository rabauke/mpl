#define BOOST_TEST_MODULE reduce_scatter

#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <mpl/mpl.hpp>

template<typename T>
T add(const T &a, const T &b) {
  return a+b;
}

template<typename T>
bool reduce_scatter_block_func_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  int N=comm_world.size();
  std::vector<T> x(N, comm_world.rank()+1);
  T y=-1;
  comm_world.reduce_scatter_block(add<T>, x.data(), y);
  return (N*N+N)/2==y;
}

template<typename T>
bool reduce_scatter_block_op_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  int N=comm_world.size();
  std::vector<T> x(N, comm_world.rank()+1);
  T y=-1;
  comm_world.reduce_scatter_block(mpl::plus<T>(), x.data(), y);
  return (N*N+N)/2==y;
}

template<typename T>
bool reduce_scatter_block_lambda_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  int N=comm_world.size();
  std::vector<T> x(N, comm_world.rank()+1);
  T y=-1;
  comm_world.reduce_scatter_block([](T a, T b) { return a+b; }, x.data(), y);
  return (N*N+N)/2==y;
}


BOOST_AUTO_TEST_CASE(reduce_scatter) {
  BOOST_TEST(reduce_scatter_block_func_test<double>());
  BOOST_TEST(reduce_scatter_block_op_test<double>());
  BOOST_TEST(reduce_scatter_block_lambda_test<double>());
}
