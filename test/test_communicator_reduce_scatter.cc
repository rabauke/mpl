#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

template<typename F, typename T>
bool reduce_scatter_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  mpl::contiguous_layouts<T> l;
  for (int i{0}; i < comm_world.size(); ++i) {
    const int block_size{i + 1};
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    l.push_back(mpl::contiguous_layout<T>(block_size));
    ++x;
  }
  const int block_size{comm_world.rank() + 1};
  std::vector<T> v_y(block_size);
  comm_world.reduce_scatter(f, v_x.data(), v_y.data(), l);
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  std::vector<T> v_expected(block_size, expected);
  return v_y == v_expected;
}


template<typename F, typename T>
bool ireduce_scatter_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  mpl::contiguous_layouts<T> l;
  for (int i{0}; i < comm_world.size(); ++i) {
    const int block_size{i + 1};
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    l.push_back(mpl::contiguous_layout<T>(block_size));
    ++x;
  }
  const int block_size{comm_world.rank() + 1};
  std::vector<T> v_y(block_size);
  auto r{comm_world.ireduce_scatter(f, v_x.data(), v_y.data(), l)};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  std::vector<T> v_expected(block_size, expected);
  r.wait();
  return v_y == v_expected;
}

}  // namespace


SCENARIO("reduce_scatter reduces and scatters variable blocks across processes",
         "[communicator][reduce_scatter]") {
  GIVEN("a world communicator combining and partitioning blocks of data") {
    WHEN("performing synchronous reduce_scatter with functors, operators, and lambdas") {
      THEN("each process receives its correctly reduced variable slice") {
        REQUIRE(reduce_scatter_test(add<double>(), 1.0));
        REQUIRE(reduce_scatter_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_test(mpl::plus<double>(), 1.0));
        REQUIRE(reduce_scatter_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(reduce_scatter_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
      }
    }

    WHEN("performing asynchronous ireduce_scatter with functors, operators, and lambdas") {
      THEN("each process receives its reduced variable slice upon completion") {
        REQUIRE(ireduce_scatter_test(add<double>(), 1.0));
        REQUIRE(ireduce_scatter_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(ireduce_scatter_test(mpl::plus<double>(), 1.0));
        REQUIRE(ireduce_scatter_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(ireduce_scatter_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(ireduce_scatter_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
      }
    }
  }
}
