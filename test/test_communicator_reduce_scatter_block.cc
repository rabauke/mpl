#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

template<typename F, typename T>
bool reduce_scatter_block_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  for (int i{0}; i < comm_world.size(); ++i) {
    v_x.push_back(x);
    ++x;
  }
  T y{};
  comm_world.reduce_scatter_block(f, v_x.data(), y);
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  return y == expected;
}


template<typename F, typename T>
bool reduce_scatter_block_test_with_layout(F f, const T &val) {
  const int block_size{3};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  for (int i{0}; i < comm_world.size(); ++i) {
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    ++x;
  }
  std::vector<T> v_y(block_size);
  mpl::contiguous_layout<T> l(block_size);
  comm_world.reduce_scatter_block(f, v_x.data(), v_y.data(), l);
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
bool ireduce_scatter_block_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  for (int i{0}; i < comm_world.size(); ++i) {
    v_x.push_back(x);
    ++x;
  }
  T y{};
  auto r{comm_world.ireduce_scatter_block(f, v_x.data(), y)};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  r.wait();
  return y == expected;
}


template<typename F, typename T>
bool ireduce_scatter_block_test_with_layout(F f, const T &val) {
  const int block_size{3};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  std::vector<T> v_x;
  for (int i{0}; i < comm_world.size(); ++i) {
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    ++x;
  }
  std::vector<T> v_y(block_size);
  mpl::contiguous_layout<T> l(block_size);
  auto r{comm_world.ireduce_scatter_block(f, v_x.data(), v_y.data(), l)};
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


SCENARIO("reduce_scatter_block reduces and scatters equal-sized blocks across processes",
         "[communicator][reduce_scatter_block]") {
  GIVEN("a world communicator combining and partitioning uniform blocks") {
    WHEN("performing synchronous reduce_scatter_block with scalars and vector layouts") {
      THEN("each process receives its correctly reduced equal block") {
        REQUIRE(reduce_scatter_block_test(add<double>(), 1.0));
        REQUIRE(reduce_scatter_block_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_block_test(mpl::plus<double>(), 1.0));
        REQUIRE(reduce_scatter_block_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_block_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(reduce_scatter_block_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(reduce_scatter_block_test_with_layout(add<double>(), 1.0));
        REQUIRE(reduce_scatter_block_test_with_layout(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_block_test_with_layout(mpl::plus<double>(), 1.0));
        REQUIRE(reduce_scatter_block_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(reduce_scatter_block_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(reduce_scatter_block_test_with_layout([](auto a, auto b) { return a + b; },
                                                     tuple{1, 2.0}));
      }
    }

    WHEN("performing asynchronous ireduce_scatter_block with scalars and vector layouts") {
      THEN("each process receives its reduced equal block upon completion") {
        REQUIRE(ireduce_scatter_block_test(add<double>(), 1.0));
        REQUIRE(ireduce_scatter_block_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(ireduce_scatter_block_test(mpl::plus<double>(), 1.0));
        REQUIRE(ireduce_scatter_block_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(ireduce_scatter_block_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(ireduce_scatter_block_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(ireduce_scatter_block_test_with_layout(add<double>(), 1.0));
        REQUIRE(ireduce_scatter_block_test_with_layout(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(ireduce_scatter_block_test_with_layout(mpl::plus<double>(), 1.0));
        REQUIRE(ireduce_scatter_block_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(
            ireduce_scatter_block_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(ireduce_scatter_block_test_with_layout([](auto a, auto b) { return a + b; },
                                                      tuple{1, 2.0}));
      }
    }
  }
}
