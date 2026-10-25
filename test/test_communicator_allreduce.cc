#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

template<typename F, typename T>
bool allreduce_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T y{};
  comm_world.allreduce(f, x, y);
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  return y == expected;
}


template<typename F, typename T>
bool allreduce_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  std::vector<T> v_y(n);
  comm_world.allreduce(f, v_x.data(), v_y.data(), l);
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_y == v_expected;
}


template<typename F, typename T>
bool iallreduce_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T y{};
  auto r{comm_world.iallreduce(f, x, y)};
  r.wait();
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  return y == expected;
}


template<typename F, typename T>
bool iallreduce_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  std::vector<T> v_y(n);
  auto r{comm_world.iallreduce(f, v_x.data(), v_y.data(), l)};
  r.wait();
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_y == v_expected;
}


template<typename F, typename T>
bool allreduce_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  comm_world.allreduce(f, x);
  T expected{val};
  T x2{val};
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x2;
    expected = f(expected, x2);
  }
  return x == expected;
}


template<typename F, typename T>
bool allreduce_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  comm_world.allreduce(f, v_x.data(), l);
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_x == v_expected;
}


template<typename F, typename T>
bool iallreduce_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T x2{val};
  auto r{comm_world.iallreduce(f, x)};
  r.wait();
  T expected{x2};
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x2;
    expected = f(expected, x2);
  }
  return x == expected;
}


template<typename F, typename T>
bool iallreduce_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  auto r{comm_world.iallreduce(f, v_x.data(), l)};
  r.wait();
  T expected{val};
  x = val;
  for (int i{1}; i < comm_world.size(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_x == v_expected;
}

}  // namespace


SCENARIO("allreduce collective reduction across all processes",
         "[communicator][allreduce]") {
  GIVEN("a world communicator combining values globally across all processes") {
    WHEN("performing allreduce synchronously with custom and built-in operators") {
      THEN("all processes compute and receive the reduced result") {
        REQUIRE(allreduce_test(add<double>(), 1.0));
        REQUIRE(allreduce_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test(mpl::plus<double>(), 1.0));
        REQUIRE(allreduce_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(allreduce_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(allreduce_test_with_layout(add<double>(), 1.0));
        REQUIRE(allreduce_test_with_layout(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_with_layout(mpl::plus<double>(), 1.0));
        REQUIRE(allreduce_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(allreduce_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
      }
    }

    WHEN("performing allreduce asynchronously using iallreduce") {
      THEN("all processes compute and receive the reduced result upon completion") {
        REQUIRE(iallreduce_test(add<double>(), 1.0));
        REQUIRE(iallreduce_test(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test(mpl::plus<double>(), 1.0));
        REQUIRE(iallreduce_test(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(iallreduce_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(iallreduce_test_with_layout(add<double>(), 1.0));
        REQUIRE(iallreduce_test_with_layout(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_with_layout(mpl::plus<double>(), 1.0));
        REQUIRE(iallreduce_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(iallreduce_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
      }
    }

    WHEN("performing in-place allreduce and iallreduce operations") {
      THEN("all processes compute in-place reductions properly") {
        REQUIRE(allreduce_test_inplace(add<double>(), 1.0));
        REQUIRE(allreduce_test_inplace(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_inplace(mpl::plus<double>(), 1.0));
        REQUIRE(allreduce_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(allreduce_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(allreduce_test_with_layout_inplace(add<double>(), 1.0));
        REQUIRE(allreduce_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_with_layout_inplace(mpl::plus<double>(), 1.0));
        REQUIRE(allreduce_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(allreduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(allreduce_test_with_layout_inplace([](auto a, auto b) { return a + b; },
                                                  tuple{1, 2.0}));

        REQUIRE(iallreduce_test_inplace(add<double>(), 1.0));
        REQUIRE(iallreduce_test_inplace(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_inplace(mpl::plus<double>(), 1.0));
        REQUIRE(iallreduce_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(iallreduce_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

        REQUIRE(iallreduce_test_with_layout_inplace(add<double>(), 1.0));
        REQUIRE(iallreduce_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_with_layout_inplace(mpl::plus<double>(), 1.0));
        REQUIRE(iallreduce_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
        REQUIRE(iallreduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
        REQUIRE(iallreduce_test_with_layout_inplace([](auto a, auto b) { return a + b; },
                                                   tuple{1, 2.0}));
      }
    }
  }
}
