#define BOOST_TEST_MODULE communicator_scan

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"

template<typename F, typename T>
bool scan_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T y{};
  comm_world.scan(f, x, y);
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  return y == expected;
}


template<typename F, typename T>
bool scan_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  std::vector<T> v_y(n);
  comm_world.scan(f, v_x.data(), v_y.data(), l);
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_y == v_expected;
}


template<typename F, typename T>
bool iscan_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T y{};
  auto r{comm_world.iscan(f, x, y)};
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  r.wait();
  return y == expected;
}


template<typename F, typename T>
bool iscan_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  std::vector<T> v_y(n);
  auto r{comm_world.iscan(f, v_x.data(), v_y.data(), l)};
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  r.wait();
  return v_y == v_expected;
}


template<typename F, typename T>
bool scan_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T x2{val};
  comm_world.scan(f, x);
  T expected{x2};
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x2;
    expected = f(expected, x2);
  }
  return x == expected;
}


template<typename F, typename T>
bool scan_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  comm_world.scan(f, v_x.data(), l);
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  return v_x == v_expected;
}


template<typename F, typename T>
bool iscan_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T x2{val};
  auto r{comm_world.iscan(f, x)};
  T expected{x2};
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x2;
    expected = f(expected, x2);
  }
  r.wait();
  return x == expected;
}


template<typename F, typename T>
bool iscan_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  auto r{comm_world.iscan(f, v_x.data(), l)};
  T expected{val};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i) {
    ++x;
    expected = f(expected, x);
  }
  std::vector<T> v_expected(n, expected);
  r.wait();
  return v_x == v_expected;
}


BOOST_AUTO_TEST_CASE(scan) {
  BOOST_TEST(scan_test(add<double>(), 1.0));
  BOOST_TEST(scan_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test(mpl::plus<double>(), 1.0));
  BOOST_TEST(scan_test(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(scan_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(scan_test_with_layout(add<double>(), 1.0));
  BOOST_TEST(scan_test_with_layout(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_with_layout(mpl::plus<double>(), 1.0));
  BOOST_TEST(scan_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(scan_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(iscan_test(add<double>(), 1.0));
  BOOST_TEST(iscan_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test(mpl::plus<double>(), 1.0));
  BOOST_TEST(iscan_test(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(iscan_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(iscan_test_with_layout(add<double>(), 1.0));
  BOOST_TEST(iscan_test_with_layout(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_with_layout(mpl::plus<double>(), 1.0));
  BOOST_TEST(iscan_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(iscan_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));


  BOOST_TEST(scan_test_inplace(add<double>(), 1.0));
  BOOST_TEST(scan_test_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(scan_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(scan_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(scan_test_with_layout_inplace(add<double>(), 1.0));
  BOOST_TEST(scan_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_with_layout_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(scan_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(scan_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(
      scan_test_with_layout_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(iscan_test_inplace(add<double>(), 1.0));
  BOOST_TEST(iscan_test_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(iscan_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(iscan_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(iscan_test_with_layout_inplace(add<double>(), 1.0));
  BOOST_TEST(iscan_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_with_layout_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(iscan_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(iscan_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(
      iscan_test_with_layout_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
}
