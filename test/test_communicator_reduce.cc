#define BOOST_TEST_MODULE communicator_reduce

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"

template<typename T>
class add {
public:
  T operator()(const T &a, const T &b) const { return a + b; }
};


template<typename F, typename T>
bool reduce_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  if (comm_world.rank() == 0) {
    T y{};
    comm_world.reduce(f, 0, x, y);
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    return y == expected;
  } else {
    comm_world.reduce(f, 0, x);
    return true;
  }
}


template<typename F, typename T>
bool reduce_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  if (comm_world.rank() == 0) {
    std::vector<T> v_y(n);
    comm_world.reduce(f, 0, v_x.data(), v_y.data(), l);
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    std::vector<T> v_expected(n, expected);
    return v_y == v_expected;
  } else {
    comm_world.reduce(f, 0, v_x.data(), l);
    return true;
  }
}


template<typename F, typename T>
bool ireduce_test(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  if (comm_world.rank() == 0) {
    T y{};
    auto r{comm_world.ireduce(f, 0, x, y)};
    r.wait();
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    return y == expected;
  } else {
    auto r{comm_world.ireduce(f, 0, x)};
    r.wait();
    return true;
  }
}


template<typename F, typename T>
bool ireduce_test_with_layout(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  if (comm_world.rank() == 0) {
    std::vector<T> v_y(n);
    auto r{comm_world.ireduce(f, 0, v_x.data(), v_y.data(), l)};
    r.wait();
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    std::vector<T> v_expected(n, expected);
    return v_y == v_expected;
  } else {
    auto r{comm_world.ireduce(f, 0, v_x.data(), l)};
    r.wait();
    return true;
  }
}


template<typename F, typename T>
bool reduce_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  if (comm_world.rank() == 0) {
    T x2{x};
    comm_world.reduce(f, 0, x);
    T expected{x2};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x2;
      expected = f(expected, x2);
    }
    return x == expected;
  } else {
    comm_world.reduce(f, 0, x);
    return true;
  }
}


template<typename F, typename T>
bool reduce_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  if (comm_world.rank() == 0) {
    comm_world.reduce(f, 0, v_x.data(), l);
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    std::vector<T> v_expected(n, expected);
    return v_x == v_expected;
  } else {
    const std::vector<T> &const_v_x{v_x};
    comm_world.reduce(f, 0, const_v_x.data(), l);
    return true;
  }
}


template<typename F, typename T>
bool ireduce_test_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  if (comm_world.rank() == 0) {
    T x2{x};
    auto r{comm_world.ireduce(f, 0, x)};
    r.wait();
    T expected{x2};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x2;
      expected = f(expected, x2);
    }
    return x == expected;
  } else {
    auto r{comm_world.ireduce(f, 0, x)};
    r.wait();
    return true;
  }
}


template<typename F, typename T>
bool ireduce_test_with_layout_inplace(F f, const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  const int n{5};
  mpl::contiguous_layout<T> l(n);
  std::vector<T> v_x(n, x);
  if (comm_world.rank() == 0) {
    auto r{comm_world.ireduce(f, 0, v_x.data(), l)};
    r.wait();
    T expected{x};
    for (int i{1}; i < comm_world.size(); ++i) {
      ++x;
      expected = f(expected, x);
    }
    std::vector<T> v_expected(n, expected);
    return v_x == v_expected;
  } else {
    const std::vector<T> &const_v_x{v_x};
    auto r{comm_world.ireduce(f, 0, const_v_x.data(), l)};
    r.wait();
    return true;
  }
}


BOOST_AUTO_TEST_CASE(reduce) {
  BOOST_TEST(reduce_test(add<double>(), 1.0));
  BOOST_TEST(reduce_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test(mpl::plus<double>(), 1.0));
  BOOST_TEST(reduce_test(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(reduce_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(reduce_test_with_layout(add<double>(), 1.0));
  BOOST_TEST(reduce_test_with_layout(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_with_layout(mpl::plus<double>(), 1.0));
  BOOST_TEST(reduce_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(reduce_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(ireduce_test(add<double>(), 1.0));
  BOOST_TEST(ireduce_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test(mpl::plus<double>(), 1.0));
  BOOST_TEST(ireduce_test(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(ireduce_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(ireduce_test_with_layout(add<double>(), 1.0));
  BOOST_TEST(ireduce_test_with_layout(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_with_layout(mpl::plus<double>(), 1.0));
  BOOST_TEST(ireduce_test_with_layout(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_with_layout([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(ireduce_test_with_layout([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));


  BOOST_TEST(reduce_test_inplace(add<double>(), 1.0));
  BOOST_TEST(reduce_test_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(reduce_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(reduce_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(reduce_test_with_layout_inplace(add<double>(), 1.0));
  BOOST_TEST(reduce_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_with_layout_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(reduce_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(reduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(ireduce_test_inplace(add<double>(), 1.0));
  BOOST_TEST(ireduce_test_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(ireduce_test_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(ireduce_test_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(ireduce_test_with_layout_inplace(add<double>(), 1.0));
  BOOST_TEST(ireduce_test_with_layout_inplace(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_with_layout_inplace(mpl::plus<double>(), 1.0));
  BOOST_TEST(ireduce_test_with_layout_inplace(mpl::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(ireduce_test_with_layout_inplace([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
}
