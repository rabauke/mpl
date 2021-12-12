#define BOOST_TEST_MODULE communicator_bcast

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>


template<typename T>
bool bcast_test(const T &val) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  comm_world.bcast(0, x);
  return x == val;
}


template<typename T>
bool bcast_test(const std::vector<T> &send, const std::vector<T> &expected,
                const mpl::layout<T> &l) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  comm_world.bcast(0, x.data(), l);
  if (comm_world.rank() == 0)
    return x == send;
  else
    return x == expected;
}


template<typename T>
bool ibcast_test(const T &val) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  auto r{comm_world.ibcast(0, x)};
  r.wait();
  return x == val;
}


template<typename T>
bool ibcast_test(const std::vector<T> &send, const std::vector<T> &expected,
                 const mpl::layout<T> &l) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  auto r{comm_world.ibcast(0, x.data(), l)};
  r.wait();
  if (comm_world.rank() == 0)
    return x == send;
  else
    return x == expected;
}


BOOST_AUTO_TEST_CASE(bcast) {
  BOOST_TEST(bcast_test(1.0));
  BOOST_TEST(bcast_test(std::array{1, 2, 3, 4}));
  BOOST_TEST(bcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                        mpl::indexed_layout<int>{{{2, 1}, {1, 4}}}));

  BOOST_TEST(ibcast_test(1.0));
  BOOST_TEST(ibcast_test(std::array{1, 2, 3, 4}));
  BOOST_TEST(ibcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                         mpl::indexed_layout<int>{{{2, 1}, {1, 4}}}));
}
