#define BOOST_TEST_MODULE communicator_allgather

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <algorithm>
#include <array>
#include <vector>


template<typename T>
bool allgather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> v(comm_world.size());
  comm_world.allgather(val, v.data());
  return std::all_of(v.begin(), v.end(), [&val](const auto &x) { return x == val; });
}


template<typename T>
bool iallgather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> v(comm_world.size());
  auto r{comm_world.iallgather(val, v.data())};
  r.wait();
  return std::all_of(v.begin(), v.end(), [&val](const auto &x) { return x == val; });
}


BOOST_AUTO_TEST_CASE(allgather) {
  BOOST_TEST(allgather_test(1.0));
  BOOST_TEST(allgather_test(std::array{1, 2, 3, 4}));

  BOOST_TEST(iallgather_test(1.0));
  BOOST_TEST(iallgather_test(std::array{1, 2, 3, 4}));
}
