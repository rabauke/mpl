#define BOOST_TEST_MODULE collective

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
bool alltoall_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<std::pair<T, T>> v(comm_world.size());
  for (int i = 0; i < comm_world.size(); ++i)
    v[i] = std::make_pair(static_cast<T>(i), static_cast<T>(comm_world.rank()));
  comm_world.alltoall(v.data());
  for (int i = 0; i < comm_world.size(); ++i)
    if (v[i] != std::make_pair(static_cast<T>(comm_world.rank()), static_cast<T>(i)))
      return false;
  return true;
}


BOOST_AUTO_TEST_CASE(collective) {
  BOOST_TEST(alltoall_test<double>());
}
