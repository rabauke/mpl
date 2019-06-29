#define BOOST_TEST_MODULE collective

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool barrier_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  comm_world.barrier();
  return true;
}

template<typename T>
bool bcast_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x;
  if (comm_world.rank() == 0)
    x = T(1);
  comm_world.bcast(0, x);
  return x == T(1);
}

template<typename T>
bool scatter_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  std::iota(begin(v), end(v), 0);
  T x;
  if (comm_world.rank() == 0) {
    comm_world.scatter(0, v.data(), x);
  } else {
    comm_world.scatter(0, x);
  }
  return x == v[comm_world.rank()];
}

template<typename T>
bool gather_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  T x = comm_world.rank();
  if (comm_world.rank() == 0) {
    comm_world.gather(0, x, v.data());
    for (int i = 0; i < comm_world.size(); ++i)
      if (v[i] != i)
        return false;
  } else {
    comm_world.gather(0, x);
  }
  return true;
}

template<typename T>
bool allgather_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  T x = comm_world.rank();
  comm_world.allgather(x, v.data());
  for (int i = 0; i < comm_world.size(); ++i)
    if (v[i] != i)
      return false;
  return true;
}

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
  BOOST_TEST(barrier_test());
  BOOST_TEST(bcast_test<double>());
  BOOST_TEST(scatter_test<double>());
  BOOST_TEST(gather_test<double>());
  BOOST_TEST(allgather_test<double>());
  BOOST_TEST(alltoall_test<double>());
}
