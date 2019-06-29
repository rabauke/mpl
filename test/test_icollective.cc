#define BOOST_TEST_MODULE icollective

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool ibarrier_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  auto r{comm_world.ibarrier()};
  r.wait();
  return true;
}

template<typename T>
bool ibcast_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x;
  if (comm_world.rank() == 0)
    x = T(1);
  auto r{comm_world.ibcast(0, x)};
  r.wait();
  return x == T(1);
}

template<typename T>
bool iscatter_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  std::iota(begin(v), end(v), 0);
  T x;
  if (comm_world.rank() == 0) {
    auto r{comm_world.iscatter(0, v.data(), x)};
    r.wait();
  } else {
    auto r{comm_world.iscatter(0, x)};
    r.wait();
  }
  return x == v[comm_world.rank()];
}

template<typename T>
bool igather_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  T x = comm_world.rank();
  if (comm_world.rank() == 0) {
    auto r{comm_world.igather(0, x, v.data())};
    r.wait();
    for (int i = 0; i < comm_world.size(); ++i)
      if (v[i] != i)
        return false;
  } else {
    auto r{comm_world.igather(0, x)};
    r.wait();
  }
  return true;
}

template<typename T>
bool iallgather_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v(comm_world.size());
  T x = comm_world.rank();
  auto r{comm_world.iallgather(x, v.data())};
  r.wait();
  for (int i = 0; i < comm_world.size(); ++i)
    if (v[i] != i)
      return false;
  return true;
}

template<typename T>
bool ialltoall_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<std::pair<T, T>> v(comm_world.size());
  for (int i = 0; i < comm_world.size(); ++i)
    v[i] = std::make_pair(static_cast<T>(i), static_cast<T>(comm_world.rank()));
  auto r{comm_world.ialltoall(v.data())};
  r.wait();
  for (int i = 0; i < comm_world.size(); ++i)
    if (v[i] != std::make_pair(static_cast<T>(comm_world.rank()), static_cast<T>(i)))
      return false;
  return true;
}


BOOST_AUTO_TEST_CASE(icollective) {
  BOOST_TEST(ibarrier_test());
  BOOST_TEST(ibcast_test<double>());
  BOOST_TEST(iscatter_test<double>());
  BOOST_TEST(igather_test<double>());
  BOOST_TEST(iallgather_test<double>());
  BOOST_TEST(ialltoall_test<double>());
}
