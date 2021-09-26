#define BOOST_TEST_MODULE cartesian_communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool cartesian_communicator_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic,
                                                     mpl::cartesian_communicator::non_periodic};
  mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  if (comm_c.dimensionality() != 2)
    return false;
  const int rank{comm_c.rank()};
  const int size{comm_c.size()};
  auto coordinate{comm_c.coordinate()};
  if (comm_c.rank(coordinate) != rank)
    return false;
  auto dims{comm_c.get_dimensions()};
  if (dims.size(0) * dims.size(1) != comm_c.size())
    return false;
  if (not(dims.periodicity(0) == mpl::cartesian_communicator::periodic and
          dims.periodicity(1) == mpl::cartesian_communicator::non_periodic))
    return false;
  auto ranks{comm_c.shift(0, 1)};
  ++coordinate[0];
  if (coordinate[0] >= dims.size(0))
    coordinate[0] = 0;
  int destination_1{comm_c.rank(coordinate)};
  coordinate[0] -= 2;
  if (coordinate[0] < 0)
    coordinate[0] += dims.size(0);
  int source_1{comm_c.rank(coordinate)};
  if (not(ranks.source == source_1 and ranks.destination == destination_1))
    return false;
  {
    double x{1};
    std::vector<double> y(4, 0.);
    comm_c.neighbor_allgather(x, y.data());
    if ((y[0] != 0 and y[0] != 1) or (y[1] != 0 and y[1] != 1) or (y[2] != 0 and y[2] != 1) or
        (y[3] != 0 and y[3] != 1))
      return false;
  }
  {
    std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    comm_c.neighbor_alltoall(x.data(), y.data());
    auto ranks_0{comm_c.shift(0, 1)};
    auto ranks_1{comm_c.shift(1, 1)};
    if (ranks_0.source != mpl::proc_null and y[0] != ranks_0.source + 1.)
      return false;
    if (ranks_0.destination != mpl::proc_null and y[1] != ranks_0.destination + 1.)
      return false;
    if (ranks_1.source != mpl::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mpl::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  {
    std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    mpl::layouts<double> ls;
    ls.push_back(mpl::indexed_layout<double>({{1, 0}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 1}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 2}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 3}}));
    comm_c.neighbor_alltoallv(x.data(), ls, y.data(), ls);
    auto ranks_0{comm_c.shift(0, 1)};
    auto ranks_1{comm_c.shift(1, 1)};
    if (ranks_0.source != mpl::proc_null and y[0] != ranks_0.source + 1.)
      return false;
    if (ranks_0.destination != mpl::proc_null and y[1] != ranks_0.destination + 1.)
      return false;
    if (ranks_1.source != mpl::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mpl::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  return true;
}


BOOST_AUTO_TEST_CASE(cartesian_communicator) {
  BOOST_TEST(cartesian_communicator_test());
}


BOOST_AUTO_TEST_CASE(cartesian_communicator_dimensions) {
  mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic,
                                                     mpl::cartesian_communicator::non_periodic,
                                                     mpl::cartesian_communicator::non_periodic};

  BOOST_TEST(dimensions.dimensionality() == 3);
  BOOST_TEST(dimensions.periodicity(0) == mpl::cartesian_communicator::periodic);
  BOOST_TEST(dimensions.periodicity(1) == mpl::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.periodicity(2) == mpl::cartesian_communicator::non_periodic);
  dimensions[1] = {10, mpl::cartesian_communicator::periodic};
  BOOST_TEST(dimensions.periodicity(1) == mpl::cartesian_communicator::periodic);
  BOOST_TEST(dimensions.size(1) == 10);
  dimensions.add(11, mpl::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.periodicity(3) == mpl::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.size(3) == 11);
}
