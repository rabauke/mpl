#define BOOST_TEST_MODULE dist_graph_communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool dist_graph_communicator_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int size = comm_world.size();
  int rank = comm_world.rank();
  mpl::dist_graph_communicator::source_set ss;
  mpl::dist_graph_communicator::dest_set ds;
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      ss.insert({i, 0});
      ds.insert({i, 0});
    }
  } else {
    ss.insert({0, 0});
    ds.insert({0, 0});
  }
  mpl::dist_graph_communicator comm_g(comm_world, ss, ds);
  if (rank == 0) {
    if (comm_g.indegree() != comm_g.size() - 1)
      return false;
    if (comm_g.outdegree() != comm_g.size() - 1)
      return false;
  } else {
    if (comm_g.indegree() != 1)
      return false;
    if (comm_g.outdegree() != 1)
      return false;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(dist_graph_communicator) {
  BOOST_TEST(dist_graph_communicator_test());
}
