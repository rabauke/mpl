#define BOOST_TEST_MODULE dist_graph_communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>


bool dist_graph_communicator_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  mpl::distributed_graph_communicator::neighbours_set sources;
  mpl::distributed_graph_communicator::neighbours_set destination;
  if (rank == 0) {
    for (int i{1}; i < size; ++i) {
      sources.add(i);
      destination.add({i, 0});
    }
  } else {
    sources.add(0);
    destination.add({0, 0});
  }
  mpl::distributed_graph_communicator comm_g(comm_world, sources, destination);
  if (rank == 0) {
    if (comm_g.in_degree() != comm_g.size() - 1)
      return false;
    if (comm_g.out_degree() != comm_g.size() - 1)
      return false;
  } else {
    if (comm_g.in_degree() != 1)
      return false;
    if (comm_g.out_degree() != 1)
      return false;
  }
  return true;
}


BOOST_AUTO_TEST_CASE(dist_graph_communicator) {
  BOOST_TEST(dist_graph_communicator_test());
}
