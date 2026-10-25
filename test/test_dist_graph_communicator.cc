#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

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

}  // namespace


SCENARIO("distributed graph communicators specify localized process graph neighborhoods",
         "[dist_graph_communicator]") {
  GIVEN("a distributed graph topology with localized source and destination sets") {
    WHEN("querying in-degrees and out-degrees on root and worker ranks") {
      THEN("degrees match the distributed neighborhood definitions") {
        REQUIRE(dist_graph_communicator_test());
      }
    }
  }
}
