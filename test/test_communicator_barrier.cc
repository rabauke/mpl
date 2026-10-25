#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

bool barrier_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  comm_world.barrier();
  return true;
}


bool ibarrier_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  auto r{comm_world.ibarrier()};
  r.wait();
  return true;
}

}  // namespace


SCENARIO("communicator barrier synchronization", "[communicator][barrier]") {
  GIVEN("a world communicator across all processes") {
    WHEN("synchronizing processes via blocking barrier") {
      THEN("all ranks reach the barrier and proceed") {
        REQUIRE(barrier_test());
      }
    }

    WHEN("synchronizing processes via non-blocking ibarrier") {
      THEN("all ranks wait for the non-blocking barrier request") {
        REQUIRE(ibarrier_test());
      }
    }
  }
}
