#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("communicators can be duplicated asynchronously with info hints",
         "[communicator][info]") {
  GIVEN("the world communicator and an info object") {
    const mpl::communicator &comm{mpl::environment::comm_world()};

    WHEN("the communicator is duplicated asynchronously") {
      const mpl::info hints;
      auto [fst, snd]{comm.idup_with_info(hints)};
      snd.wait();

      THEN("the completed communicator has the same rank and size") {
        REQUIRE(fst.is_valid());
        CHECK(fst.rank() == comm.rank());
        CHECK(fst.size() == comm.size());
      }
    }
  }
}
