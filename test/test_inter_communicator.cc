#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("inter-communicator creation and merging", "[inter_communicator]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};

  GIVEN("a world communicator split into odd and even disjoint local groups") {
    const int world_rank{comm_world.rank()};
    const int world_size{comm_world.size()};
    const int my_group{world_rank % 2};
    mpl::communicator local_communicator{mpl::communicator::split, comm_world, my_group};
    const int local_leader{0};
    const int remote_leader{my_group == 0 ? 1 : 0};

    WHEN("creating an inter-communicator across local and remote leaders") {
      mpl::inter_communicator inter_com{local_communicator, local_leader, comm_world,
                                        remote_leader};

      THEN("the sum of local and remote sizes matches world size") {
        REQUIRE(inter_com.size() + inter_com.remote_size() == world_size);
        if (my_group == 0) {
          REQUIRE(inter_com.size() == (world_size + 1) / 2);
          REQUIRE(inter_com.remote_size() == world_size / 2);
        } else {
          REQUIRE(inter_com.remote_size() == (world_size + 1) / 2);
          REQUIRE(inter_com.size() == world_size / 2);
        }
      }

      AND_WHEN("merging the inter-communicator into an intra-communicator") {
        mpl::communicator com{inter_com, mpl::communicator::order_low};
        const auto communicator_equality{com.compare(comm_world)};

        THEN("the merged communicator is congruent or similar to comm_world") {
          REQUIRE((communicator_equality == mpl::communicator::congruent or
                   communicator_equality == mpl::communicator::similar));
        }
      }
    }
  }
}
