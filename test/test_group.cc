#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("groups manage collections of MPI process ranks", "[group]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::communicator &comm_self{mpl::environment::comm_self()};

  GIVEN("process groups derived from comm_world and comm_self") {
    mpl::group group_world{comm_world};
    mpl::group group_self{comm_self};

    THEN("group sizes and ranks match their parent communicators") {
      REQUIRE(group_world.size() == comm_world.size());
      REQUIRE(group_world.rank() == comm_world.rank());
      REQUIRE(group_self.size() == comm_self.size());
    }

    WHEN("group_world is copy constructed") {
      mpl::group group_world_copy{group_world};

      THEN("the copy compares equal to the original") {
        REQUIRE(group_world == group_world_copy);
      }
    }

    WHEN("comparing group_world and group_self") {
      THEN("comparison matches expectation based on communicator size") {
        if (comm_world.size() > 1) {
          REQUIRE(group_world != group_self);
          REQUIRE(group_world.compare(group_self) == mpl::group::unequal);
        } else {
          REQUIRE(group_world == group_self);
          REQUIRE(group_world.compare(group_self) == mpl::group::identical);
        }
      }
    }

    WHEN("translating rank 0 from group_self to group_world") {
      THEN("the translated rank equals group_world.rank()") {
        REQUIRE(group_self.translate(0, group_world) == group_world.rank());
      }
    }

    WHEN("performing set operations (union, intersection, difference, include, exclude)") {
      mpl::group group_union(mpl::group::Union, group_world, group_self);
      mpl::group group_intersection(mpl::group::intersection, group_world, group_self);
      mpl::group group_difference(mpl::group::difference, group_world, group_self);
      mpl::group group_with_0(mpl::group::include, group_world, {0});
      mpl::group group_without_0(mpl::group::exclude, group_world, {0});

      THEN("resulting group sizes reflect the set algebra") {
        REQUIRE(group_union.size() == group_world.size());
        REQUIRE(group_intersection.size() == 1);
        REQUIRE(group_difference.size() == group_world.size() - 1);
        REQUIRE(group_with_0.size() == 1);
        REQUIRE(group_without_0.size() == group_world.size() - 1);
      }
    }
  }
}
