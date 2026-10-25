#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("communicator management and properties", "[communicator]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::communicator &comm_self{mpl::environment::comm_self()};

  GIVEN("the predefined comm_world communicator") {
    THEN("it is valid, has positive size, and ranks are within range") {
      REQUIRE(comm_world.is_valid());
      const int size{comm_world.size()};
      const int rank{comm_world.rank()};
      REQUIRE(size >= 1);
      REQUIRE(rank >= 0);
      REQUIRE(rank < size);

      const mpl::communicator &comm_world_2{mpl::environment::comm_world()};
      REQUIRE(comm_world_2 == comm_world);
      REQUIRE(comm_self != comm_world);

      if (size == 1) {
        REQUIRE(comm_self.compare(comm_world) == mpl::communicator::congruent);
      } else {
        REQUIRE(comm_self.compare(comm_world) == mpl::communicator::unequal);
      }
    }

    WHEN("comm_world is copy constructed") {
      mpl::communicator comm_new{comm_world};

      THEN("the copy is valid, has matching size, and is congruent") {
        REQUIRE(comm_new.is_valid());
        REQUIRE(comm_new.size() == comm_world.size());
        REQUIRE(comm_new.compare(comm_world) == mpl::communicator::congruent);
      }
    }

    WHEN("splitting comm_world into odd and even color groups") {
      const int size{comm_world.size()};
      const int rank{comm_world.rank()};
      mpl::communicator comm_new{mpl::communicator::split, comm_world, rank % 2 == 0};

      THEN("the split communicators are valid and sized correctly") {
        REQUIRE(comm_new.is_valid());
        const int size_new{comm_new.size()};
        if (size % 2 == 0) {
          REQUIRE(size_new == size / 2);
        } else {
          if (rank % 2 == 0) {
            REQUIRE(size_new == (size + 1) / 2);
          } else {
            REQUIRE(size_new == (size - 1) / 2);
          }
        }
      }
    }

    WHEN("splitting comm_world by shared memory") {
      const int rank{comm_world.rank()};
      mpl::communicator comm_new{mpl::communicator::split_shared_memory, comm_world,
                                 rank % 2 == 0};

      THEN("the shared memory communicator is valid") {
        REQUIRE(comm_new.is_valid());
      }
    }
  }

  GIVEN("the predefined comm_self communicator") {
    THEN("it is valid, has size 1 and rank 0") {
      REQUIRE(comm_self.is_valid());
      REQUIRE(comm_self.size() == 1);
      REQUIRE(comm_self.rank() == 0);

      const mpl::communicator &comm_self_2{mpl::environment::comm_self()};
      REQUIRE(comm_self_2 == comm_self);
      REQUIRE(comm_self != comm_world);

      if (comm_world.size() == 1) {
        REQUIRE(comm_self.compare(comm_world) == mpl::communicator::congruent);
      } else {
        REQUIRE(comm_self.compare(comm_world) == mpl::communicator::unequal);
      }
    }
  }
}
