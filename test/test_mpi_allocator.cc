#include <catch2/catch_test_macros.hpp>
#include <new>
#include <vector>
#include <mpl/mpl.hpp>


SCENARIO("MPI allocator manages STL container storage", "[allocator]") {
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};

  GIVEN("an MPI-backed vector allocator") {
    std::vector<int, mpl::mpi_allocator<int>> values(8);

    WHEN("values are stored in the vector") {
      for (std::size_t i{0}; i < values.size(); ++i)
        values[i] = static_cast<int>(i * 3);

      THEN("the values are retained and allocated storage meets type alignment") {
        for (std::size_t i{0}; i < values.size(); ++i)
          CHECK(values[i] == static_cast<int>(i * 3));

        struct alignas(64) over_aligned_value {
          int value;
        };
        std::vector<over_aligned_value, mpl::mpi_allocator<over_aligned_value>> aligned_values(
            1);
        CHECK(reinterpret_cast<std::uintptr_t>(aligned_values.data()) %
                  alignof(over_aligned_value) ==
              0);
      }
    }
  }

  GIVEN("an allocator request larger than the MPI address range") {
    mpl::mpi_allocator<int> allocator;

    THEN("allocation is rejected before byte-count conversion can overflow") {
      REQUIRE_THROWS_AS(allocator.allocate(allocator.max_size() + 1),
                        std::bad_array_new_length);
    }
  }
}
