#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("displacements container manages memory offsets for layout indexing",
         "[displacements]") {
  GIVEN("displacements constructed with sizes and initializer lists") {
    mpl::displacements displacements_1(10);
    mpl::displacements displacements_2{1, 2, 3};

    THEN("initial sizes match constructor arguments") {
      REQUIRE(displacements_1.size() == 10);
      REQUIRE(displacements_2.size() == 3);
    }

    WHEN("appending additional displacements") {
      displacements_2.push_back(10);

      THEN("the container size increases accordingly") {
        REQUIRE(displacements_2.size() == 4);
      }
    }
  }
}
