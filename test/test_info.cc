#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


SCENARIO("info objects store key-value configuration pairs", "[info]") {
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};

  GIVEN("a newly created info object") {
    mpl::info info_1;

    THEN("the info object is initially empty") {
      REQUIRE(info_1.size() == 0);
    }

    WHEN("key-value pairs are added") {
      info_1.set("Douglas Adams", "The Hitchhiker's Guide to the Galaxy");
      info_1.set("Isaac Asimov", "Nightfall");

      THEN("the size is updated and keys can be queried") {
        REQUIRE(info_1.size() == 2);
        REQUIRE(info_1.value("Isaac Asimov").value() == "Nightfall");
      }

      AND_WHEN("the info object is copy constructed") {
        mpl::info info_2{info_1};

        THEN("the copy contains the same key-value pairs and missing keys return empty") {
          REQUIRE(info_1.size() == 2);
          REQUIRE(info_2.value("Isaac Asimov").value() == "Nightfall");
          REQUIRE_FALSE(info_2.value("no such thing").has_value());
        }
      }
    }
  }
}
