#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <vector>
#include <mpl/mpl.hpp>


namespace {

template<typename T>
bool allgather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> v(comm_world.size());
  comm_world.allgather(val, v.data());
  return std::all_of(v.begin(), v.end(), [&val](const auto &x) { return x == val; });
}


template<typename T>
bool iallgather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> v(comm_world.size());
  auto r{comm_world.iallgather(val, v.data())};
  r.wait();
  return std::all_of(v.begin(), v.end(), [&val](const auto &x) { return x == val; });
}

}  // namespace


SCENARIO("allgather distributes data from all processes to all processes",
         "[communicator][allgather]") {
  GIVEN("a world communicator where every process contributes data") {
    WHEN("performing allgather synchronously") {
      THEN("all processes receive the full collection of data") {
        REQUIRE(allgather_test(1.0));
        REQUIRE(allgather_test(std::array{1, 2, 3, 4}));
      }
    }

    WHEN("performing allgather asynchronously using iallgather") {
      THEN("all processes receive the full collection upon completion") {
        REQUIRE(iallgather_test(1.0));
        REQUIRE(iallgather_test(std::array{1, 2, 3, 4}));
      }
    }
  }
}
