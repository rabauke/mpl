#include <catch2/catch_test_macros.hpp>
#include <array>
#include <vector>
#include <mpl/mpl.hpp>


namespace {

template<typename T>
bool bcast_test(const T &val) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  comm_world.bcast(0, x);
  return x == val;
}


template<typename T>
bool bcast_test(const std::vector<T> &send, const std::vector<T> &expected,
                const mpl::layout<T> &l) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  comm_world.bcast(0, x.data(), l);
  if (comm_world.rank() == 0)
    return x == send;
  return x == expected;
}


template<typename T>
bool ibcast_test(const T &val) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  auto r{comm_world.ibcast(0, x)};
  r.wait();
  return x == val;
}


template<typename T>
bool ibcast_test(const std::vector<T> &send, const std::vector<T> &expected,
                 const mpl::layout<T> &l) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  auto r{comm_world.ibcast(0, x.data(), l)};
  r.wait();
  if (comm_world.rank() == 0)
    return x == send;
  return x == expected;
}

}  // namespace


SCENARIO("broadcasting data from root rank to all communicator processes",
         "[communicator][bcast]") {
  GIVEN("a root process with scalar, array, and layout-configured vector data") {
    WHEN("broadcasting data synchronously using bcast") {
      THEN("all ranks receive the broadcasted data") {
        REQUIRE(bcast_test(1.0));
        REQUIRE(bcast_test(std::array{1, 2, 3, 4}));
        REQUIRE(bcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                           mpl::indexed_layout<int>{{{2, 1}, {1, 4}}}));
      }
    }

    WHEN("broadcasting data asynchronously using ibcast") {
      THEN("all ranks receive the broadcasted data after completion") {
        REQUIRE(ibcast_test(1.0));
        REQUIRE(ibcast_test(std::array{1, 2, 3, 4}));
        REQUIRE(ibcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                            mpl::indexed_layout<int>{{{2, 1}, {1, 4}}}));
      }
    }
  }
}
