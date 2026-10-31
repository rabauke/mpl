#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

template<typename F, typename G>
bool reduce_op_test(F f, G gen) {
  using T = decltype(gen(0));
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T x{gen(comm_world.rank())};
  T y{};
  comm_world.allreduce(f, x, y);
  T expected{gen(0)};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, gen(i));
  T x_inplace{gen(comm_world.rank())};
  comm_world.allreduce(f, x_inplace);
  return y == expected and x_inplace == expected;
}

}  // namespace


SCENARIO("allreduce with max and min operators", "[operator][allreduce]") {
  GIVEN("a world communicator and signed integer and floating point values per process") {
    auto int_gen = [](int r) { return (r * 7 + 3) % 11 - 5; };
    auto double_gen = [](int r) { return static_cast<double>((r * 7 + 3) % 11) - 5.5; };

    WHEN("reducing with mpl::max and mpl::min") {
      THEN("all processes receive the sequential maximum and minimum") {
        REQUIRE(reduce_op_test(mpl::max<int>(), int_gen));
        REQUIRE(reduce_op_test(mpl::min<int>(), int_gen));
        REQUIRE(reduce_op_test(mpl::max<double>(), double_gen));
        REQUIRE(reduce_op_test(mpl::min<double>(), double_gen));
      }
    }
  }
}

SCENARIO("allreduce with bitwise operators", "[operator][allreduce]") {
  GIVEN("a world communicator and unsigned integer values per process") {
    auto unsigned_gen = [](int r) { return (0xA5u >> (r % 4)) ^ (1u << (r % 8)); };

    WHEN("reducing with mpl::bit_and, mpl::bit_or and mpl::bit_xor") {
      THEN("all processes receive the sequential bitwise result") {
        REQUIRE(reduce_op_test(mpl::bit_and<unsigned>(), unsigned_gen));
        REQUIRE(reduce_op_test(mpl::bit_or<unsigned>(), unsigned_gen));
        REQUIRE(reduce_op_test(mpl::bit_xor<unsigned>(), unsigned_gen));
      }
    }
  }
}

SCENARIO("allreduce with logical operators", "[operator][allreduce]") {
  GIVEN("a world communicator and boolean values per process") {
    auto bool_gen = [](int r) { return r % 3 == 0; };

    WHEN("reducing with mpl::logical_and, mpl::logical_or and mpl::logical_xor") {
      THEN("all processes receive the sequential logical result") {
        REQUIRE(reduce_op_test(mpl::logical_and<bool>(), bool_gen));
        REQUIRE(reduce_op_test(mpl::logical_or<bool>(), bool_gen));
        REQUIRE(reduce_op_test(mpl::logical_xor<bool>(), bool_gen));
      }
    }
  }
}
