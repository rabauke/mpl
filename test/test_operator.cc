#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

  struct first_operand {
    int operator()(const int &in, const int &) const {
      return in;
    }
  };


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
    auto int_gen = [](const int r) { return (r * 7 + 3) % 11 - 5; };
    auto double_gen = [](const int r) { return static_cast<double>((r * 7 + 3) % 11) - 5.5; };

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
    auto unsigned_gen = [](const int r) { return (0xA5u >> (r % 4)) ^ (1u << (r % 8)); };

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
    auto bool_gen = [](const int r) { return r % 3 == 0; };

    WHEN("reducing with mpl::logical_and, mpl::logical_or and mpl::logical_xor") {
      THEN("all processes receive the sequential logical result") {
        REQUIRE(reduce_op_test(mpl::logical_and<bool>(), bool_gen));
        REQUIRE(reduce_op_test(mpl::logical_or<bool>(), bool_gen));
        REQUIRE(reduce_op_test(mpl::logical_xor<bool>(), bool_gen));
      }
    }
  }
}


SCENARIO("local reductions update scalar and contiguous data", "[operator][reduce_local]") {
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};

  GIVEN("two scalar values and two contiguous buffers") {
    const int scalar_in{7};
    int scalar_inout{20};
    int const custom_in{12};
    int custom_inout{5};
    int buffer_in[]{1, 2, 3};
    int buffer_inout[]{4, 5, 6};
    const mpl::contiguous_layout<int> layout{3};

    WHEN("applying the reduction functors locally") {
      mpl::reduce_local(mpl::plus<int>{}, scalar_in, scalar_inout);
      mpl::reduce_local(first_operand{}, custom_in, custom_inout);
      mpl::reduce_local(mpl::plus<int>{}, buffer_in, buffer_inout, layout);

      THEN("the in-out values contain the reduced results") {
        CHECK(scalar_inout == 27);
        CHECK(custom_inout == 12);
        CHECK(buffer_inout[0] == 5);
        CHECK(buffer_inout[1] == 7);
        CHECK(buffer_inout[2] == 9);
      }
    }
  }
}
