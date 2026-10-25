#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <iterator>
#include <vector>
#include <mpl/mpl.hpp>


namespace {

template<typename T>
bool gather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    comm_world.gather(0, val, v.data());
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    return v == v_expected;
  } else {
    comm_world.gather(0, val);
    return true;
  }
}


template<typename T>
bool gather_test(const std::vector<T> &send, const std::vector<T> &expected,
                 const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    comm_world.gather(0, send.data(), l, v.data(), l);
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    return v == v_expected;
  } else {
    comm_world.gather(0, send.data(), l);
    return true;
  }
}


template<typename T>
bool igather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    auto r{comm_world.igather(0, val, v.data())};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    r.wait();
    return v == v_expected;
  } else {
    auto r{comm_world.igather(0, val)};
    r.wait();
    return true;
  }
}


template<typename T>
bool igather_test(const std::vector<T> &send, const std::vector<T> &expected,
                  const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    auto r{comm_world.igather(0, send.data(), l, v.data(), l)};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    r.wait();
    return v == v_expected;
  } else {
    auto r{comm_world.igather(0, send.data(), l)};
    r.wait();
    return true;
  }
}

}  // namespace


SCENARIO("gathering data from all communicator processes to the root process",
         "[communicator][gather]") {
  GIVEN("a world communicator with sender processes and root rank 0") {
    WHEN("gathering fundamental, array, and layout-configured vector data synchronously") {
      THEN("root receives data gathered across all processes") {
        REQUIRE(gather_test(1.0));
        REQUIRE(gather_test(std::array{1, 2, 3, 4}));
        {
          const std::vector send{1, 2, 3, 4, 5, 6};
          const std::vector expected{0, 2, 3, 0, 5, 0};
          mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
          l.resize(0, 6);
          REQUIRE(gather_test(send, expected, l));
        }
      }
    }

    WHEN("gathering data asynchronously using igather") {
      THEN("root receives gathered data upon request completion") {
        REQUIRE(igather_test(1.0));
        REQUIRE(igather_test(std::array{1, 2, 3, 4}));
        {
          const std::vector send{1, 2, 3, 4, 5, 6};
          const std::vector expected{0, 2, 3, 0, 5, 0};
          mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
          l.resize(0, 6);
          REQUIRE(igather_test(send, expected, l));
        }
      }
    }
  }
}
