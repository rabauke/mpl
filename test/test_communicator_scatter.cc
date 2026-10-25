#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <iterator>
#include <vector>
#include <mpl/mpl.hpp>


namespace {

template<typename T>
bool scatter_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T recv;
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size(), val);
    comm_world.scatter(0, v.data(), recv);
  } else {
    comm_world.scatter(0, recv);
  }
  return recv == val;
}


template<typename T>
bool scatter_test(const std::vector<T> &send, const std::vector<T> &expected,
                  const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> recv(send.size());
  if (comm_world.rank() == 0) {
    std::vector<T> v_send;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_send));
    comm_world.scatter(0, v_send.data(), l, recv.data(), l);
  } else {
    comm_world.scatter(0, recv.data(), l);
  }
  return recv == expected;
}


template<typename T>
bool iscatter_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T recv;
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size(), val);
    auto r{comm_world.iscatter(0, v.data(), recv)};
    r.wait();
  } else {
    auto r{comm_world.iscatter(0, recv)};
    r.wait();
  }
  return recv == val;
}


template<typename T>
bool iscatter_test(const std::vector<T> &send, const std::vector<T> &expected,
                   const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::vector<T> recv(send.size());
  if (comm_world.rank() == 0) {
    std::vector<T> v_send;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_send));
    auto r{comm_world.iscatter(0, v_send.data(), l, recv.data(), l)};
    r.wait();
  } else {
    auto r{comm_world.iscatter(0, recv.data(), l)};
    r.wait();
  }
  return recv == expected;
}

}  // namespace


SCENARIO("scattering data from root process to all communicator processes",
         "[communicator][scatter]") {
  GIVEN("a world communicator with root process rank 0") {
    WHEN("scattering fundamental, array, and layout-configured vector data synchronously") {
      THEN("each process receives its expected portion") {
        REQUIRE(scatter_test(1.0));
        REQUIRE(scatter_test(std::array{1, 2, 3, 4}));
        {
          const std::vector send{1, 2, 3, 4, 5, 6};
          const std::vector expected{0, 2, 3, 0, 5, 0};
          mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
          l.resize(0, 6);
          REQUIRE(scatter_test(send, expected, l));
        }
      }
    }

    WHEN("scattering data asynchronously using iscatter") {
      THEN("each process receives its expected portion upon request completion") {
        REQUIRE(iscatter_test(1.0));
        REQUIRE(iscatter_test(std::array{1, 2, 3, 4}));
        {
          const std::vector send{1, 2, 3, 4, 5, 6};
          const std::vector expected{0, 2, 3, 0, 5, 0};
          mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
          l.resize(0, 6);
          REQUIRE(iscatter_test(send, expected, l));
        }
      }
    }
  }
}
