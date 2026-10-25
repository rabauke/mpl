#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

template<typename T>
bool alltoall_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  comm_world.alltoall(send_data.data(), recv_data.data());
  return recv_data == expected;
}


template<typename T>
bool alltoall_layout_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mpl::indexed_layout<T> sendl({{1, 0}, {1, 2}});
  mpl::vector_layout<T> recvl(2);
  comm_world.alltoall(send_data.data(), sendl, recv_data.data(), recvl);
  return recv_data == expected;
}


template<typename T>
bool alltoall_inplace_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  comm_world.alltoall(sendrecv_data.data());
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoall_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{comm_world.ialltoall(send_data.data(), recv_data.data())};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_layout_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mpl::indexed_layout<T> sendl({{1, 0}, {1, 2}});
  mpl::vector_layout<T> recvl(2);
  auto r{comm_world.ialltoall(send_data.data(), sendl, recv_data.data(), recvl)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_inplace_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{comm_world.ialltoall(sendrecv_data.data())};
  r.wait();
  return sendrecv_data == expected;
}

}  // namespace


SCENARIO("all-to-all communication transposes data across all processes",
         "[communicator][alltoall]") {
  GIVEN("a world communicator with per-process distinct send data") {
    WHEN("performing alltoall synchronously in out-of-place, layout, and in-place modes") {
      THEN("each process receives data transposed from all processes") {
        REQUIRE(alltoall_test(1.0));
        REQUIRE(alltoall_test(tuple{1, 2.0}));

        REQUIRE(alltoall_layout_test(1.0));
        REQUIRE(alltoall_layout_test(tuple{1, 2.0}));

        REQUIRE(alltoall_inplace_test(1.0));
        REQUIRE(alltoall_inplace_test(tuple{1, 2.0}));
      }
    }

    WHEN("performing alltoall asynchronously using ialltoall") {
      THEN("each process receives transposed data upon request completion") {
        REQUIRE(ialltoall_test(1.0));
        REQUIRE(ialltoall_test(tuple{1, 2.0}));

        REQUIRE(ialltoall_layout_test(1.0));
        REQUIRE(ialltoall_layout_test(tuple{1, 2.0}));

        REQUIRE(ialltoall_inplace_test(1.0));
        REQUIRE(ialltoall_inplace_test(tuple{1, 2.0}));
      }
    }
  }
}
