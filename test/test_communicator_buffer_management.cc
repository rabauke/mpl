#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>
#include <mpl/mpl.hpp>


namespace {

  template<typename Attach, typename Flush, typename Detach>
  void buffered_send_test(const mpl::communicator &comm, Attach attach, Flush flush,
                          Detach detach) {
    REQUIRE(comm.size() >= 2);
    constexpr int sent{42};
    if (comm.rank() == 0) {
      std::vector<char> buffer(static_cast<std::size_t>(comm.bsend_size<int>()));
      attach(buffer.data(), static_cast<MPI_Count>(buffer.size()));
      comm.bsend(sent, 1);
      flush();
      const auto detached{detach()};
      CHECK(detached.first == buffer.data());
      CHECK(detached.second == static_cast<MPI_Count>(buffer.size()));
    } else if (comm.rank() == 1) {
      int received{0};
      comm.recv(received, 0);
      CHECK(received == sent);
    }
  }

}  // namespace


TEST_CASE("global buffer flush operations complete buffered sends", "[environment][buffer]") {
  const mpl::communicator &comm{mpl::environment::comm_world()};

  SECTION("blocking flush") {
    buffered_send_test(comm, mpl::environment::buffer_attach, mpl::environment::buffer_flush,
                       mpl::environment::buffer_detach);
  }

  SECTION("nonblocking flush") {
    buffered_send_test(
        comm, mpl::environment::buffer_attach,
        []() {
          auto request{mpl::environment::buffer_iflush()};
          request.wait();
        },
        mpl::environment::buffer_detach);
  }
}



TEST_CASE("communicator buffer flush operations complete buffered sends",
          "[communicator][buffer]") {
  const mpl::communicator &comm{mpl::environment::comm_world()};

  SECTION("blocking flush") {
    buffered_send_test(
        comm, [&comm](void *buffer, MPI_Count size) { comm.attach_buffer(buffer, size); },
        [&comm]() { comm.flush_buffer(); }, [&comm]() { return comm.detach_buffer(); });
  }

  SECTION("nonblocking flush") {
    buffered_send_test(
        comm, [&comm](void *buffer, MPI_Count size) { comm.attach_buffer(buffer, size); },
        [&comm]() {
          auto request{comm.iflush_buffer()};
          request.wait();
        },
        [&comm]() { return comm.detach_buffer(); });
  }
}