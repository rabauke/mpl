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


SCENARIO("global buffer flush operations complete buffered sends", "[environment][buffer]") {
  GIVEN("the world communicator and an attached global buffer") {
    const mpl::communicator &comm{mpl::environment::comm_world()};

    WHEN("the global buffer is flushed synchronously") {
      THEN("the buffered message is delivered and the buffer is detached") {
        buffered_send_test(comm, mpl::environment::buffer_attach,
                           mpl::environment::buffer_flush, mpl::environment::buffer_detach);
      }
    }

    WHEN("the global buffer is flushed asynchronously") {
      THEN("the buffered message is delivered after the flush request completes") {
        buffered_send_test(
            comm, mpl::environment::buffer_attach,
            []() {
              auto request{mpl::environment::buffer_iflush()};
              request.wait();
            },
            mpl::environment::buffer_detach);
      }
    }
  }
}


SCENARIO("communicator buffer flush operations complete buffered sends",
         "[communicator][buffer]") {
  GIVEN("the world communicator and a buffer attached to it") {
    const mpl::communicator &comm{mpl::environment::comm_world()};

    WHEN("the communicator buffer is flushed synchronously") {
      THEN("the buffered message is delivered and the buffer is detached") {
        buffered_send_test(
            comm, [&comm](void *buffer, MPI_Count size) { comm.attach_buffer(buffer, size); },
            [&comm]() { comm.flush_buffer(); }, [&comm]() { return comm.detach_buffer(); });
      }
    }

    WHEN("the communicator buffer is flushed asynchronously") {
      THEN("the buffered message is delivered after the flush request completes") {
        buffered_send_test(
            comm, [&comm](void *buffer, MPI_Count size) { comm.attach_buffer(buffer, size); },
            [&comm]() {
              auto request{comm.iflush_buffer()};
              request.wait();
            },
            [&comm]() { return comm.detach_buffer(); });
      }
    }
  }
}
