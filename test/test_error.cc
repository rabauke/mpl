// Negative-path tests for the mpl exception hierarchy.
// Checks guarded by MPL_DEBUG are only compiled when the test build defines MPL_DEBUG
// (option MPL_TEST_DEBUG_CHECKS). Without it, those tests are skipped and not passed silently.
// Spawn-based error paths are intentionally not tested here (they require spawning processes).

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <limits>
#include <mpl/mpl.hpp>


SCENARIO("mpl error types derive from mpl::error and std::exception", "[error]") {
  GIVEN("the mpl exception types and std::exception") {
    WHEN("the static type relationships are evaluated") {
      THEN(
          "every specific mpl error derives from mpl::error and mpl::error derives from "
          "std::exception") {
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_rank>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_tag>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_size>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_count>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_displacement>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_layout>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_dim>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_datatype_bound>);
        CHECK(std::is_base_of_v<mpl::error, mpl::invalid_argument>);
        CHECK(std::is_base_of_v<std::exception, mpl::error>);
      }
    }
  }
}


SCENARIO("mpl::invalid_argument is thrown and caught as its specific type", "[error]") {
  GIVEN("an mpl::invalid_argument exception") {
    WHEN("it is thrown") {
      THEN("it is caught as invalid_argument, mpl::error and std::exception") {
        CHECK_THROWS_AS(throw mpl::invalid_argument(), mpl::invalid_argument);
        CHECK_THROWS_AS(throw mpl::invalid_argument(), mpl::error);
        CHECK_THROWS_AS(throw mpl::invalid_argument(), std::exception);
      }
    }
  }
}

#ifdef MPL_DEBUG

SCENARIO("invalid destination and source ranks throw invalid_rank", "[error]") {
  GIVEN("the world communicator and a value to send or receive") {
    const mpl::communicator &comm_world{mpl::environment::comm_world()};
    int value{0};
    WHEN("sending or receiving with a rank outside the communicator") {
      THEN("invalid_rank is thrown for each out-of-range destination and source") {
        REQUIRE_THROWS_AS(comm_world.send(value, comm_world.size()), mpl::invalid_rank);
        // -1 and -2 are not used here, because they may equal MPI_ANY_SOURCE or
        // MPI_PROC_NULL, depending on the MPI implementation
        REQUIRE_THROWS_AS(comm_world.send(value, -100), mpl::invalid_rank);
        REQUIRE_THROWS_AS(comm_world.recv(value, comm_world.size()), mpl::invalid_rank);
        REQUIRE_THROWS_AS(comm_world.recv(value, -100), mpl::invalid_rank);
      }
    }
  }
}


SCENARIO("proc_null and any_source are accepted by rank checks", "[error]") {
  GIVEN("the world communicator and a value to send or receive") {
    const mpl::communicator &comm_world{mpl::environment::comm_world()};
    int value{0};
    WHEN("receiving from any_source or sending to proc_null") {
      THEN("neither operation throws") {
        // the receive needs a matching message, otherwise it would block forever
        const int send_value{0};
        mpl::irequest request{comm_world.isend(send_value, comm_world.rank(), mpl::tag_t{40})};
        CHECK_NOTHROW(comm_world.recv(value, mpl::any_source, mpl::tag_t{40}));
        request.wait();
        CHECK_NOTHROW(comm_world.send(value, mpl::proc_null));
      }
    }
  }
}


SCENARIO("negative and oversized send tags throw invalid_tag", "[error]") {
  GIVEN("the world communicator, a valid destination and a value to send") {
    const mpl::communicator &comm_world{mpl::environment::comm_world()};
    int value{0};
    const int dest{comm_world.rank()};
    WHEN("sending with a tag below zero or above the upper bound") {
      THEN("invalid_tag is thrown") {
        REQUIRE_THROWS_AS(comm_world.send(value, dest, mpl::tag_t(-1)), mpl::invalid_tag);
        if (static_cast<int>(mpl::tag_t::up()) < std::numeric_limits<int>::max())
          REQUIRE_THROWS_AS(
              comm_world.send(value, dest, mpl::tag_t(static_cast<int>(mpl::tag_t::up()) + 1)),
              mpl::invalid_tag);
      }
    }
  }
}


SCENARIO("cartesian dimension mismatch throws invalid_size", "[error]") {
  GIVEN("a cartesian communicator and an included_tags vector of the wrong size") {
    const mpl::communicator &comm_world{mpl::environment::comm_world()};
    mpl::cartesian_communicator::dimensions dims{mpl::cartesian_communicator::periodic};
    mpl::cartesian_communicator comm_c{comm_world, mpl::dims_create(comm_world.size(), dims)};
    mpl::cartesian_communicator::included_tags wrong_size{2};
    WHEN("constructing a cartesian communicator from the mismatched included_tags") {
      THEN("invalid_size is thrown") {
        REQUIRE_THROWS_AS(mpl::cartesian_communicator(comm_c, wrong_size), mpl::invalid_size);
      }
    }
  }
}

#else

SCENARIO("MPL_DEBUG-guarded error checks are skipped without MPL_DEBUG", "[error]") {
  GIVEN("a build without MPL_DEBUG defined") {
    WHEN("the MPL_DEBUG-guarded checks are reached") {
      THEN("they are reported as skipped with a warning and succeed") {
        WARN(
            "Rank, tag and dimension error tests are skipped; configure with "
            "-DMPL_TEST_DEBUG_CHECKS=ON to run them.");
        SUCCEED();
      }
    }
  }
}

#endif

SCENARIO("mpi_error and its derived types form the expected hierarchy", "[error]") {
  GIVEN("the mpl mpi_error type and its derived types") {
    WHEN("the static type relationships are evaluated") {
      THEN("invalid_rank is not an mpi_error while io_failure and request_error are") {
        STATIC_REQUIRE_FALSE(std::is_base_of_v<mpl::mpi_error, mpl::invalid_rank>);
        STATIC_REQUIRE(std::is_base_of_v<mpl::error, mpl::mpi_error>);
        STATIC_REQUIRE(std::is_base_of_v<mpl::mpi_error, mpl::io_failure>);
        STATIC_REQUIRE(std::is_base_of_v<mpl::mpi_error, mpl::request_error>);
      }
    }
  }
}


SCENARIO("mpi_error carries code, class and message", "[error]") {
  GIVEN("an mpi_error constructed from MPI_ERR_TRUNCATE") {
    const mpl::mpi_error e{MPI_ERR_TRUNCATE};
    WHEN("its error class, error code and message are queried") {
      THEN("class and code are MPI_ERR_TRUNCATE and the message is non-empty") {
        CHECK(e.error_class() == MPI_ERR_TRUNCATE);
        CHECK(e.error_code() == MPI_ERR_TRUNCATE);
        CHECK(not std::string{e.what()}.empty());
      }
    }
  }
}


// Lifecycle inquiries are also covered by test_error_no_mpi.
SCENARIO("error helpers query MPI while initialized", "[error]") {
  GIVEN("an initialized MPI environment and a valid error code") {
    [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};
    WHEN("the error class and message are determined") {
      THEN("the class and message agree with direct MPI inquiries") {
        int expected_class{0};
        char message[MPI_MAX_ERROR_STRING + 1]{};
        int length{0};
        REQUIRE(MPI_Error_class(MPI_ERR_TRUNCATE, &expected_class) == MPI_SUCCESS);
        REQUIRE(MPI_Error_string(MPI_ERR_TRUNCATE, message, &length) == MPI_SUCCESS);
        CHECK(mpl::detail::error_class_of(MPI_ERR_TRUNCATE) == expected_class);
        CHECK(mpl::detail::error_message(MPI_ERR_TRUNCATE) == std::string(message, length));
      }
    }
  }
}


SCENARIO("receive into too small buffer throws mpi_error with class MPI_ERR_TRUNCATE",
         "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has sent two ints to itself and posts a receive for one int") {
    int send_data[2]{1, 2};
    WHEN("the too small blocking receive is performed on rank 0") {
      if (comm_world.rank() == 0) {
        auto send{
            comm_world.isend(send_data, mpl::contiguous_layout<int>(2), 0, mpl::tag_t{1})};
        bool caught{false};
        try {
          int recv_data{0};
          comm_world.recv(recv_data, 0, mpl::tag_t{1});
        } catch (const mpl::error &e) {
          const auto *e_mpi{dynamic_cast<const mpl::mpi_error *>(&e)};
          CHECK(e_mpi != nullptr);
          if (e_mpi != nullptr)
            CHECK(e_mpi->error_class() == MPI_ERR_TRUNCATE);
          CHECK(not std::string{e.what()}.empty());
          caught = true;
        }
        send.wait();
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is caught") {
          CHECK(caught);
        }
      }
    }
  }
}


SCENARIO("waitall with truncated receive throws request_error", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has sent two ints to itself and posts a non-blocking receive for one int") {
    int send_data[2]{1, 2};
    WHEN("waitall is called on rank 0") {
      if (comm_world.rank() == 0) {
        mpl::irequest_pool requests;
        int recv_data{0};
        auto send{
            comm_world.isend(send_data, mpl::contiguous_layout<int>(2), 0, mpl::tag_t{2})};
        requests.push(comm_world.irecv(recv_data, 0, mpl::tag_t{2}));
        bool caught{false};
        try {
          requests.waitall();
        } catch (const mpl::request_error &e) {
          caught = true;
          CHECK(e.error_class() == MPI_ERR_TRUNCATE);
          CHECK(e.failed_requests() == std::vector<int>{0});
        }
        send.wait();
        THEN("a request_error with class MPI_ERR_TRUNCATE listing request 0 is caught") {
          CHECK(caught);
        }
      }
    }
  }
}


SCENARIO("an error inside the helper thread of a container irecv is reported on wait",
         "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("a non-blocking receive of a std::vector from a rank outside the communicator") {
    std::vector<int> data;
    const int invalid_source{comm_world.size()};
#ifdef MPL_DEBUG
    WHEN("irecv is called with MPL_DEBUG checks enabled") {
      THEN("the invalid rank is rejected synchronously") {
        CHECK_THROWS_AS(comm_world.irecv(data, invalid_source, mpl::tag_t{30}),
                        mpl::invalid_rank);
      }
    }
#else
    WHEN("the helper thread fails and the request is waited for") {
      mpl::irequest request{comm_world.irecv(data, invalid_source, mpl::tag_t{30})};
      bool caught{false};
      try {
        request.wait();
      } catch (const mpl::mpi_error &e) {
        caught = true;
        CHECK(e.error_class() == MPI_ERR_RANK);
      }
      THEN("the failure is reported as mpi_error instead of hanging or terminating") {
        CHECK(caught);
      }
    }
#endif
  }
}


namespace {

  int error_class(const mpl::status_t &status) {
    int result{MPI_SUCCESS};
    REQUIRE(MPI_Error_class(status.error(), &result) == MPI_SUCCESS);
    return result;
  }


  // Receives and discards all messages that a failed test may have left behind.
  class drain_guard {
    const mpl::communicator &comm_;
    mpl::tag_t tag_;

  public:
    drain_guard(const mpl::communicator &comm, mpl::tag_t t) : comm_{comm}, tag_{t} {
    }

    drain_guard(const drain_guard &) = delete;
    drain_guard &operator=(const drain_guard &) = delete;
    drain_guard(drain_guard &&) = delete;
    drain_guard &operator=(drain_guard &&) = delete;

    ~drain_guard() {
      try {
        while (comm_.iprobe(0, tag_)) {
          int data[2]{};
          comm_.recv(data, mpl::contiguous_layout<int>(2), 0, tag_);
        }
      } catch (...) {
        // destructor must not throw
      }
    }
  };


  class self_sends {
    const mpl::communicator &comm_;
    const int data_[2]{1, 2};
    std::vector<mpl::irequest> requests_;

  public:
    explicit self_sends(const mpl::communicator &comm) : comm_{comm} {
    }

    self_sends(const self_sends &) = delete;
    self_sends &operator=(const self_sends &) = delete;
    self_sends(self_sends &&) = delete;
    self_sends &operator=(self_sends &&) = delete;

    void send(mpl::tag_t tag, int count = 2) {
      requests_.push_back(comm_.isend(data_, mpl::contiguous_layout<int>(count), 0, tag));
    }

    ~self_sends() {
      for (auto &request : requests_) {
        try {
          request.wait();
        } catch (...) {
          // destructor must not throw
        }
      }
    }
  };


  // Calls f until it throws mpl::request_error or the limit is reached.
  template<typename F>
  std::optional<mpl::request_error> expect_request_error(const F &f) {
    for (int i{0}; i < 1000000; ++i) {
      try {
        f();
      } catch (const mpl::request_error &e) {
        return e;
      }
    }
    return std::nullopt;
  }


  template<typename F>
  std::optional<mpl::mpi_error> expect_mpi_error(const F &f) {
    for (int i{0}; i < 1000000; ++i) {
      try {
        f();
      } catch (const mpl::mpi_error &e) {
        return e;
      }
    }
    return std::nullopt;
  }


  class delayed_release {
    const mpl::communicator &control_;
    mpl::irequest_pool &pool_;
    bool pending_{false};
    bool released_{false};
    bool started_{false};

  public:
    delayed_release(const mpl::communicator &control, mpl::irequest_pool &pool,
                    bool requires_start = false)
        : control_{control}, pool_{pool}, started_{not requires_start} {
    }

    delayed_release(const delayed_release &) = delete;
    delayed_release &operator=(const delayed_release &) = delete;
    delayed_release(delayed_release &&) = delete;
    delayed_release &operator=(delayed_release &&) = delete;

    void posted() noexcept {
      pending_ = true;
    }

    void start() {
      if (not started_) {
        constexpr int signal{1};
        control_.send(signal, 1, mpl::tag_t{84});
        started_ = true;
      }
    }

    void release() {
      start();
      if (not released_) {
        constexpr int signal{1};
        control_.send(signal, 1, mpl::tag_t{91});
        released_ = true;
      }
      if (pending_) {
        pool_.wait(1);
        pending_ = false;
      }
    }

    ~delayed_release() {
      try {
        release();
      } catch (...) {
        // destructor must not throw
      }
    }
  };

}  // namespace


SCENARIO("testall with truncated receive throws request_error", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has a pending truncated non-blocking receive in a pool") {
    const mpl::tag_t tag{10};
    const drain_guard guard{comm_world, tag};
    if (comm_world.rank() == 0) {
      mpl::irequest_pool requests;
      int recv_data{0};
      self_sends sends{comm_world};
      sends.send(tag);
      requests.push(comm_world.irecv(recv_data, 0, tag));
      WHEN("testall is called repeatedly") {
        const auto e{expect_request_error([&]() { requests.testall(); })};
        THEN("a request_error with class MPI_ERR_TRUNCATE listing request 0 is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
          CHECK(e->failed_requests() == std::vector<int>{0});
          CHECK(e->completed_requests().empty());
          CHECK(error_class(requests.get_status(0)) == MPI_ERR_TRUNCATE);
        }
      }
    }
  }
}


SCENARIO("waitall reports failed and completed requests of a mixed pool", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has a matching and a truncated non-blocking receive in a pool") {
    const mpl::tag_t tag_ok{11};
    const mpl::tag_t tag_truncated{12};
    const drain_guard guard_ok{comm_world, tag_ok};
    const drain_guard guard_truncated{comm_world, tag_truncated};
    if (comm_world.rank() == 0) {
      mpl::irequest_pool requests;
      int recv_truncated{0};
      int recv_ok{0};
      self_sends sends{comm_world};
      sends.send(tag_ok, 1);
      sends.send(tag_truncated);
      requests.push(comm_world.irecv(recv_ok, 0, tag_ok));
      requests.push(comm_world.irecv(recv_truncated, 0, tag_truncated));
      WHEN("waitall is called") {
        std::optional<mpl::request_error> error;
        try {
          requests.waitall();
        } catch (const mpl::request_error &e) {
          error = e;
        }
        THEN("request 1 is reported as failed and request 0 as completed") {
          REQUIRE(error.has_value());
          CHECK(error->error_class() == MPI_ERR_TRUNCATE);
          CHECK(error->failed_requests() == std::vector<int>{1});
          CHECK(error->completed_requests() == std::vector<int>{0});
          CHECK(recv_ok == 1);
          CHECK(requests.get_status(0).error() == MPI_SUCCESS);
          CHECK(error_class(requests.get_status(1)) == MPI_ERR_TRUNCATE);
        }
      }
    }
  }
}


SCENARIO("waitsome and testsome keep the statuses of requests that completed", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has a matching and a truncated non-blocking receive in a pool") {
    const mpl::tag_t tag_ok{13};
    const mpl::tag_t tag_truncated{14};
    const drain_guard guard_ok{comm_world, tag_ok};
    const drain_guard guard_truncated{comm_world, tag_truncated};
    if (comm_world.rank() == 0) {
      mpl::irequest_pool requests_wait;
      mpl::irequest_pool requests_test;
      int recv_truncated_2{0};
      int recv_ok_2{0};
      int recv_truncated_1{0};
      int recv_ok_1{0};
      self_sends sends{comm_world};
      sends.send(tag_ok, 1);
      sends.send(tag_ok, 1);
      sends.send(tag_truncated);
      sends.send(tag_truncated);
      requests_wait.push(comm_world.irecv(recv_ok_1, 0, tag_ok));
      requests_wait.push(comm_world.irecv(recv_truncated_1, 0, tag_truncated));
      requests_test.push(comm_world.irecv(recv_ok_2, 0, tag_ok));
      requests_test.push(comm_world.irecv(recv_truncated_2, 0, tag_truncated));
      WHEN("waitsome is called until the failure is reported") {
        const auto e{expect_request_error([&]() { requests_wait.waitsome(); })};
        THEN("the failed request is listed and the statuses of the completed one are stored") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
          CHECK(e->failed_requests() == std::vector<int>{1});
          CHECK(error_class(requests_wait.get_status(1)) == MPI_ERR_TRUNCATE);
          // the matching request either completed in the failing call or in an earlier one
          if (not e->completed_requests().empty()) {
            CHECK(e->completed_requests() == std::vector<int>{0});
            CHECK(requests_wait.get_status(0).error() == MPI_SUCCESS);
            CHECK(requests_wait.get_status(0).tag() == tag_ok);
            CHECK(requests_wait.get_status(0).source() == 0);
          }
        }
      }
      WHEN("testsome is called until the failure is reported") {
        const auto e{expect_request_error([&]() { requests_test.testsome(); })};
        THEN("the failed request is listed and the statuses of the completed one are stored") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
          CHECK(e->failed_requests() == std::vector<int>{1});
          CHECK(error_class(requests_test.get_status(1)) == MPI_ERR_TRUNCATE);
          if (not e->completed_requests().empty()) {
            CHECK(e->completed_requests() == std::vector<int>{0});
            CHECK(requests_test.get_status(0).error() == MPI_SUCCESS);
            CHECK(requests_test.get_status(0).tag() == tag_ok);
            CHECK(requests_test.get_status(0).source() == 0);
          }
        }
      }
    }
  }
}


SCENARIO("single-completion calls throw mpi_error on a truncated receive", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  GIVEN("rank 0 has a pending truncated non-blocking receive") {
    const mpl::tag_t tag{15};
    const drain_guard guard{comm_world, tag};
    if (comm_world.rank() == 0) {
      int recv_data{0};
      self_sends sends{comm_world};
      WHEN("the pool waits for any request") {
        sends.send(tag);
        mpl::irequest_pool requests;
        requests.push(comm_world.irecv(recv_data, 0, tag));
        const auto e{expect_mpi_error([&]() { requests.waitany(); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
      WHEN("the pool tests for any request") {
        sends.send(tag);
        mpl::irequest_pool requests;
        requests.push(comm_world.irecv(recv_data, 0, tag));
        const auto e{expect_mpi_error([&]() { requests.testany(); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
      WHEN("the pool waits for a single request") {
        sends.send(tag);
        mpl::irequest_pool requests;
        requests.push(comm_world.irecv(recv_data, 0, tag));
        const auto e{expect_mpi_error([&]() { requests.wait(0); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
      WHEN("the pool tests a single request") {
        sends.send(tag);
        mpl::irequest_pool requests;
        requests.push(comm_world.irecv(recv_data, 0, tag));
        const auto e{expect_mpi_error([&]() { requests.test(0); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
      WHEN("a single request is waited for") {
        sends.send(tag);
        mpl::irequest request{comm_world.irecv(recv_data, 0, tag)};
        const auto e{expect_mpi_error([&]() { request.wait(); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
      WHEN("a single request is tested") {
        sends.send(tag);
        mpl::irequest request{comm_world.irecv(recv_data, 0, tag)};
        const auto e{expect_mpi_error([&]() { request.test(); })};
        THEN("an mpi_error with class MPI_ERR_TRUNCATE is thrown") {
          REQUIRE(e.has_value());
          CHECK(e->error_class() == MPI_ERR_TRUNCATE);
        }
      }
    }
  }
}


SCENARIO("check_statuses classifies synthetic statuses", "[error]") {
  using pool_type = mpl::impl::request_pool<mpl::irequest>;
  GIVEN("statuses with a successful, a pending and a failed request") {
    MPI_Status statuses[3]{};
    statuses[0].MPI_ERROR = MPI_SUCCESS;
    statuses[1].MPI_ERROR = MPI_ERR_PENDING;
    statuses[2].MPI_ERROR = MPI_ERR_TRUNCATE;
    WHEN("MPI_ERR_IN_STATUS is reported") {
      THEN("pending requests are neither failed nor completed") {
        try {
          pool_type::check_statuses(MPI_ERR_IN_STATUS, statuses, 3);
          FAIL("request_error expected");
        } catch (const mpl::request_error &e) {
          CHECK(e.error_class() == MPI_ERR_TRUNCATE);
          CHECK(e.failed_requests() == std::vector<int>{2});
          CHECK(e.completed_requests() == std::vector<int>{0});
        }
      }
      THEN("an index map is applied to failed and completed requests") {
        constexpr int indices[3]{5, 7, 9};
        try {
          pool_type::check_statuses(MPI_ERR_IN_STATUS, statuses, 3, indices);
          FAIL("request_error expected");
        } catch (const mpl::request_error &e) {
          CHECK(e.failed_requests() == std::vector<int>{9});
          CHECK(e.completed_requests() == std::vector<int>{5});
        }
      }
    }
    WHEN("an error other than MPI_ERR_IN_STATUS is reported") {
      THEN("a plain mpi_error is thrown instead of a request_error") {
        try {
          pool_type::check_statuses(MPI_ERR_TRUNCATE, statuses, 3);
          FAIL("mpi_error expected");
        } catch (const mpl::mpi_error &e) {
          CHECK(dynamic_cast<const mpl::request_error *>(&e) == nullptr);
          CHECK(e.error_class() == MPI_ERR_TRUNCATE);
        }
      }
    }
    WHEN("MPI_SUCCESS is reported") {
      THEN("nothing is thrown") {
        CHECK_NOTHROW(pool_type::check_statuses(MPI_SUCCESS, statuses, 3));
      }
    }
  }
  GIVEN("MPI_ERR_IN_STATUS without any failed request") {
    MPI_Status statuses[2]{};
    statuses[0].MPI_ERROR = MPI_SUCCESS;
    statuses[1].MPI_ERROR = MPI_ERR_PENDING;
    WHEN("the statuses are checked") {
      THEN("a plain mpi_error with the original code is thrown") {
        try {
          pool_type::check_statuses(MPI_ERR_IN_STATUS, statuses, 2);
          FAIL("mpi_error expected");
        } catch (const mpl::mpi_error &e) {
          CHECK(dynamic_cast<const mpl::request_error *>(&e) == nullptr);
          CHECK(e.error_code() == MPI_ERR_IN_STATUS);
        }
      }
    }
  }
}


SCENARIO("completion calls on an empty request pool are safe", "[error]") {
  GIVEN("an empty request pool") {
    WHEN("each completion call is performed") {
      THEN("nothing is thrown and no active requests are reported") {
        mpl::irequest_pool requests;
        CHECK_NOTHROW(requests.waitall());
        CHECK_NOTHROW(requests.testall());
        CHECK(requests.waitany().first == mpl::test_result::no_active_requests);
        CHECK(requests.testany().first == mpl::test_result::no_active_requests);
        const auto some_wait{requests.waitsome()};
        CHECK(some_wait.first == mpl::test_result::no_active_requests);
        CHECK(some_wait.second.empty());
        const auto some_test{requests.testsome()};
        CHECK(some_test.first == mpl::test_result::no_active_requests);
        CHECK(some_test.second.empty());
      }
    }
  }
}


SCENARIO("a delayed receive remains pending while another completes", "[error][request]") {
  const auto &world{mpl::environment::comm_world()};
  if (world.size() < 2) {
    WARN("Coordinated request coverage requires at least two ranks");
    return;
  }
  const mpl::communicator control{world};
  if (world.rank() == 0) {
    int previous{0};
    int good{0};
    int delayed{0};
    mpl::irequest_pool pool;
    delayed_release guard{control, pool, true};
    pool.push(world.irecv(previous, mpl::proc_null, mpl::tag_t{80}));
    pool.wait(0);
    pool.push(world.irecv(delayed, 1, mpl::tag_t{81}));
    guard.posted();
    pool.push(world.irecv(good, 1, mpl::tag_t{82}));
    CHECK_FALSE(pool.testall());
    CHECK(pool.testany().first == mpl::test_result::no_completed);
    CHECK(pool.testsome().first == mpl::test_result::no_completed);
    guard.start();
    const auto completed{pool.waitsome()};
    guard.release();
    CHECK(delayed == 99);
    CHECK(completed.first == mpl::test_result::completed);
    CHECK(completed.second == std::vector<mpl::irequest_pool::size_type>{2});
    CHECK(good == 42);
  } else if (world.rank() == 1) {
    int signal{0};
    control.recv(signal, 0, mpl::tag_t{84});
    constexpr int good{42};
    auto send_good{world.isend(good, 0, mpl::tag_t{82})};
    control.recv(signal, 0, mpl::tag_t{91});
    constexpr int delayed{99};
    auto send_delayed{world.isend(delayed, 0, mpl::tag_t{81})};
    send_good.wait();
    send_delayed.wait();
  }
}


SCENARIO("live mixed requests release a delayed receive after an error", "[error][request]") {
  const auto &world{mpl::environment::comm_world()};
  if (world.size() < 2) {
    WARN("Coordinated request coverage requires at least two ranks");
    return;
  }
  const mpl::communicator control{world};
  if (world.rank() == 0) {
    int previous{0};
    int delayed{0};
    int good{0};
    int truncated{0};
    mpl::irequest_pool pool;
    delayed_release guard{control, pool};
    pool.push(world.irecv(previous, mpl::proc_null, mpl::tag_t{87}));
    pool.wait(0);
    pool.push(world.irecv(delayed, 1, mpl::tag_t{88}));
    guard.posted();
    pool.push(world.irecv(good, 1, mpl::tag_t{89}));
    pool.push(world.irecv(truncated, 1, mpl::tag_t{90}));
    std::optional<mpl::request_error> error;
    for (int i{0}; i < 1000000 and not error; ++i) {
      try {
        pool.testsome();
      } catch (const mpl::request_error &e) {
        error = e;
      }
    }
    const bool error_before_release{error.has_value()};
    guard.release();
    if (not error)
      error = expect_request_error([&]() { pool.testsome(); });
    CHECK(delayed == 99);
    CHECK(good == 42);
    REQUIRE(error.has_value());
    CHECK(error->error_class() == MPI_ERR_TRUNCATE);
    CHECK(error->failed_requests() == std::vector<int>{3});
    if (error_before_release) {
      const auto &completed{error->completed_requests()};
      CHECK(std::find(completed.begin(), completed.end(), 1) == completed.end());
    }
  } else if (world.rank() == 1) {
    constexpr int good{42};
    constexpr int bad[2]{1, 2};
    auto send_good{world.isend(good, 0, mpl::tag_t{89})};
    auto send_bad{world.isend(bad, mpl::contiguous_layout<int>(2), 0, mpl::tag_t{90})};
    int signal{0};
    control.recv(signal, 0, mpl::tag_t{91});
    constexpr int delayed{99};
    auto send_delayed{world.isend(delayed, 0, mpl::tag_t{88})};
    send_good.wait();
    send_bad.wait();
    send_delayed.wait();
  }
}


SCENARIO("inactive persistent requests can be restarted", "[error][request]") {
  const auto &comm{mpl::environment::comm_world()};
  int value{0};
  mpl::prequest_pool pool;
  pool.push(comm.recv_init(value, mpl::proc_null, mpl::tag_t{86}));
  const auto check_no_active_requests{[&]() {
    CHECK(pool.testany().first == mpl::test_result::no_active_requests);
    CHECK(pool.testsome().first == mpl::test_result::no_active_requests);
    CHECK(pool.waitany().first == mpl::test_result::no_active_requests);
    CHECK(pool.waitsome().first == mpl::test_result::no_active_requests);
  }};
  check_no_active_requests();

  pool.startall();
  CHECK(pool.testall());
  check_no_active_requests();

  pool.startall();
  pool.waitall();
  check_no_active_requests();
}


namespace {

  template<typename Communicator>
  bool has_errors_return(const Communicator &comm) {
    MPI_Errhandler handler{MPI_ERRHANDLER_NULL};
    if (MPI_Comm_get_errhandler(comm.native_handle(), &handler) != MPI_SUCCESS)
      return false;
    const bool result{handler == MPI_ERRORS_RETURN};
    MPI_Errhandler_free(&handler);
    return result;
  }

}  // namespace


SCENARIO("MPI_ERRORS_RETURN is installed on every communicator creation path", "[error]") {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::communicator &comm_self{mpl::environment::comm_self()};
  GIVEN("the world and the self communicator of the environment") {
    THEN("both use MPI_ERRORS_RETURN") {
      CHECK(has_errors_return(comm_world));
      CHECK(has_errors_return(comm_self));
    }
  }
  GIVEN("communicators derived from the world communicator") {
    WHEN("a communicator is copied, moved and assigned") {
      const mpl::communicator copy{comm_world};
      const mpl::communicator moved{mpl::communicator{comm_world}};
      mpl::communicator copy_assigned;
      copy_assigned = comm_world;
      mpl::communicator move_assigned;
      move_assigned = mpl::communicator{comm_world};
      THEN("all of them use MPI_ERRORS_RETURN") {
        CHECK(has_errors_return(copy));
        CHECK(has_errors_return(moved));
        CHECK(has_errors_return(copy_assigned));
        CHECK(has_errors_return(move_assigned));
      }
    }
    WHEN("a communicator is created by split, create and create_group") {
      const mpl::communicator split{mpl::communicator::split, comm_world, 0};
      const mpl::group group_world{comm_world};
      const mpl::communicator created{mpl::communicator::comm_collective, comm_world,
                                      group_world};
      const mpl::communicator created_group{mpl::communicator::group_collective, comm_world,
                                            group_world, mpl::tag_t{3}};
      THEN("all of them use MPI_ERRORS_RETURN") {
        CHECK(has_errors_return(split));
        CHECK(has_errors_return(created));
        CHECK(has_errors_return(created_group));
      }
    }
  }
  GIVEN("topology communicators") {
    WHEN("Cartesian communicators are created, copied and partitioned") {
      const mpl::cartesian_communicator::dimensions dims{
          mpl::cartesian_communicator::periodic, mpl::cartesian_communicator::non_periodic};
      const mpl::cartesian_communicator comm_c{comm_world,
                                               mpl::dims_create(comm_world.size(), dims)};
      const mpl::cartesian_communicator comm_c_copy{comm_c};
      const mpl::cartesian_communicator comm_c_sub{
          comm_c,
          mpl::cartesian_communicator::included_tags{mpl::cartesian_communicator::included,
                                                     mpl::cartesian_communicator::excluded}};
      THEN("all of them use MPI_ERRORS_RETURN") {
        CHECK(has_errors_return(comm_c));
        CHECK(has_errors_return(comm_c_copy));
        CHECK(has_errors_return(comm_c_sub));
      }
    }
    WHEN("a graph communicator is created and copied") {
      if (comm_world.size() < 2) {
        // every process must be a node of the graph, which needs at least one edge
        WARN("Graph communicator test skipped, it requires at least two processes.");
        SUCCEED();
      } else {
        mpl::graph_communicator::edge_set es;
        for (int i{0}; i < comm_world.size(); ++i)
          es.add({i, (i + 1) % comm_world.size()});
        const mpl::graph_communicator comm_g{comm_world, es};
        const mpl::graph_communicator comm_g_copy{comm_g};
        THEN("both use MPI_ERRORS_RETURN") {
          CHECK(has_errors_return(comm_g));
          CHECK(has_errors_return(comm_g_copy));
        }
      }
    }
    WHEN("a distributed graph communicator is created and copied") {
      mpl::distributed_graph_communicator::neighbours_set sources;
      mpl::distributed_graph_communicator::neighbours_set destination;
      if (comm_world.rank() == 0) {
        for (int i{1}; i < comm_world.size(); ++i) {
          sources.add(i);
          destination.add({i, 0});
        }
      } else {
        sources.add(0);
        destination.add({0, 0});
      }
      const mpl::distributed_graph_communicator comm_d{comm_world, sources, destination};
      const mpl::distributed_graph_communicator comm_d_copy{comm_d};
      THEN("both use MPI_ERRORS_RETURN") {
        CHECK(has_errors_return(comm_d));
        CHECK(has_errors_return(comm_d_copy));
      }
    }
  }
}
