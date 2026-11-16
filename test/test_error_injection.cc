#include <catch2/catch_test_macros.hpp>
#include <type_traits>
#include <utility>
#include <mpl/mpl.hpp>
#include "mpi_error_injection.hpp"

namespace {

  template<typename T>
  void assignment_failure(T &destination, const T &source, injection::operation op) {
    const MPI_Comm original{destination.native_handle()};
    injection::reset();
    {
      injection::failure fail{op};
      CHECK_THROWS_AS(destination = source, mpl::mpi_error);
    }
    CHECK(destination.native_handle() == original);
    CHECK(destination.is_valid());
    int size{0};
    CHECK(PMPI_Comm_size(destination.native_handle(), &size) == MPI_SUCCESS);
    CHECK(size > 0);
    const auto events{injection::events()};
    CHECK(events.acquired == (op == injection::operation::dup ? 0 : 1));
    CHECK(events.released == events.acquired);
    CHECK(events.duplicate_releases == 0);
  }

  template<typename T>
  void assignment_contract(T &destination, const T &source) {
    assignment_failure(destination, source, injection::operation::dup);
    assignment_failure(destination, source, injection::operation::set_handler);
    injection::reset();
    const MPI_Comm original{destination.native_handle()};
    destination = destination;
    CHECK(destination.native_handle() == original);
    CHECK(injection::events().acquired == 0);
    destination = source;
    CHECK(destination.is_valid());
    CHECK(injection::events().acquired == 1);
  }
}  // namespace

SCENARIO("error inquiries preserve codes and use nonrecursive fallbacks", "[inquiry]") {
  GIVEN("an initialized MPI environment and injected inquiry failures") {
    (void)mpl::environment::comm_self();

    WHEN("constructing errors or routing MPI error codes") {
      THEN("codes are preserved and fallback behavior avoids incorrect classifications") {
        {
          injection::failure fail{injection::operation::error_class};
          const mpl::mpi_error error{MPI_ERR_TRUNCATE};
          CHECK(error.error_code() == MPI_ERR_TRUNCATE);
          CHECK(error.error_class() == MPI_ERR_OTHER);
        }
        for (const auto op : {injection::operation::error_string,
                              injection::operation::error_string_bad_length}) {
          injection::failure fail{op};
          const mpl::mpi_error error{MPI_ERR_TRUNCATE};
          CHECK(error.error_code() == MPI_ERR_TRUNCATE);
          CHECK(error.error_class() == MPI_ERR_TRUNCATE);
          CHECK(std::string{error.what()}.find(std::to_string(MPI_ERR_TRUNCATE)) !=
                std::string::npos);
        }
        {
          injection::failure fail{injection::operation::error_class};
          try {
            mpl::detail::check(MPI_ERR_FILE);
            FAIL("expected an MPI exception");
          } catch (const mpl::io_failure &) {
            FAIL("failed class inquiry must not route to io_failure");
          } catch (const mpl::mpi_error &error) {
            CHECK(error.error_code() == MPI_ERR_FILE);
          }
        }
        CHECK_THROWS_AS(mpl::detail::check(MPI_ERR_FILE), mpl::io_failure);
        try {
          mpl::detail::check(MPI_ERR_ARG);
          FAIL("expected an MPI exception");
        } catch (const mpl::io_failure &) {
          FAIL("MPI_ERR_ARG must not route to io_failure");
        } catch (const mpl::mpi_error &error) {
          CHECK(error.error_class() == MPI_ERR_ARG);
        }
      }
    }
  }
}

SCENARIO("user-defined error codes retain their registered class", "[inquiry]") {
  GIVEN("an MPI error class, code, and diagnostic registered by the user") {
    (void)mpl::environment::comm_self();
    int error_class{0};
    int code{0};
    REQUIRE(MPI_Add_error_class(&error_class) == MPI_SUCCESS);
    REQUIRE(MPI_Add_error_code(error_class, &code) == MPI_SUCCESS);
    REQUIRE(code != error_class);
    REQUIRE(MPI_Add_error_string(code, "custom diagnostic") == MPI_SUCCESS);
    int expected_class{0};
    REQUIRE(MPI_Error_class(code, &expected_class) == MPI_SUCCESS);

    WHEN("the registered error code is wrapped and checked") {
      const mpl::mpi_error error{code};

      THEN("the wrapper and routed exception preserve the registered details") {
        CHECK(error.error_code() == code);
        CHECK(error.error_class() == expected_class);
        CHECK(std::string{error.what()} == "custom diagnostic");
        try {
          mpl::detail::check(code);
          FAIL("expected an MPI exception");
        } catch (const mpl::io_failure &) {
          FAIL("user-defined class must not route to io_failure");
        } catch (const mpl::mpi_error &routed) {
          CHECK(routed.error_code() == code);
          CHECK(routed.error_class() == expected_class);
        }
      }
    }
  }
}

SCENARIO("communicator assignment exception specifications", "[assignment_traits]") {
  GIVEN("the communicator types provided by MPL") {
    WHEN("copy and move assignment traits are inspected") {
      THEN("copy assignment may throw and move assignment is nonthrowing") {
        CHECK_FALSE(std::is_nothrow_copy_assignable_v<mpl::communicator>);
        CHECK_FALSE(std::is_nothrow_copy_assignable_v<mpl::inter_communicator>);
        CHECK_FALSE(std::is_nothrow_copy_assignable_v<mpl::cartesian_communicator>);
        CHECK_FALSE(std::is_nothrow_copy_assignable_v<mpl::graph_communicator>);
        CHECK_FALSE(std::is_nothrow_copy_assignable_v<mpl::distributed_graph_communicator>);
        CHECK(std::is_nothrow_move_assignable_v<mpl::communicator>);
        CHECK(std::is_nothrow_move_assignable_v<mpl::inter_communicator>);
        CHECK(std::is_nothrow_move_assignable_v<mpl::cartesian_communicator>);
        CHECK(std::is_nothrow_move_assignable_v<mpl::graph_communicator>);
        CHECK(std::is_nothrow_move_assignable_v<mpl::distributed_graph_communicator>);
      }
    }
  }
}

SCENARIO("communicator copy assignment preserves ownership on failure", "[assignment]") {
  GIVEN("source and destination communicators") {
    const auto &self{mpl::environment::comm_self()};
    mpl::communicator source{self};
    mpl::communicator destination{self};

    WHEN("copy assignment encounters an injected MPI failure") {
      THEN("the destination remains valid and retains its ownership") {
        assignment_contract(destination, source);
      }
    }
  }
}

SCENARIO("successful replacement is released exactly once", "[assignment]") {
  GIVEN("a destination communicator that will be replaced by a copy") {
    const auto &self{mpl::environment::comm_self()};
    mpl::communicator source{self};

    WHEN("the destination is assigned and then destroyed") {
      {
        mpl::communicator destination{self};
        injection::reset();
        destination = source;
        CHECK(injection::events().acquired == 1);
        CHECK(injection::events().released == 0);
      }

      THEN("the replacement communicator is released exactly once") {
        const auto events{injection::events()};
        CHECK(events.acquired == 1);
        CHECK(events.released == 1);
        CHECK(events.duplicate_releases == 0);
      }
    }
  }
}

SCENARIO("cartesian copy assignment preserves topology on failure", "[assignment]") {
  GIVEN("source and destination one-dimensional Cartesian communicators") {
    const auto &self{mpl::environment::comm_self()};
    const mpl::cartesian_communicator::dimensions dims{mpl::cartesian_communicator::periodic};
    mpl::cartesian_communicator source{self, mpl::dims_create(1, dims)};
    mpl::cartesian_communicator destination{self, mpl::dims_create(1, dims)};

    WHEN("copy assignment encounters an injected MPI failure") {
      THEN("the destination remains valid with its topology intact") {
        assignment_contract(destination, source);
        CHECK(destination.dimensionality() == 1);
      }
    }
  }
}

SCENARIO("intercommunicator copy assignment preserves ownership on failure", "[assignment]") {
  GIVEN("an intercommunicator connecting each process to the other") {
    const auto &world{mpl::environment::comm_world()};
    REQUIRE(world.size() == 2);
    mpl::communicator local{mpl::communicator::split, world, world.rank()};
    mpl::inter_communicator source{local, 0, world, 1 - world.rank()};
    mpl::inter_communicator destination{source};

    WHEN("copy assignment encounters an injected MPI failure") {
      THEN("the destination remains valid with its remote group intact") {
        assignment_contract(destination, source);
        CHECK(destination.remote_size() == 1);
      }
    }
  }
}

SCENARIO("move and self assignment retain valid ownership", "[move]") {
  GIVEN("two communicators with valid ownership") {
    const auto &self{mpl::environment::comm_self()};
    mpl::communicator source{self};
    mpl::communicator destination{self};
    const MPI_Comm original{source.native_handle()};

    WHEN("a communicator is move-assigned to itself and then to another") {
      injection::reset();
      auto &alias{destination};
      destination = std::move(alias);
      destination = std::move(source);

      THEN("ownership remains valid and no new communicator is acquired") {
        CHECK(destination.is_valid());
        CHECK(destination.native_handle() == original);
        CHECK_FALSE(source.is_valid());
        CHECK(injection::events().acquired == 0);
      }
    }
  }
}

SCENARIO("failed owned adoption releases the newly created communicator", "[adoption]") {
  GIVEN("an initialized MPI environment") {
    const auto &self{mpl::environment::comm_self()};

    WHEN("owned communicator adoption fails while installing its error handler") {
      injection::reset();
      {
        injection::failure fail{injection::operation::set_handler};
        CHECK_THROWS_AS((mpl::communicator{mpl::communicator::split, self, 0}), mpl::mpi_error);
      }

      THEN("the newly created communicator is released exactly once") {
        const auto events{injection::events()};
        CHECK(events.acquired == 1);
        CHECK(events.released == 1);
        CHECK(events.duplicate_releases == 0);
      }
    }
  }
}

SCENARIO("failed borrowed adoption does not free the external communicator", "[adoption]") {
  GIVEN("an external communicator handle") {
    (void)mpl::environment::comm_self();
    MPI_Comm raw{MPI_COMM_NULL};
    REQUIRE(PMPI_Comm_dup(MPI_COMM_SELF, &raw) == MPI_SUCCESS);
    injection::reset();
    injection::track(raw);

    WHEN("borrowed communicator adoption fails while installing its error handler") {
      {
        injection::failure fail{injection::operation::set_handler};
        CHECK_THROWS_AS((mpl::mpi_communicator{raw}), mpl::mpi_error);
      }

      THEN("the external communicator remains valid and is not released") {
        const auto events{injection::events()};
        CHECK(events.released == 0);
        if (events.released == 0) {
          int size{0};
          CHECK(PMPI_Comm_size(raw, &size) == MPI_SUCCESS);
          CHECK(size == 1);
          CHECK(PMPI_Comm_free(&raw) == MPI_SUCCESS);
        }
      }
    }
  }
}

SCENARIO("struct datatype construction rolls back and can be retried", "[struct_datatype]") {
  GIVEN("a builder for a struct datatype") {
    (void)mpl::environment::comm_self();
    using builder = mpl::struct_builder<std::pair<int, double>>;

    WHEN("an MPI operation fails during datatype construction") {
      THEN("acquired datatypes are released and a later construction succeeds") {
        for (const auto op :
             {injection::operation::struct_create, injection::operation::struct_commit,
              injection::operation::resize, injection::operation::resized_commit}) {
          INFO(static_cast<int>(op));
          injection::reset();
          {
            injection::failure fail{op};
            CHECK_THROWS_AS(builder{}, mpl::mpi_error);
          }
          const auto failed{injection::datatype_events()};
          const int expected{op == injection::operation::struct_create    ? 0
                             : op == injection::operation::resized_commit ? 2
                                                                          : 1};
          CHECK(failed.acquired == expected);
          CHECK(failed.released == expected);
          CHECK(failed.duplicate_releases == 0);
          injection::reset();
          {
            builder retry;
            const auto live{injection::datatype_events()};
            CHECK(live.acquired == 2);
            CHECK(live.released == 1);
            CHECK(live.duplicate_releases == 0);
          }
          const auto completed{injection::datatype_events()};
          CHECK(completed.acquired == 2);
          CHECK(completed.released == 2);
          CHECK(completed.duplicate_releases == 0);
        }
      }
    }
  }
}

SCENARIO("layout adoption releases datatypes on failure and success", "[layout_adoption]") {
  GIVEN("an initialized MPI environment") {
    (void)mpl::environment::comm_self();

    WHEN("layout commitment fails and is then retried successfully") {
      injection::reset();
      {
        injection::failure fail{injection::operation::layout_commit};
        CHECK_THROWS_AS((mpl::contiguous_layout<int>{2}), mpl::mpi_error);
      }
      auto events{injection::datatype_events()};
      CHECK(events.acquired == 1);
      CHECK(events.released == 1);
      CHECK(events.duplicate_releases == 0);
      injection::reset();
      {
        mpl::contiguous_layout<int> layout{2};
        CHECK(injection::datatype_events().released == 0);
      }

      THEN("each acquired datatype is released exactly once") {
        events = injection::datatype_events();
        CHECK(events.acquired == 1);
        CHECK(events.released == 1);
        CHECK(events.duplicate_releases == 0);
      }
    }
  }
}

SCENARIO("buffer attachment owns allocation only after success", "[buffer_attachment]") {
  GIVEN("an initialized MPI environment") {
    (void)mpl::environment::comm_self();

    WHEN("buffer attachment fails and is then retried successfully") {
      injection::reset();
      {
        injection::failure fail{injection::operation::buffer_attach};
        CHECK_THROWS_AS((mpl::bsend_buffer{1024}), mpl::mpi_error);
      }
      auto events{injection::buffer_events()};
      CHECK(events.acquired == 1);
      CHECK(events.released == 1);
      CHECK(events.duplicate_releases == 0);
      injection::reset();
      {
        mpl::bsend_buffer buffer{1024};
        CHECK(injection::buffer_events().released == 0);
      }

      THEN("the buffer allocation is released exactly once") {
        events = injection::buffer_events();
        CHECK(events.acquired == 1);
        CHECK(events.released == 1);
        CHECK(events.duplicate_releases == 1);
        injection::reset();
      }
    }
  }
}

SCENARIO("communicator buffer attachment transfers ownership only after success",
         "[buffer_attachment]") {
  GIVEN("a communicator and a buffer to attach") {
    const auto &self{mpl::environment::comm_self()};

    WHEN("attachment or detachment encounters an injected failure") {
      injection::reset();
      {
        injection::failure fail{injection::operation::buffer_attach};
        CHECK_THROWS_AS((mpl::communicator_buffer{self, 1024}), mpl::mpi_error);
      }
      auto events{injection::buffer_events()};
      CHECK(events.acquired == 1);
      CHECK(events.released == 1);
      CHECK(events.duplicate_releases == 0);

      injection::reset();
      {
        mpl::communicator_buffer buffer{self, 1024};
        CHECK(injection::buffer_events().released == 0);
      }
      events = injection::buffer_events();
      CHECK(events.acquired == 1);
      CHECK(events.released == 1);
      CHECK(events.duplicate_releases == 1);

      injection::reset();
      void *buffer{nullptr};
      {
        injection::failure fail{injection::operation::buffer_detach};
        buffer = [&self]() {
          mpl::communicator_buffer attached{self, 1024};
          return injection::buffer_pointer();
        }();
      }
      events = injection::buffer_events();
      CHECK(events.acquired == 1);
      CHECK(events.released == 0);
      CHECK(events.duplicate_releases == 1);

      THEN("ownership transfers only on success and the detached allocation is retained") {
        void *detached_buffer{nullptr};
        MPI_Count detached_size{0};
        CHECK(PMPI_Comm_detach_buffer_c(self.native_handle(), &detached_buffer,
                                        &detached_size) == MPI_SUCCESS);
        CHECK(detached_buffer == buffer);
        ::operator delete(detached_buffer);
        events = injection::buffer_events();
        CHECK(events.released == 1);
        CHECK(events.duplicate_releases == 1);
        injection::reset();
      }
    }
  }
}

SCENARIO("cancellation errors are reported without abandoning requests", "[cancel_injection]") {
  GIVEN("pending nonblocking receives") {
    const auto &self{mpl::environment::comm_self()};
    int data[3]{};
    auto request{self.irecv(data[0], 0)};

    WHEN("request and pool cancellation fail once and are retried") {
      injection::reset();
      {
        injection::failure fail{injection::operation::cancel};
        CHECK_THROWS_AS(request.cancel(), mpl::mpi_error);
      }
      request.cancel();
      request.wait();

      mpl::irequest_pool pool;
      pool.push(self.irecv(data[1], 0));
      pool.push(self.irecv(data[2], 0));
      injection::reset();
      {
        injection::failure fail{injection::operation::cancel};
        CHECK_THROWS_AS(pool.cancelall(), mpl::mpi_error);
      }
      const int calls{injection::cancel_calls()};
      pool.cancelall();
      pool.waitall();

      THEN("the requests remain manageable and cancellation is retried once") {
        CHECK(calls == 1);
      }
    }
  }
}

SCENARIO("persistent request restarts after an injected cancellation failure",
         "[cancel_injection][request]") {
  GIVEN("a persistent receive request pool targeting MPI_PROC_NULL") {
    const auto &self{mpl::environment::comm_self()};
    int value{0};
    mpl::prequest_pool pool;
    pool.push(self.recv_init(value, mpl::proc_null));
    struct completion_guard {
      mpl::prequest_pool &pool;
      bool active{false};
      ~completion_guard() {
        if (active) {
          try {
            pool.waitall();
          } catch (...) {
          }
        }
      }
    } guard{pool};

    WHEN("cancellation fails and the request is subsequently restarted") {
      pool.start(0);
      guard.active = true;
      {
        injection::failure fail{injection::operation::cancel};
        CHECK_THROWS_AS(pool.cancel(0), mpl::mpi_error);
      }
      pool.waitall();
      guard.active = false;
      pool.start(0);
      guard.active = true;
      pool.waitall();
      guard.active = false;

      THEN("the pool reports no active requests") {
        CHECK(pool.testany().first == mpl::test_result::no_active_requests);
      }
    }
  }
}

SCENARIO("representative wrapper errors preserve MPI classes", "[wrapper_injection]") {
  GIVEN("communicator-backed layout, info, and group wrappers") {
    const auto &self{mpl::environment::comm_self()};
    mpl::info info;
    mpl::group group{self};

    WHEN("MPI failures are injected into representative wrapper operations") {
      {
        injection::failure fail{injection::operation::datatype_create};
        CHECK_THROWS_AS((mpl::contiguous_layout<int>{2}), mpl::mpi_error);
      }
      {
        injection::failure fail{injection::operation::info_set};
        CHECK_THROWS_AS(info.set("key", "value"), mpl::mpi_error);
      }
      {
        injection::failure fail{injection::operation::group_size};
        CHECK_THROWS_AS(group.size(), mpl::mpi_error);
      }

      THEN("the wrappers report MPI errors and remain usable") {
        CHECK(group.size() == 1);
      }
    }
  }
}

SCENARIO("file operations route errors by MPI class", "[file_injection]") {
  GIVEN("an open temporary MPI file") {
    const auto &self{mpl::environment::comm_self()};
    const auto name{std::string{"mpl-error-injection-"} +
                    std::to_string(mpl::environment::comm_world().rank()) + ".tmp"};
    mpl::file file{self, name,
                   mpl::file::access_mode::create | mpl::file::access_mode::read_write |
                       mpl::file::access_mode::delete_on_close};

    WHEN("file-related MPI failures are injected") {
      {
        injection::failure fail{injection::operation::file_group};
        CHECK_THROWS_AS((mpl::group{file}), mpl::io_failure);
      }
      CHECK(mpl::group{file}.size() == 1);
      {
        injection::failure fail{injection::operation::file_arg};
        try {
          (void)file.size();
          FAIL("expected an MPI exception");
        } catch (const mpl::io_failure &) {
          FAIL("MPI_ERR_ARG must not route to io_failure");
        } catch (const mpl::mpi_error &error) {
          CHECK(error.error_code() == MPI_ERR_ARG);
        }
      }

      THEN("I/O-class errors are routed to io_failure") {
        injection::failure fail{injection::operation::file_io};
        CHECK_THROWS_AS(file.size(), mpl::io_failure);
      }
    }
  }
}
