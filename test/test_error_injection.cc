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

TEST_CASE("error inquiries preserve codes and use nonrecursive fallbacks", "[inquiry]") {
  (void)mpl::environment::comm_self();
  {
    injection::failure fail{injection::operation::error_class};
    const mpl::mpi_error error{MPI_ERR_TRUNCATE};
    CHECK(error.error_code() == MPI_ERR_TRUNCATE);
    CHECK(error.error_class() == MPI_ERR_OTHER);
  }
  for (const auto op :
       {injection::operation::error_string, injection::operation::error_string_bad_length}) {
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

TEST_CASE("user-defined error codes retain their registered class", "[inquiry]") {
  (void)mpl::environment::comm_self();
  int error_class{0};
  int code{0};
  REQUIRE(MPI_Add_error_class(&error_class) == MPI_SUCCESS);
  REQUIRE(MPI_Add_error_code(error_class, &code) == MPI_SUCCESS);
  REQUIRE(code != error_class);
  REQUIRE(MPI_Add_error_string(code, "custom diagnostic") == MPI_SUCCESS);
  int expected_class{0};
  REQUIRE(MPI_Error_class(code, &expected_class) == MPI_SUCCESS);
  const mpl::mpi_error error{code};
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

TEST_CASE("communicator assignment exception specifications", "[assignment_traits]") {
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

TEST_CASE("communicator copy assignment preserves ownership on failure", "[assignment]") {
  const auto &self{mpl::environment::comm_self()};
  mpl::communicator source{self};
  mpl::communicator destination{self};
  assignment_contract(destination, source);
}

TEST_CASE("successful replacement is released exactly once", "[assignment]") {
  const auto &self{mpl::environment::comm_self()};
  mpl::communicator source{self};
  {
    mpl::communicator destination{self};
    injection::reset();
    destination = source;
    CHECK(injection::events().acquired == 1);
    CHECK(injection::events().released == 0);
  }
  const auto events{injection::events()};
  CHECK(events.acquired == 1);
  CHECK(events.released == 1);
  CHECK(events.duplicate_releases == 0);
}

TEST_CASE("cartesian copy assignment preserves topology on failure", "[assignment]") {
  const auto &self{mpl::environment::comm_self()};
  const mpl::cartesian_communicator::dimensions dims{mpl::cartesian_communicator::periodic};
  mpl::cartesian_communicator source{self, mpl::dims_create(1, dims)};
  mpl::cartesian_communicator destination{self, mpl::dims_create(1, dims)};
  assignment_contract(destination, source);
  CHECK(destination.dimensionality() == 1);
}

TEST_CASE("intercommunicator copy assignment preserves ownership on failure", "[assignment]") {
  const auto &world{mpl::environment::comm_world()};
  REQUIRE(world.size() == 2);
  mpl::communicator local{mpl::communicator::split, world, world.rank()};
  mpl::inter_communicator source{local, 0, world, 1 - world.rank()};
  mpl::inter_communicator destination{source};
  assignment_contract(destination, source);
  CHECK(destination.remote_size() == 1);
}

TEST_CASE("move and self assignment retain valid ownership", "[move]") {
  const auto &self{mpl::environment::comm_self()};
  mpl::communicator source{self};
  mpl::communicator destination{self};
  const MPI_Comm original{source.native_handle()};
  injection::reset();
  auto &alias{destination};
  destination = std::move(alias);
  CHECK(destination.is_valid());
  destination = std::move(source);
  CHECK(destination.native_handle() == original);
  CHECK_FALSE(source.is_valid());
  CHECK(injection::events().acquired == 0);
}

TEST_CASE("failed owned adoption releases the newly created communicator", "[adoption]") {
  const auto &self{mpl::environment::comm_self()};
  injection::reset();
  {
    injection::failure fail{injection::operation::set_handler};
    CHECK_THROWS_AS((mpl::communicator{mpl::communicator::split, self, 0}), mpl::mpi_error);
  }
  const auto events{injection::events()};
  CHECK(events.acquired == 1);
  CHECK(events.released == 1);
  CHECK(events.duplicate_releases == 0);
}

TEST_CASE("failed borrowed adoption does not free the external communicator", "[adoption]") {
  (void)mpl::environment::comm_self();
  MPI_Comm raw{MPI_COMM_NULL};
  REQUIRE(PMPI_Comm_dup(MPI_COMM_SELF, &raw) == MPI_SUCCESS);
  injection::reset();
  injection::track(raw);
  {
    injection::failure fail{injection::operation::set_handler};
    CHECK_THROWS_AS((mpl::mpi_communicator{raw}), mpl::mpi_error);
  }
  const auto events{injection::events()};
  CHECK(events.released == 0);
  if (events.released == 0) {
    int size{0};
    CHECK(PMPI_Comm_size(raw, &size) == MPI_SUCCESS);
    CHECK(size == 1);
    CHECK(PMPI_Comm_free(&raw) == MPI_SUCCESS);
  }
}

TEST_CASE("struct datatype construction rolls back and can be retried", "[struct_datatype]") {
  (void)mpl::environment::comm_self();
  using builder = mpl::struct_builder<std::pair<int, double>>;
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

TEST_CASE("layout adoption releases datatypes on failure and success", "[layout_adoption]") {
  (void)mpl::environment::comm_self();
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
  events = injection::datatype_events();
  CHECK(events.acquired == 1);
  CHECK(events.released == 1);
  CHECK(events.duplicate_releases == 0);
}

TEST_CASE("buffer attachment owns allocation only after success", "[buffer_attachment]") {
  (void)mpl::environment::comm_self();
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
  events = injection::buffer_events();
  CHECK(events.acquired == 1);
  CHECK(events.released == 1);
  CHECK(events.duplicate_releases == 1);
  injection::reset();
}

TEST_CASE("communicator buffer attachment transfers ownership only after success",
          "[buffer_attachment]") {
  const auto &self{mpl::environment::comm_self()};
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

  void *detached_buffer{nullptr};
  MPI_Count detached_size{0};
  CHECK(PMPI_Comm_detach_buffer_c(self.native_handle(), &detached_buffer, &detached_size) ==
        MPI_SUCCESS);
  CHECK(detached_buffer == buffer);
  ::operator delete(detached_buffer);
  events = injection::buffer_events();
  CHECK(events.released == 1);
  CHECK(events.duplicate_releases == 1);
  injection::reset();
}

TEST_CASE("cancellation errors are reported without abandoning requests",
          "[cancel_injection]") {
  const auto &self{mpl::environment::comm_self()};
  int data[3]{};
  auto request{self.irecv(data[0], 0)};
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
  CHECK(calls == 1);
}

TEST_CASE("persistent request restarts after an injected cancellation failure",
          "[cancel_injection][request]") {
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
  CHECK(pool.testany().first == mpl::test_result::no_active_requests);
}

TEST_CASE("representative wrapper errors preserve MPI classes", "[wrapper_injection]") {
  const auto &self{mpl::environment::comm_self()};
  {
    injection::failure fail{injection::operation::datatype_create};
    CHECK_THROWS_AS((mpl::contiguous_layout<int>{2}), mpl::mpi_error);
  }
  mpl::info info;
  {
    injection::failure fail{injection::operation::info_set};
    CHECK_THROWS_AS(info.set("key", "value"), mpl::mpi_error);
  }
  mpl::group group{self};
  {
    injection::failure fail{injection::operation::group_size};
    CHECK_THROWS_AS(group.size(), mpl::mpi_error);
  }
  CHECK(group.size() == 1);
}

TEST_CASE("file operations route errors by MPI class", "[file_injection]") {
  const auto &self{mpl::environment::comm_self()};
  const auto name{std::string{"mpl-error-injection-"} +
                  std::to_string(mpl::environment::comm_world().rank()) + ".tmp"};
  mpl::file file{self, name,
                 mpl::file::access_mode::create | mpl::file::access_mode::read_write |
                     mpl::file::access_mode::delete_on_close};
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
  {
    injection::failure fail{injection::operation::file_io};
    CHECK_THROWS_AS(file.size(), mpl::io_failure);
  }
}
