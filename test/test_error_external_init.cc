#include <mpi.h>
#include <cstdio>
#include <exception>
#include <mpl/mpl.hpp>

namespace {

  void custom_handler(MPI_Comm *, int *, ...) {
  }

  void mpi_check(int result) {
    if (result != MPI_SUCCESS) {
      std::fprintf(stderr, "MPI fixture failed: %d\n", result);
      MPI_Abort(MPI_COMM_WORLD, result);
    }
  }

  class external_environment {
  public:
    external_environment(int *argc, char ***argv) {
      mpi_check(MPI_Init(argc, argv));
    }

    ~external_environment() {
      MPI_Finalize();
    }
  };

  bool has_handler(MPI_Comm comm, MPI_Errhandler expected) {
    MPI_Errhandler actual;
    mpi_check(MPI_Comm_get_errhandler(comm, &actual));
    const bool equal{actual == expected};
    mpi_check(MPI_Errhandler_free(&actual));
    return equal;
  }

}  // namespace

int main(int argc, char **argv) {
  static external_environment environment{&argc, &argv};
  int rank{0}, size{0};
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size));
  if (size != 2) {
    std::fprintf(stderr, "This test requires exactly two ranks\n");
    return 1;
  }

  MPI_Errhandler custom;
  mpi_check(MPI_Comm_create_errhandler(custom_handler, &custom));
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, custom));
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_SELF, custom));
  bool ok{true};

  try {
    {
      mpl::info info;
      info.set("mpl_test", "value");
      ok = ok && info.value("mpl_test") == "value";
      mpl::contiguous_layout<int> layout{2};
      (void)layout;
    }
    ok = has_handler(MPI_COMM_WORLD, custom) && ok;
    ok = has_handler(MPI_COMM_SELF, custom) && ok;

    MPI_Comm intra, local, inter;
    mpi_check(MPI_Comm_dup(MPI_COMM_WORLD, &intra));
    mpi_check(MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &local));
    mpi_check(MPI_Intercomm_create(local, 0, MPI_COMM_WORLD, 1 - rank, 42, &inter));
    mpi_check(MPI_Comm_set_errhandler(intra, custom));
    mpi_check(MPI_Comm_set_errhandler(inter, custom));
    {
      mpl::mpi_communicator borrowed{intra};
      mpl::mpi_inter_communicator borrowed_inter{inter};
      ok = has_handler(intra, MPI_ERRORS_RETURN) && ok;
      ok = has_handler(inter, MPI_ERRORS_RETURN) && ok;
      ok = borrowed.rank() == rank && ok;
      ok = borrowed_inter.rank() == 0 && ok;
    }
    int raw_rank{-1}, remote_size{0};
    mpi_check(MPI_Comm_rank(intra, &raw_rank));
    mpi_check(MPI_Comm_remote_size(inter, &remote_size));
    ok = raw_rank == rank && remote_size == 1 && ok;
    ok = has_handler(intra, MPI_ERRORS_RETURN) && ok;
    ok = has_handler(inter, MPI_ERRORS_RETURN) && ok;
    mpi_check(MPI_Comm_free(&inter));
    mpi_check(MPI_Comm_free(&local));
    mpi_check(MPI_Comm_free(&intra));

    ok = has_handler(MPI_COMM_WORLD, custom) && ok;
    ok = has_handler(MPI_COMM_SELF, custom) && ok;
    const auto &world{mpl::environment::comm_world()};
    ok = world.rank() == rank && ok;
    ok = has_handler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) && ok;
    ok = has_handler(MPI_COMM_SELF, MPI_ERRORS_RETURN) && ok;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rank %d: unexpected exception: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int local_ok{ok ? 1 : 0}, all_ok{0};
  mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (!ok)
    std::fprintf(stderr, "rank %d: handler or ownership check failed\n", rank);
  mpi_check(MPI_Errhandler_free(&custom));
  return all_ok ? 0 : 1;
}