// Checks that mpl::mpi_error can be constructed before MPI_Init and after MPI_Finalize.
// This program intentionally does not use mpl::environment and Catch2, because the
// environment would initialize MPI at the first use.
#include <cstdlib>
#include <iostream>
#include <string>
#include <mpl/error.hpp>


namespace {

  int failures{0};

  void expect(const bool condition, const char *description) {
    if (not condition) {
      std::cerr << "FAILED: " << description << '\n';
      ++failures;
    }
  }

  void check_mpi_error(const char *phase) {
    int expected_class{0};
    char message[MPI_MAX_ERROR_STRING + 1]{};
    int length{0};
    expect(MPI_Error_class(MPI_ERR_TRUNCATE, &expected_class) == MPI_SUCCESS, phase);
    expect(MPI_Error_string(MPI_ERR_TRUNCATE, message, &length) == MPI_SUCCESS, phase);
    const mpl::mpi_error e{MPI_ERR_TRUNCATE};
    expect(std::string{e.what()} == std::string(message, length), phase);
    expect(e.error_class() == expected_class, phase);
    expect(e.error_code() == MPI_ERR_TRUNCATE, phase);

    bool caught{false};
    try {
      mpl::detail::check(MPI_ERR_TRUNCATE);
    } catch (const mpl::mpi_error &ex) {
      caught = ex.error_class() == MPI_ERR_TRUNCATE;
    }
    expect(caught, phase);
  }

}  // namespace


int main() {
  check_mpi_error("before MPI_Init");
  MPI_Init(nullptr, nullptr);
  check_mpi_error("while MPI is initialized");
  MPI_Finalize();
  check_mpi_error("after MPI_Finalize");
  if (failures == 0)
    std::cout << "all checks passed\n";
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
