#include <cstdlib>
#include <exception>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    // send two integers to ourselves, but receive only one: this is an MPI error
    const int send_data[2]{1, 2};
    int recv_data;
    auto send_request{comm_world.isend(send_data, mpl::contiguous_layout<int>(2), 0)};
    // An exception_ptr is initially empty and can hold an exception for later rethrow.
    std::exception_ptr receive_error;
    try {
      comm_world.recv(recv_data, 0);
    } catch (...) {
      // Preserve the active exception so the nonblocking send can finish first.
      receive_error = std::current_exception();
    }
    send_request.wait();
    try {
      // Rethrow the original exception for the handlers below, if one was captured.
      if (receive_error)
        std::rethrow_exception(receive_error);
    } catch (const mpl::mpi_error &e) {
      // MPI errors are reported as exceptions carrying MPI error code and class
      std::cout << "caught MPI error: " << e.what() << " (class " << e.error_class() << ")\n";
    } catch (const mpl::error &e) {
      // errors detected by MPL itself have no MPI error code
      std::cout << "caught MPL error: " << e.what() << '\n';
    }
  }
  return EXIT_SUCCESS;
}
