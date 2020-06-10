# MPL - A message passing library

MPL is a message passing library written in C++11 based on the
[Message Passing Interface](http://mpi-forum.org/) (MPI) standard.  Since
the C++ API has been dropped from the MPI standard in version 3.0 it is the
aim of MPL to provide a modern C++ message passing library for high
performance computing.

MPL will neither bring all functions of the C language MPI-API to C++ nor
provide a direct mapping of the C API to some C++ functions and classes.
Its focus lies on the MPI core message passing functions, ease of use, type
safety, and elegance.  This library is most useful for developers who have at 
least some basic knowledge of the Message Passing Interface and would like to 
utilize it via a more user friendly interface. 


## Installation

MPL is a header-only library.  Just download the
[source](https://github.com/rabauke/mpl) and copy the `mpl` directory
containing all header files to a place, where the compiler will find
it, e.g., `/usr/local/include` on a typical Unix/Linux system.  As MPL is 
built on MPI, an MPI implementation needs to be installed, e.g.,
[Open MPI](https://www.open-mpi.org/) or
[MPICH](https://www.mpich.org/).


## Hello World

MPL is build on top of the Message Passing Interface (MPI) standard.  Therefore, 
MPL shares many concepts known from the MPI standard, e.g., the concept of a
communicator.  Communicators manage communication between processes.  
Messages are sent and received with the help of a communicator.  

The MPL envirionment provides a global default communicator `comm_world`, which will 
be used in the following Hello-World program.  The program prints out some infomation 
about each process:
* its rank, 
* the total number of processes and 
* the computer name the process is running on.

If there are two or more processes, a message is sent from process 0 to process 1, 
which is also printed.
 
```C++
#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  // each process prints a message containing the processor name, the rank
  // in communicator world and the size of communicator world
  // output may depend on MPI implementation
  std::cout << "Hello world! I am running on \"" << mpl::environment::processor_name()
            << "\". My rank is " << comm_world.rank() << " out of " << comm_world.size()
            << " processes.\n";
  // if there are two or more processes send a message from process 0 to process 1
  if (comm_world.size() >= 2) {
    if (comm_world.rank() == 0) {
      std::string message{"Hello world!"};
      comm_world.send(message, 1);  // send message to rank 1
    } else if (comm_world.rank() == 1) {
      std::string message;
      comm_world.recv(message, 0);  // receive message from rank 0
      std::cout << "got: \"" << message << "\"\n";
    }
  }
  return EXIT_SUCCESS;
}
```


## Documentation

For further documentation see the
[Doxygen-generated documentation](https://rabauke.github.io/mpl/html/), the blog posts

  * [MPL – A message passing library](https://www.numbercrunch.de/blog/2015/08/mpl-a-message-passing-library/),
  * [MPL – Collective communication](https://www.numbercrunch.de/blog/2015/09/mpl-collective-communication/),
  * [MPL – Data types](https://www.numbercrunch.de/blog/2015/09/mpl-data-types/),

the presentation

  * [Message Passing mit modernem C++](https://rabauke.github.io/mpl/mpl_parallel_2018.pdf) (German only),

and the files in the `examples` directory of the source package.
