# MPL - A message passing library

[![build-with-openmpi](https://github.com/rabauke/mpl/actions/workflows/build-with-openmpi.yml/badge.svg)](https://github.com/rabauke/mpl/actions/workflows/build-with-openmpi.yml)
[![build-with-mpich](https://github.com/rabauke/mpl/actions/workflows/build-with-mpich.yml/badge.svg)](https://github.com/rabauke/mpl/actions/workflows/build-with-mpich.yml)
[![build-with-IntelMPI](https://github.com/rabauke/mpl/actions/workflows/build-with-IntelMPI.yml/badge.svg)](https://github.com/rabauke/mpl/actions/workflows/build-with-IntelMPI.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

MPL is a message passing library written in C++17 based on the
[Message Passing Interface](http://mpi-forum.org/) (MPI) standard. 
Since the C++ API has been dropped from the MPI standard in version 
3.1, it is the aim of MPL to provide a modern C++ message passing 
library for high performance computing.

MPL will neither bring all functions of the C language MPI-API to C++
nor provide a direct mapping of the C API to some C++ functions and
classes. The library's focus lies on the MPI core message passing
functions, ease of use, type safety, and elegance. The aim of MPL is to
provide an idiomatic C++ message passing library without introducing a
significant overhead compared to utilizing MPI via its plain C-API.
This library is most useful for developers who have at least some basic
knowledge of the Message Passing Interface standard and would like to
utilize it via a more user-friendly interface in modern C++. Unlike
[Boost.MPI](https://www.boost.org/doc/libs/1_77_0/doc/html/mpi.html),
MPL does not rely on an external serialization library and has a
negligible run-time overhead.


## Supported features

MPL assumes that the underlying MPI implementation supports the 
[version 3.1](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf) 
of the Message Passing Interface standard.  Future versions of MPL 
may also employ features of the new 
[version 4.0](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf) 
or later MPI versions.  

MPL gives currently access via a convenient C++ interface to the 
following features of the Message Passing Interface standard:

* environmental management (implicit initialization and finalization, timers, but no error handling).
* point-to-point communication (blocking and non-blocking),
* collective communication (blocking and non-blocking),
* derived data types (happens automatically for many custom data types or via the `base_struct_builder` helper class and the layout classes of MPL),
* communicator- and group-management,
* process topologies (cartesian and graph topologies),
* inter-communicators,
* dynamic process creation and
* file i/o.

Currently, the following MPI features are not yet supported by MPL:

* error handling and
* one-sided communication.

Although MPL covers a subset of the MPI functionality only, it has 
probably the largest MPI-feature coverage among all alternative C++ 
interfaces to MPI.


## Hello parallel world

MPL is built on top of the Message Passing Interface (MPI) standard.  Therefore, 
MPL shares many concepts known from the MPI standard, e.g., the concept of a
communicator.  Communicators manage the message exchange between different processes, 
i.e., messages are sent and received with the help of a communicator.  

The MPL environment provides a global default communicator `comm_world`, which will 
be used in the following Hello-World program.  The program prints out some information 
about each process:
* its rank, 
* the total number of processes and 
* the computer's name the process is running on.

If there are two or more processes, a message is sent from process 0 to process 1, 
which is also printed.
 
```C++
#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // each process prints a message containing the processor name, the rank
  // in communicator world and the size of communicator world
  // output may depend on the underlying MPI implementation
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
[documentation](https://rabauke.github.io/mpl/html/), the blog posts

  * [MPL – A message passing library](https://www.numbercrunch.de/blog/2015/08/mpl-a-message-passing-library/),
  * [MPL – Collective communication](https://www.numbercrunch.de/blog/2015/09/mpl-collective-communication/),
  * [MPL – Data types](https://www.numbercrunch.de/blog/2015/09/mpl-data-types/),

the presentation

  * [Message Passing mit modernem C++](https://rabauke.github.io/mpl/mpl_parallel_2018.pdf) (German only),

the book

  * [Parallel Programming for Science and Engineering](https://web.corral.tacc.utexas.edu/CompEdu/pdf/pcse/EijkhoutParallelProgramming.pdf) by Victor Eijkhout, or

the workshop material

  * [SMU O'Donnell Data Science and Research Computing Institute Parallel C++ Workshop](https://southernmethodistuniversity.github.io/parallel_cpp/intro.html)

and the files in the `examples` directory of the source package.
