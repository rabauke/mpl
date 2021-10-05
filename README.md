# MPL - A message passing library

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
(Boost.MPI)[https://www.boost.org/doc/libs/1_77_0/doc/html/mpi.html],
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
* communicator- and group-management and
* process topologies (cartesian and graph topologies),

Currently, the following MPI features are not yet supported by MPL:

* inter-communicators (planed for v0.2)
* error handling,
* process creation and management,
* one-sided communication and
* I/O.

Although MPL covers a subset of the MPI functionality only, it has 
probably the largest MPI-feature coverage among all alternative C++ 
interfaces to MPI.


## Installation

MPL is built on MPI.  An MPI implementation needs to be installed as a 
prerequisite, e.g., [Open MPI](https://www.open-mpi.org/) or
[MPICH](https://www.mpich.org/).  As MPL is a header-only library, 
it suffices to download the [source](https://github.com/rabauke/mpl) 
and copy the `mpl` directory, which contains all header files to a place, 
where the compiler will find it, e.g., `/usr/local/include` on a typical 
Unix/Linux system.  

For convenience and better integration into various IDEs, MPL also comes 
with CMake support.  To install MPL via CMake get the sources and create
a new `build` folder in the MPL source directory, e.g.,
```shell
user@host:~/mpl$ mkdir build
user@host:~/mpl$ cd build
```
Then, call the CMake tool to detect all dependencies and to generate the
project configuration for your build system or IDE, e.g.
```shell
user@host:~/mpl/build$ cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..
```
The option `-DCMAKE_INSTALL_PREFIX:PATH` specifies the installation path. 
Cmake can also be utilized to install the MPL header files.  Just call
CMake a second time and specify the `--install` option now, e.g.,
```shell
user@host:~/mpl/build$ cmake --install .
```

A set of unit tests and a collection of examples that illustrate the 
usage of MPL can be complied via CMake, too, if required.  To build the
MPL unit tests add the option `-DBUILD_TESTING=ON` to the initial CMake
call.  Similarly, `-DMPL_BUILD_EXAMPLES=ON` enables building example
codes. Thus,
```shell
user@host:~/mpl/build$ cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_TESTING=ON -DMPL_BUILD_EXAMPLES=ON ..
```
enables both, building unit tests and examples.  MPL unit tests utilize
the 
[Boost.Test](https://www.boost.org/doc/libs/1_77_0/libs/test/doc/html/index.html) 
framework.  Finally, build the unit tests and/or the example code via
```shell
user@host:~/mpl/build$ cmake --build .
```
After the unit test have been build successfully, they can be run 
conveniently by utilizing the CTest tool, i.e., via
```shell
user@host:~/mpl/build$ ctest
Test project /home/user/mpl/build
      Start  1: test_communicator
 1/27 Test  #1: test_communicator ........................   Passed    0.19 sec
      Start  2: test_cartesian_communicator
 2/27 Test  #2: test_cartesian_communicator ..............   Passed    0.11 sec
      Start  3: test_graph_communicator
 3/27 Test  #3: test_graph_communicator ..................   Passed    0.07 sec
      Start  4: test_dist_graph_communicator
 4/27 Test  #4: test_dist_graph_communicator .............   Passed    0.11 sec
      Start  5: test_communicator_send_recv
 5/27 Test  #5: test_communicator_send_recv ..............   Passed    0.11 sec
      Start  6: test_communicator_isend_irecv
 6/27 Test  #6: test_communicator_isend_irecv ............   Passed    0.12 sec
      Start  7: test_communicator_init_send_init_recv
 7/27 Test  #7: test_communicator_init_send_init_recv ....   Passed    0.11 sec
      Start  8: test_communicator_sendrecv
 8/27 Test  #8: test_communicator_sendrecv ...............   Passed    0.11 sec
      Start  9: test_communicator_probe
 9/27 Test  #9: test_communicator_probe ..................   Passed    0.11 sec
      Start 10: test_communicator_mprobe_mrecv
10/27 Test #10: test_communicator_mprobe_mrecv ...........   Passed    0.11 sec
      Start 11: test_communicator_barrier
11/27 Test #11: test_communicator_barrier ................   Passed    0.11 sec
      Start 12: test_communicator_bcast
12/27 Test #12: test_communicator_bcast ..................   Passed    0.10 sec
      Start 13: test_communicator_gather
13/27 Test #13: test_communicator_gather .................   Passed    0.10 sec
      Start 14: test_communicator_gatherv
14/27 Test #14: test_communicator_gatherv ................   Passed    0.06 sec
      Start 15: test_communicator_allgather
15/27 Test #15: test_communicator_allgather ..............   Passed    0.11 sec
      Start 16: test_communicator_allgatherv
16/27 Test #16: test_communicator_allgatherv .............   Passed    0.14 sec
      Start 17: test_communicator_scatter
17/27 Test #17: test_communicator_scatter ................   Passed    0.12 sec
      Start 18: test_communicator_scatterv
18/27 Test #18: test_communicator_scatterv ...............   Passed    0.12 sec
      Start 19: test_communicator_alltoall
19/27 Test #19: test_communicator_alltoall ...............   Passed    0.11 sec
      Start 20: test_communicator_alltoallv
20/27 Test #20: test_communicator_alltoallv ..............   Passed    0.15 sec
      Start 21: test_communicator_reduce
21/27 Test #21: test_communicator_reduce .................   Passed    0.13 sec
      Start 22: test_communicator_allreduce
22/27 Test #22: test_communicator_allreduce ..............   Passed    0.13 sec
      Start 23: test_communicator_reduce_scatter_block
23/27 Test #23: test_communicator_reduce_scatter_block ...   Passed    0.12 sec
      Start 24: test_communicator_reduce_scatter
24/27 Test #24: test_communicator_reduce_scatter .........   Passed    0.08 sec
      Start 25: test_communicator_scan
25/27 Test #25: test_communicator_scan ...................   Passed    0.05 sec
      Start 26: test_communicator_exscan
26/27 Test #26: test_communicator_exscan .................   Passed    0.05 sec
      Start 27: test_displacements
27/27 Test #27: test_displacements .......................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 27

Total Test time (real) =   2.86 sec
```
or via your IDE if it features support for CTest.

Alternatively, MPL may be installed via the 
[Spack](https://spack.readthedocs.io/) package manager.  This will 
install the library headers ony but not compile the unit tests and the
examples.

Usually, CMake will find the required MPI installation as well as the 
Boost Test library automatically.  Depending on the local setup, however, 
CMake may need some hints to find these dependencies.  See the CMake 
documentation on 
[FindMPI](https://cmake.org/cmake/help/git-master/module/FindMPI.html#variables-for-locating-mpi) 
and 
[FindBoost](https://cmake.org/cmake/help/git-master/module/FindBoost.html?highlight=boost#hints)
for further details.


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
[Doxygen-generated documentation](https://rabauke.github.io/mpl/html/), the blog posts

  * [MPL – A message passing library](https://www.numbercrunch.de/blog/2015/08/mpl-a-message-passing-library/),
  * [MPL – Collective communication](https://www.numbercrunch.de/blog/2015/09/mpl-collective-communication/),
  * [MPL – Data types](https://www.numbercrunch.de/blog/2015/09/mpl-data-types/),

the presentation

  * [Message Passing mit modernem C++](https://rabauke.github.io/mpl/mpl_parallel_2018.pdf) (German only),

the book

  * [Parallel Programming for Science and Engineering](https://pages.tacc.utexas.edu/~eijkhout/pdf/pcse/EijkhoutParComp.pdf) by Victor Eijkhout

and the files in the `examples` directory of the source package.
