# MPL - A message passing library

MPL is a message passing library written in C++17 based on the
[Message Passing Interface](http://mpi-forum.org/) (MPI) standard.  Since
the C++ API has been dropped from the MPI standard in version 3.0 it is the
aim of MPL to provide a modern C++ message passing library for high
performance computing.

MPL will neither bring all functions of the C language MPI-API to C++ nor
provide a direct mapping of the C API to some C++ functions and classes.
Its focus lies on the MPI core message passing functions, ease of use, type
safety, and elegance.  This library is most useful for developers who have at 
least some basic knowledge of the Message Passing Interface and would like to 
utilize it via a more user-friendly interface. 


## Supported features

The Message Passing Library gives currently access via a convinient C++ 
interface to the following features of the Message Passing Interface standard:

* environmental management (implicit initialization and finalization, timers, but no error handlig).
* point-to-point communication (blocking and non-blocking),
* collective communication (blocking and non-blocking),
* derived data types (happens automatically for many custom datatypes or via the `base_struct_builder` helper class and the layout classes of MPL),
* communicator- and group-management and
* process topolgies (cartesian and graph topologies),

Currently, the following MPI features are not yet suppoted by MPL:

* error handling,
* process creation and management,
* one-sided communication and
* I/O.

MPL assumes that the underlaying MPI implementation supports the 
[version 3.1](https://www.mpi-forum.org/docs/) of the Message Passing 
Interface standard.  Future versions of MPL may also employ features of 
the upcoming version 4.0 or later MPI versions.


## Installation

MPL is built on MPI.  An MPI implementation needs to be installed as a 
prerequisite, e.g., [Open MPI](https://www.open-mpi.org/) or
[MPICH](https://www.mpich.org/).  As MPL is a header-only library, 
it suffices to download the [source](https://github.com/rabauke/mpl) 
and copy the `mpl` directory, which contains all header files to a place, 
where the compiler will find it, e.g., `/usr/local/include` on a typical 
Unix/Linux system.  

For convinance and better integration into various IDEs, MPL also comes 
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
The option `-DCMAKE_INSTALL_PREFIX:PATH` specifies the installaztion path. 
Cmake can also be utilized to install the MPL header files.  Just call
CMake a second time and specify the `--install` option now, e.g.,
```shell
user@host:~/mpl/build$ cmake --install .
```

A set of unit tests and a collection of exmaples that illustrate the 
usage of MPL can be complied via CMake, too, if required.  To build the
MPL unit tests add the option `-DBUILD_TESTING=ON` to the initial CMake
call.  Similarily, `-DMPL_BUILD_EXAMPLES=ON` enables building example
codes. Thus,
```shell
user@host:~/mpl/build$ cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_TESTING=ON -DMPL_BUILD_EXAMPLES=ON ..
```
enables both, building unit tests and examples.  MPL unit tests utilize
the [Boost Test](https://www.boost.org/doc/libs/1_76_0/libs/test/doc/html/index.html) 
library.  Finally, build the unit tests and/or the example code via
```shell
user@host:~/mpl/build$ cmake --build .
```
After the unit test have been build successfully, they can be run 
conviniently by utilizing the CTest tool, i.e., via
```shell
user@host:~/mpl/build$ ctest
Test project /home/user/mpl/build
      Start  1: test_communicator
 1/19 Test  #1: test_communicator ................   Passed    0.45 sec
      Start  2: test_cart_communicator
 2/19 Test  #2: test_cart_communicator ...........   Passed    0.35 sec
      Start  3: test_graph_communicator
 3/19 Test  #3: test_graph_communicator ..........   Passed    0.36 sec
      Start  4: test_dist_graph_communicator
 4/19 Test  #4: test_dist_graph_communicator .....   Passed    0.31 sec
      Start  5: test_send_recv
 5/19 Test  #5: test_send_recv ...................   Passed    0.34 sec
      Start  6: test_isend_irecv
 6/19 Test  #6: test_isend_irecv .................   Passed    0.34 sec
      Start  7: test_psend_precv
 7/19 Test  #7: test_psend_precv .................   Passed    0.36 sec
      Start  8: test_collective
 8/19 Test  #8: test_collective ..................   Passed    0.37 sec
      Start  9: test_icollective
 9/19 Test  #9: test_icollective .................   Passed    0.31 sec
      Start 10: test_collectivev
10/19 Test #10: test_collectivev .................   Passed    0.28 sec
      Start 11: test_icollectivev
11/19 Test #11: test_icollectivev ................   Passed    0.34 sec
      Start 12: test_reduce
12/19 Test #12: test_reduce ......................   Passed    0.40 sec
      Start 13: test_ireduce
13/19 Test #13: test_ireduce .....................   Passed    0.36 sec
      Start 14: test_scan
14/19 Test #14: test_scan ........................   Passed    0.36 sec
      Start 15: test_iscan
15/19 Test #15: test_iscan .......................   Passed    0.34 sec
      Start 16: test_exscan
16/19 Test #16: test_exscan ......................   Passed    0.44 sec
      Start 17: test_iexscan
17/19 Test #17: test_iexscan .....................   Passed    0.30 sec
      Start 18: test_reduce_scatter
18/19 Test #18: test_reduce_scatter ..............   Passed    0.36 sec
      Start 19: test_ireduce_scatter
19/19 Test #19: test_ireduce_scatter .............   Passed    0.36 sec

100% tests passed, 0 tests failed out of 19

Total Test time (real) =   6.75 sec
```
or via your IDE if it features support for CTest.

Usually, CMake will find the required MPI installation as well as the 
Boost Test library automatically.  Depending on the local setu, however, 
CMake may need some hints to find these dependencies.  See the CMake 
documantation on 
[FindMPI](https://cmake.org/cmake/help/git-master/module/FindMPI.html#variables-for-locating-mpi) 
and 
[FindBoost](https://cmake.org/cmake/help/git-master/module/FindBoost.html?highlight=boost#hints)
for further details.


## Hello World

MPL is built on top of the Message Passing Interface (MPI) standard.  Therefore, 
MPL shares many concepts known from the MPI standard, e.g., the concept of a
communicator.  Communicators manage the message exchange between different processes, 
i.e., messages are sent and received with the help of a communicator.  

The MPL envirionment provides a global default communicator `comm_world`, which will 
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

the book

  * [Parallel Programming for Science and Engineering](https://pages.tacc.utexas.edu/~eijkhout/pdf/pcse/EijkhoutParComp.pdf) by Victor Eijkhout

and the files in the `examples` directory of the source package.
