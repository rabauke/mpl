# MPL - A message passing library

MPL is a message passing library written in C++11 based on the 
[Message Passing Interface](http://mpi-forum.org/) (MPI) standard.  Since 
the C++ API has been dropped from the MPI standard in version 3.0 it is the 
aim of MPL is to provide a modern C++ message passing library for high 
performance computing.

MPL will neither bring all functions of the C language MPI-API to C++ nor 
provide a direct mapping of the C API to some C++ functions and classes. 
Its focus is on the MPI core functions, ease of use, type safety, and 
elegance.


## Installation

MPL is a header-only library.  Just copy the `mpl` directory containing all 
header files to a place, where the compiler will find it, e.g., 
`/usr/local/include`.  As MPL is built on MPI, an MPI implementation needs to 
be installed, e.g., [Open MPI](https://www.open-mpi.org/) or 
[MPICH](https://www.mpich.org/).


## Documentation

For documentation see the
[Doxygen-generated documentation](https://rabauke.github.io/mpl/html/) and
the files in the `examples` directory of the source.
