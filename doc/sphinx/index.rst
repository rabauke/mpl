MPL: A message passing library
==============================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   installation
   data_types
   environmental_management
   group
   tags
   communicator
   layouts
   reduction_operations
   file
   info
   constants
   auxiliary
   error
   grid
   examples/index


Introduction
------------

MPL is a message passing library written in C++17 based on the `Message Passing Interface <http://mpi-forum.org/>`_ (MPI) standard. Since the C++ API has been dropped from the MPI standard in version 3.1, it is the aim of MPL to provide a modern C++ message passing library for high performance computing.

MPL will neither bring all functions of the C language MPI-API to C++ nor provide a direct mapping of the C API to some C++ functions and classes. The library's focus lies on the MPI core message passing functions, ease of use, type safety, and elegance. The aim of MPL is to provide an idiomatic C++ message passing library without introducing a significant overhead compared to utilizing MPI via its plain C-API. This library is most useful for developers who have at least some basic knowledge of the Message Passing Interface standard and would like to utilize it via a more user-friendly interface in modern C++. Unlike `Boost.MPI <https://www.boost.org/doc/libs/1_77_0/doc/html/mpi.html>`_, MPL does not rely on an external serialization library and has a negligible run-time overhead.


Supported features
------------------

MPL assumes that the underlying MPI implementation supports the `version 3.1 <https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf>`_ of the Message Passing Interface standard. Future versions of MPL may also employ features of the new `version 4.0 <https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf>`_ or later MPI versions.

MPL gives currently access via a convenient C++ interface to the following features of the Message Passing Interface standard:

* environmental management (implicit initialization and finalization, timers, but no error handling).
* point-to-point communication (blocking and non-blocking),
* collective communication (blocking and non-blocking),
* derived data types (happens automatically for many custom data types or via the ``base_struct_builder`` helper class and the layout classes of MPL),
* communicator- and group-management and
* process topologies (cartesian and graph topologies),
* inter-communicators,
* dynamic process creation and
* file I/O.

Currently, the following MPI features are not yet supported by MPL:

* error handling and
* one-sided communication.

Although MPL covers a subset of the MPI functionality only, it has probably the largest MPI-feature coverage among all alternative C++ interfaces to MPI.


Hello parallel world
--------------------

MPL is built on top of the Message Passing Interface (MPI) standard. Therefore, MPL shares many concepts known from the MPI standard, e.g., the concept of a communicator. Communicators manage the message exchange between different processes, i.e., messages are sent and received with the help of a communicator.

The MPL environment provides a global default communicator ``comm_world``, which will be used in the following Hello-World program. The program prints out some information about each process:

* its rank,
* the total number of processes and
* the computer’s name the process is running on.

If there are two or more processes, a message is sent from process 0 to process 1, which is also printed.

.. literalinclude:: ../../examples/hello_world.cc
   :language: c++


Further documentation
---------------------

For further documentation see the blog posts

*  `MPL – A message passinglibrary <https://www.numbercrunch.de/blog/2015/08/mpl-a-message-passing-library/>`__,
*  `MPL – Collective communication <https://www.numbercrunch.de/blog/2015/09/mpl-collective-communication/>`__,
*  `MPL – Data types <https://www.numbercrunch.de/blog/2015/09/mpl-data-types/>`__,

the presentation

*  `Message Passing mit modernem C++ <https://rabauke.github.io/mpl/mpl_parallel_2018.pdf>`__ (German only),

the book

*  `Parallel Programming for Science and Engineering <https://web.corral.tacc.utexas.edu/CompEdu/pdf/pcse/EijkhoutParallelProgramming.pdf>`__ by Victor Eijkhout

and the files in the ``examples`` directory of the source package.


Indices and tables
------------------

* :ref:`genindex`
