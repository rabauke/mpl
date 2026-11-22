MPL: A message passing library
==============================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   design
   installation
   cmake_integration
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
   interoperability
   examples/index


Introduction
------------

MPL is a message passing library written in C++17 based on the `Message Passing Interface <http://mpi-forum.org/>`_ (MPI) standard. Since the C++ API has been dropped from the MPI standard in version 3.1, it is the aim of MPL to provide a modern C++ message passing library for high performance computing.

MPL will neither bring all functions of the C language MPI-API to C++ nor provide a direct mapping of the C API to some C++ functions and classes. The library's focus lies on the MPI core message passing functions, ease of use, type safety, and elegance. The aim of MPL is to provide an idiomatic C++ message passing library without introducing a significant overhead compared to utilizing MPI via its plain C-API. This library is most useful for developers who have at least some basic knowledge of the Message Passing Interface standard and would like to utilize it via a more user-friendly interface in modern C++. Unlike `Boost.MPI <https://www.boost.org/doc/libs/latest/doc/html/mpi.html>`_, MPL does not rely on an external serialization library and has a negligible run-time overhead.


Supported features
------------------

MPL requires an MPI 4.1 implementation and uses its large-count interfaces.

MPL gives currently access via a convenient C++ interface to the following features of the Message Passing Interface standard:

* environmental management (implicit initialization and finalization, timers, runtime library
  identification, and environment-derived info objects),
* point-to-point communication (blocking and non-blocking),
* collective communication (blocking and non-blocking),
* buffered sends with process-wide and communicator-scoped buffers and blocking or non-blocking
  flush operations,
* derived data types (happens automatically for many custom data types or via the ``base_struct_builder`` helper class and the layout classes of MPL),
* communicator- and group-management, including communicator info hints and info-aware
  asynchronous duplication,
* MPI-managed STL allocation and process-local reduction,
* process topologies (cartesian and graph topologies),
* inter-communicators,
* dynamic process creation,
* file i/o, and
* error reporting through MPL exceptions.

Currently, the following MPI features are not yet supported by MPL:

* one-sided communication and
* session-scoped buffer management, which remains unsupported until MPL provides a session
  abstraction.

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


Error handling
--------------

When a checked MPI call returns an error, MPL reports it as ``mpl::mpi_error`` (derived from
``mpl::error``), carrying its code, class, and message. Recognized I/O error classes become
``mpl::io_failure``, including outside file operations; other classes from file operations remain
``mpl::mpi_error``.

MPL installs ``MPI_ERRORS_RETURN`` when it initializes its environment and wraps
``MPI_COMM_WORLD`` and ``MPI_COMM_SELF``, and when it creates or wraps communicators. With external
MPI initialization, earlier operations do not initialize the MPL environment or change those
handlers; a fatal handler may abort before a call returns. Borrowed ``mpi_communicator`` and
``mpi_inter_communicator`` adapters change the original handle's handler without restoring it or
taking ownership. MPI file and objectless operations can use different handlers; exceptions are
possible only if MPI returns an error to MPL.

Error reporting does not guarantee recovery: MPI may not permit continued execution, and other
ranks may remain blocked. Cleanup and destructors are best-effort and do not throw. See
:doc:`error` for request completion details.


Building and testing
--------------------

MPL can be built and tested using standard CMake commands:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DMPL_BUILD_EXAMPLES=ON
   cmake --build build -j
   ctest --test-dir build --output-on-failure

MPICH / hwloc troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running unit tests or executables directly via ``mpirun`` or ``mpiexec`` under MPICH on
systems with ``libhwloc`` GL plugins installed (``hwloc_gl.so``), MPI initialization may hang in
remote, headless, or containerized environments if an unreachable ``$DISPLAY`` socket endpoint is
queried via ``XOpenDisplay``.

While CTest targets in MPL configure this automatically, when executing binaries directly under
MPICH, disable the hwloc OpenGL discovery component by setting:

.. code-block:: bash

   export HWLOC_COMPONENTS=-gl
   # or prepend to direct mpiexec invocations:
   HWLOC_COMPONENTS=-gl mpiexec -n 2 ./build/test/test_communicator_send_recv


Further documentation
---------------------

For further documentation see the
`documentation <https://rabauke.github.io/mpl/html/>`__, the blog posts

* `MPL – A message passing library <https://www.numbercrunch.de/blog/2015/08/mpl-a-message-passing-library/>`__,
* `MPL – Collective communication <https://www.numbercrunch.de/blog/2015/09/mpl-collective-communication/>`__,
* `MPL – Data types <https://www.numbercrunch.de/blog/2015/09/mpl-data-types/>`__,

the presentation

* `Message Passing mit modernem C++ <https://rabauke.github.io/mpl/mpl_parallel_2018.pdf>`__ (German only),

the book

* `Parallel Programming for Science and Engineering <https://theartofhpc.com/pcse.html>`__ by Victor Eijkhout, or

the workshop material

* `SMU O'Donnell Data Science and Research Computing Institute Parallel C++ Workshop <https://southernmethodistuniversity.github.io/parallel_cpp/intro.html>`__

and the files in the ``examples`` directory of the source package.


Indices and tables
------------------

* :ref:`genindex`
