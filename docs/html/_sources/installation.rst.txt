.. _Installation:

Installation
============

MPL is built on MPI. An MPI implementation needs to be installed as a prerequisite, e.g., `Open MPI <https://www.open-mpi.org/>`__ or `MPICH <https://www.mpich.org/>`__. MPL is a header-only library. Thus, it suffices to download the `source <https://github.com/rabauke/mpl>`__ and to copy the ``mpl`` directory, which contains all header files to a folder, where the compiler will find it, e.g., ``/usr/local/include`` on a typical Unix/Linux system.

For convenience and better integration into various IDEs, MPL also comes with CMake support. To install MPL via CMake get the sources and create a new ``build`` folder in the MPL source directory, e.g.,

.. code:: shell

   user@host:~/mpl$ mkdir build
   user@host:~/mpl$ cd build

Then, call the CMake tool to detect all dependencies and to generate the project configuration for your build system or IDE, e.g.,

.. code:: shell

   user@host:~/mpl/build$ cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..

The option ``-DCMAKE_INSTALL_PREFIX:PATH`` specifies the installation path.

Usually, CMake will find the required MPI installation as well as the Boost Test library automatically. Depending on the local setup, however, CMake may need some hints to find these dependencies. See the CMake documentation on `FindMPI <https://cmake.org/cmake/help/git-master/module/FindMPI.html#variables-for-locating-mpi>`__ and `FindBoost <https://cmake.org/cmake/help/git-master/module/FindBoost.html?highlight=boost#hints>`__ for further details.

CMake can also be utilized to install the MPL header files. Just call CMake a second time and specify the ``--install`` option now, e.g.,

.. code:: shell

   user@host:~/mpl/build$ cmake --install .

A set of unit tests and a collection of examples that illustrate the usage of MPL can be complied via CMake, too, if required. To build the MPL unit tests add the option ``-DBUILD_TESTING=ON`` to the initial CMake call. Similarly, ``-DMPL_BUILD_EXAMPLES=ON`` enables building example codes. Thus,

.. code:: shell

   user@host:~/mpl/build$ cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_TESTING=ON -DMPL_BUILD_EXAMPLES=ON ..

enables both, building unit tests and examples. MPL unit tests utilize the `Boost.Test <https://www.boost.org/doc/libs/1_77_0/libs/test/doc/html/index.html>`__ framework. Furthermore, adding ``-DMPL_BUILD_DOCUMENTATION=ON`` to the set of CMake options during configuration enables building the MPL documentation, which requires `Doxygen <https://www.doxygen.nl>`__, `Breathe <https://www.breathe-doc.org>`__ and `Sphinx <https://www.sphinx-doc.org>`__. Finally, build the unit tests, the example code and/or the documentation via

.. code:: shell

   user@host:~/mpl/build$ cmake --build .

After the unit test have been build successfully, they can be run conveniently by utilizing the CTest tool, i.e., via

.. code:: shell

   user@host:~/mpl/build$ ctest
   Test project /home/user/mpl/build
         Start  1: test_group
    1/30 Test  #1: test_group ...............................   Passed    0.44 sec
         Start  2: test_communicator
    2/30 Test  #2: test_communicator ........................   Passed    0.31 sec
         Start  3: test_cartesian_communicator
    3/30 Test  #3: test_cartesian_communicator ..............   Passed    0.38 sec
         Start  4: test_graph_communicator
    4/30 Test  #4: test_graph_communicator ..................   Passed    0.30 sec
         Start  5: test_dist_graph_communicator
    5/30 Test  #5: test_dist_graph_communicator .............   Passed    0.31 sec
         Start  6: test_communicator_send_recv
    6/30 Test  #6: test_communicator_send_recv ..............   Passed    0.36 sec
         Start  7: test_communicator_isend_irecv
    7/30 Test  #7: test_communicator_isend_irecv ............   Passed    0.30 sec
         Start  8: test_communicator_init_send_init_recv
    8/30 Test  #8: test_communicator_init_send_init_recv ....   Passed    0.34 sec
         Start  9: test_communicator_sendrecv
    9/30 Test  #9: test_communicator_sendrecv ...............   Passed    0.29 sec
         Start 10: test_communicator_probe
   10/30 Test #10: test_communicator_probe ..................   Passed    0.34 sec
         Start 11: test_communicator_mprobe_mrecv
   11/30 Test #11: test_communicator_mprobe_mrecv ...........   Passed    0.38 sec
         Start 12: test_communicator_barrier
   12/30 Test #12: test_communicator_barrier ................   Passed    0.37 sec
         Start 13: test_communicator_bcast
   13/30 Test #13: test_communicator_bcast ..................   Passed    0.30 sec
         Start 14: test_communicator_gather
   14/30 Test #14: test_communicator_gather .................   Passed    0.29 sec
         Start 15: test_communicator_gatherv
   15/30 Test #15: test_communicator_gatherv ................   Passed    0.30 sec
         Start 16: test_communicator_allgather
   16/30 Test #16: test_communicator_allgather ..............   Passed    0.40 sec
         Start 17: test_communicator_allgatherv
   17/30 Test #17: test_communicator_allgatherv .............   Passed    0.31 sec
         Start 18: test_communicator_scatter
   18/30 Test #18: test_communicator_scatter ................   Passed    0.29 sec
         Start 19: test_communicator_scatterv
   19/30 Test #19: test_communicator_scatterv ...............   Passed    0.29 sec
         Start 20: test_communicator_alltoall
   20/30 Test #20: test_communicator_alltoall ...............   Passed    0.29 sec
         Start 21: test_communicator_alltoallv
   21/30 Test #21: test_communicator_alltoallv ..............   Passed    0.36 sec
         Start 22: test_communicator_reduce
   22/30 Test #22: test_communicator_reduce .................   Passed    0.31 sec
         Start 23: test_communicator_allreduce
   23/30 Test #23: test_communicator_allreduce ..............   Passed    0.29 sec
         Start 24: test_communicator_reduce_scatter_block
   24/30 Test #24: test_communicator_reduce_scatter_block ...   Passed    0.34 sec
         Start 25: test_communicator_reduce_scatter
   25/30 Test #25: test_communicator_reduce_scatter .........   Passed    0.38 sec
         Start 26: test_communicator_scan
   26/30 Test #26: test_communicator_scan ...................   Passed    0.32 sec
         Start 27: test_communicator_exscan
   27/30 Test #27: test_communicator_exscan .................   Passed    0.29 sec
         Start 28: test_displacements
   28/30 Test #28: test_displacements .......................   Passed    0.06 sec
         Start 29: test_inter_communicator
   29/30 Test #29: test_inter_communicator ..................   Passed    0.29 sec
         Start 30: test_info
   30/30 Test #30: test_info ................................   Passed    0.29 sec

   100% tests passed, 0 tests failed out of 30

   Total Test time (real) =   9.57 sec

or via your IDE if it features support for CTest.

Alternatively, MPL may be installed via the `Spack <https://spack.readthedocs.io/>`__ package manager. This will install the library headers ony but not compile the unit tests and the examples.
