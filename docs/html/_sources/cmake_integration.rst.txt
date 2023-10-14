CMake integration
=================

MPL provides supporting files for CMake integration. These are also installed during the installation step.  CMake integration is realized via the ``mpl`` CMake package, which provides the library target ``mpl::mpl``.  The following example ``CMakeLists.txt`` file illustrates the usage of the ``mpl`` CMake package for creating an MPL application. First, the ``mpl`` package is loaded via the ``find_package`` function. Then, all targets with MPL dependency must be linked against ``mpl::mpl``.  In this way, CMake adds all necessary compiler flags and linker flags that are required for building an MPL application.

.. code-block:: CMake

	# MPI CMake module available since version 3.10
	cmake_minimum_required(VERSION 3.10)

	project(hello_mpl)

	# project requires c++17 to build
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)

	# find the MPL library and its dependencies, e.g., an MPI library
	find_package(mpl REQUIRED)

	# create executable and link against mpl and its dependencies
	add_executable(hello_world hello_world.cc)
	target_link_libraries(hello_world PRIVATE mpl::mpl)

When using ``find_package``, CMake searches in a set of platform-dependent standard directories for the requested CMake package.  CMake may fail to find the MPL CMake package when MPL was installed in a custom directory.  If MPL was installed in a custom directory, add the installation directory (given via ``CMAKE_INSTALL_PREFIX`` during MPL configuration see :ref:`Installation`) to the ``CMAKE_PREFIX_PATH`` variable during the configuration of the MPL application, e.g.:

.. code:: shell

   user@host:~/hello_mpl/build$ cmake -DCMAKE_PREFIX_PATH:PATH=/path/to/mpl ..
