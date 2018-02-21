cmake_minimum_required(VERSION 3.0)

project(mpl_tests)

set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED) 
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMPL_DEBUG")

include_directories("../")

add_executable(test_send_recv test_send_recv.cc)
add_executable(test_isend_irecv test_isend_irecv.cc)
add_executable(test_collective test_collective.cc)

add_custom_target(run
        COMMAND mpiexec -n 2 ./send_recv
        DEPENDS send_recv
        COMMAND mpiexec -n 2 ./isend_irecv
        DEPENDS isend_irecv
        COMMAND mpiexec -n 8 ./collective
        DEPENDS collective
  )
