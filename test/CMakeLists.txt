cmake_minimum_required(VERSION 3.0)

project(mpl_tests)

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED) 
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

include_directories("../")

add_executable(send_recv send_recv.cc)
add_executable(isend_irecv isend_irecv.cc)

add_custom_target(run
  COMMAND mpiexec -n 2 ./send_recv
  DEPENDS send_recv
  COMMAND mpiexec -n 2 ./isend_irecv
  DEPENDS isend_irecv
  )
