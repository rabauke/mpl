cmake_minimum_required(VERSION 3.0)

project(mpl_tests)

set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMPL_DEBUG")

include_directories("../")

add_executable(test_communicator test_communicator.cc)
add_executable(test_cart_communicator test_cart_communicator.cc)
set_property(TARGET test_cart_communicator PROPERTY CXX_STANDARD 17)
add_executable(test_send_recv test_send_recv.cc)
set_property(TARGET test_send_recv PROPERTY CXX_STANDARD 17)
add_executable(test_isend_irecv test_isend_irecv.cc)
set_property(TARGET test_isend_irecv PROPERTY CXX_STANDARD 17)
add_executable(test_psend_precv test_psend_precv.cc)
set_property(TARGET test_psend_precv PROPERTY CXX_STANDARD 17)
add_executable(test_collective test_collective.cc)
add_executable(test_icollective test_icollective.cc)
add_executable(test_collectivev test_collectivev.cc)
add_executable(test_icollectivev test_icollectivev.cc)
add_executable(test_reduce test_reduce.cc)
add_executable(test_ireduce test_ireduce.cc)
add_executable(test_scan test_scan.cc)
add_executable(test_iscan test_iscan.cc)
add_executable(test_exscan test_exscan.cc)
add_executable(test_iexscan test_iexscan.cc)
add_executable(test_reduce_scatter test_reduce_scatter.cc)
add_executable(test_ireduce_scatter test_ireduce_scatter.cc)

add_custom_target(run_tests
        COMMAND mpiexec -n 1 ./test_communicator
        DEPENDS test_communicator
        COMMAND mpiexec -n 4 ./test_communicator
        DEPENDS test_communicator
        COMMAND mpiexec -n 16 ./test_cart_communicator
        DEPENDS test_cart_communicator
        COMMAND mpiexec -n 8 ./test_send_recv
        DEPENDS test_send_recv
        COMMAND mpiexec -n 8 ./test_isend_irecv
        DEPENDS test_isend_irecv
        COMMAND mpiexec -n 8 ./test_psend_precv
        DEPENDS test_psend_precv
        COMMAND mpiexec -n 8 ./test_collective
        DEPENDS test_collective
        COMMAND mpiexec -n 8 ./test_icollective
        DEPENDS test_icollective
        COMMAND mpiexec -n 8 ./test_collectivev
        DEPENDS test_collectivev
        COMMAND mpiexec -n 8 ./test_icollectivev
        DEPENDS test_icollectivev
        COMMAND mpiexec -n 8 ./test_reduce
        DEPENDS test_reduce
        COMMAND mpiexec -n 8 ./test_ireduce
        DEPENDS test_ireduce
        COMMAND mpiexec -n 8 ./test_scan
        DEPENDS test_scan
        COMMAND mpiexec -n 8 ./test_iscan
        DEPENDS test_iscan
        COMMAND mpiexec -n 8 ./test_exscan
        DEPENDS test_exscan
        COMMAND mpiexec -n 8 ./test_iexscan
        DEPENDS test_iexscan
        COMMAND mpiexec -n 8 ./test_reduce_scatter
        DEPENDS test_reduce_scatter
        COMMAND mpiexec -n 8 ./test_ireduce_scatter
        DEPENDS test_ireduce_scatter
        )
