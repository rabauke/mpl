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
add_executable(test_graph_communicator test_graph_communicator.cc)
add_executable(test_dist_graph_communicator test_dist_graph_communicator.cc)
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
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_communicator ${MPIEXEC_POSTFLAGS}
        DEPENDS test_communicator
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_cart_communicator ${MPIEXEC_POSTFLAGS}
        DEPENDS test_cart_communicator
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_graph_communicator ${MPIEXEC_POSTFLAGS}
        DEPENDS test_graph_communicator
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_dist_graph_communicator ${MPIEXEC_POSTFLAGS}
        DEPENDS test_dist_graph_communicator
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_send_recv ${MPIEXEC_POSTFLAGS}
        DEPENDS test_send_recv
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_isend_irecv ${MPIEXEC_POSTFLAGS}
        DEPENDS test_isend_irecv
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_psend_precv ${MPIEXEC_POSTFLAGS}
        DEPENDS test_psend_precv
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_collective ${MPIEXEC_POSTFLAGS}
        DEPENDS test_collective
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_icollective ${MPIEXEC_POSTFLAGS}
        DEPENDS test_icollective
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_collectivev ${MPIEXEC_POSTFLAGS}
        DEPENDS test_collectivev
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_icollectivev ${MPIEXEC_POSTFLAGS}
        DEPENDS test_icollectivev
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_reduce ${MPIEXEC_POSTFLAGS}
        DEPENDS test_reduce
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_ireduce ${MPIEXEC_POSTFLAGS}
        DEPENDS test_ireduce
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_scan ${MPIEXEC_POSTFLAGS}
        DEPENDS test_scan
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_iscan ${MPIEXEC_POSTFLAGS}
        DEPENDS test_iscan
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_exscan ${MPIEXEC_POSTFLAGS}
        DEPENDS test_exscan
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_iexscan ${MPIEXEC_POSTFLAGS}
        DEPENDS test_iexscan
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_reduce_scatter ${MPIEXEC_POSTFLAGS}
        DEPENDS test_reduce_scatter
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./test_ireduce_scatter ${MPIEXEC_POSTFLAGS}
        DEPENDS test_ireduce_scatter
        )
