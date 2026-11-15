find_package(Catch2 3 REQUIRED)

function(add_test_executable TARGET_NAME SOURCES)
  add_executable(${TARGET_NAME} ${SOURCES})
  target_compile_options(${TARGET_NAME} PRIVATE
     $<$<CXX_COMPILER_ID:GNU>:
          -Wall -Wextra -Wpedantic>
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
          -Wall -Wextra -Wpedantic -Wno-c++98-compat>
     $<$<CXX_COMPILER_ID:Intel>:
          -Wall>
     $<$<CXX_COMPILER_ID:MSVC>:
          /permissive- /W4 /WX>)
  target_link_libraries(${TARGET_NAME} PRIVATE mpl::mpl Catch2::Catch2WithMain)
  if(TARGET_NAME MATCHES "^test_error($|_debug$|_no_debug$|_injection($|_debug$))")
    set(TEST_RANKS 2)
  else()
    set(TEST_RANKS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  add_test(NAME ${TARGET_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${TEST_RANKS} ${MPIEXEC_PREFLAGS} ./${TARGET_NAME} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${TARGET_NAME} PROPERTIES ENVIRONMENT "HWLOC_COMPONENTS=-gl")
endfunction()

function(disable_mpl_debug TARGET_NAME)
  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /UMPL_DEBUG)
  else()
    target_compile_options(${TARGET_NAME} PRIVATE -UMPL_DEBUG)
  endif()
endfunction()

add_test_executable(test_group test_group.cc)
add_test_executable(test_communicator test_communicator.cc)
add_test_executable(test_initialization test_initialization.cc)
add_test_executable(test_cartesian_communicator test_cartesian_communicator.cc)
add_test_executable(test_graph_communicator test_graph_communicator.cc)
add_test_executable(test_dist_graph_communicator test_dist_graph_communicator.cc)
add_test_executable(test_communicator_send_recv test_communicator_send_recv.cc)
add_test_executable(test_communicator_isend_irecv test_communicator_isend_irecv.cc test_helper.hpp)
add_test_executable(test_communicator_init_send_init_recv test_communicator_init_send_init_recv.cc test_helper.hpp)
add_test_executable(test_communicator_sendrecv test_communicator_sendrecv.cc test_helper.hpp)
add_test_executable(test_communicator_isendrecv test_communicator_isendrecv.cc test_helper.hpp)
add_test_executable(test_communicator_probe test_communicator_probe.cc test_helper.hpp)
add_test_executable(test_communicator_mprobe_mrecv test_communicator_mprobe_mrecv.cc test_helper.hpp)
add_test_executable(test_communicator_barrier test_communicator_barrier.cc)
add_test_executable(test_communicator_bcast test_communicator_bcast.cc)
add_test_executable(test_communicator_gather test_communicator_gather.cc)
add_test_executable(test_communicator_gatherv test_communicator_gatherv.cc)
add_test_executable(test_communicator_allgather test_communicator_allgather.cc)
add_test_executable(test_communicator_allgatherv test_communicator_allgatherv.cc)
add_test_executable(test_communicator_scatter test_communicator_scatter.cc)
add_test_executable(test_communicator_scatterv test_communicator_scatterv.cc)
add_test_executable(test_communicator_alltoall test_communicator_alltoall.cc)
add_test_executable(test_communicator_alltoallv test_communicator_alltoallv.cc)
add_test_executable(test_communicator_reduce test_communicator_reduce.cc)
add_test_executable(test_communicator_allreduce test_communicator_allreduce.cc)
add_test_executable(test_communicator_reduce_scatter_block test_communicator_reduce_scatter_block.cc)
add_test_executable(test_communicator_reduce_scatter test_communicator_reduce_scatter.cc)
add_test_executable(test_communicator_scan test_communicator_scan.cc)
add_test_executable(test_communicator_exscan test_communicator_exscan.cc)
add_test_executable(test_displacements test_displacements.cc)
add_test_executable(test_inter_communicator test_inter_communicator.cc)
add_test_executable(test_info test_info.cc)
add_test_executable(test_file test_file.cc)
add_test_executable(test_error test_error.cc)
add_test_executable(test_error_debug test_error.cc)
target_compile_definitions(test_error_debug PRIVATE MPL_DEBUG)
add_test_executable(test_error_no_debug test_error.cc)
disable_mpl_debug(test_error_no_debug)
add_test_executable(test_mpi_communicator test_mpi_communicator.cc)
add_test_executable(test_error_injection "test_error_injection.cc;mpi_error_injection.cc")
disable_mpl_debug(test_error_injection)
add_test_executable(test_error_injection_debug "test_error_injection.cc;mpi_error_injection.cc")
target_compile_definitions(test_error_injection_debug PRIVATE MPL_DEBUG)
set_tests_properties(test_error test_error_debug test_error_no_debug test_error_injection test_error_injection_debug PROPERTIES TIMEOUT 30)
add_test(NAME test_error_injection_single COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_error_injection "~intercommunicator copy assignment preserves ownership on failure" ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_error_injection_single PROPERTIES ENVIRONMENT "HWLOC_COMPONENTS=-gl" TIMEOUT 30)
add_test_executable(test_operator test_operator.cc)
add_test_executable(test_distributed_grid test_distributed_grid.cc)

# Compile test_error with MPL_DEBUG, which enables the rank, tag and dimension argument checks
# of MPL.  Without this option, the corresponding tests are reported as skipped.
option(MPL_TEST_DEBUG_CHECKS "Build test_error with MPL_DEBUG to run the argument check tests" OFF)
if(MPL_TEST_DEBUG_CHECKS)
  target_compile_definitions(test_error PRIVATE MPL_DEBUG)
endif()

# Plain program (no Catch2, no mpl::environment) that runs on a single process and checks
# that mpl::mpi_error is usable before MPI_Init and after MPI_Finalize.
add_executable(test_error_no_mpi test_error_no_mpi.cc)
target_compile_options(test_error_no_mpi PRIVATE
   $<$<CXX_COMPILER_ID:GNU>:
        -Wall -Wextra -Wpedantic>
   $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall -Wextra -Wpedantic -Wno-c++98-compat>)
target_link_libraries(test_error_no_mpi PRIVATE mpl::mpl)
add_test(NAME test_error_no_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./test_error_no_mpi ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_error_no_mpi PROPERTIES ENVIRONMENT "HWLOC_COMPONENTS=-gl" TIMEOUT 30)

add_executable(test_error_external_init test_error_external_init.cc)
target_compile_options(test_error_external_init PRIVATE
   $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
   $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic -Wno-c++98-compat>)
target_link_libraries(test_error_external_init PRIVATE mpl::mpl)
disable_mpl_debug(test_error_external_init)
add_test(NAME test_error_external_init COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_error_external_init ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_error_external_init PROPERTIES ENVIRONMENT "HWLOC_COMPONENTS=-gl" TIMEOUT 30)

add_executable(test_error_external_init_debug test_error_external_init.cc)
target_compile_options(test_error_external_init_debug PRIVATE
   $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
   $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic -Wno-c++98-compat>)
target_link_libraries(test_error_external_init_debug PRIVATE mpl::mpl)
target_compile_definitions(test_error_external_init_debug PRIVATE MPL_DEBUG)
add_test(NAME test_error_external_init_debug COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_error_external_init_debug ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_error_external_init_debug PROPERTIES ENVIRONMENT "HWLOC_COMPONENTS=-gl" TIMEOUT 30)
