find_package(Boost COMPONENTS unit_test_framework REQUIRED)

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
  target_link_libraries(${TARGET_NAME} PUBLIC mpl::mpl Boost::unit_test_framework)
  add_test(NAME ${TARGET_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./${TARGET_NAME} ${MPIEXEC_POSTFLAGS})
endfunction()

add_test_executable(test_group test_group.cc)
add_test_executable(test_communicator test_communicator.cc)
add_test_executable(test_cartesian_communicator test_cartesian_communicator.cc)
add_test_executable(test_graph_communicator test_graph_communicator.cc)
add_test_executable(test_dist_graph_communicator test_dist_graph_communicator.cc)
add_test_executable(test_communicator_send_recv test_communicator_send_recv.cc)
add_test_executable(test_communicator_isend_irecv test_communicator_isend_irecv.cc test_helper.hpp)
add_test_executable(test_communicator_init_send_init_recv test_communicator_init_send_init_recv.cc test_helper.hpp)
add_test_executable(test_communicator_sendrecv test_communicator_sendrecv.cc test_helper.hpp)
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
