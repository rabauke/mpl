function(add_mpl_executable TARGET_NAME SOURCES)
  add_executable(${TARGET_NAME} ${SOURCES})
  target_link_libraries(${TARGET_NAME} PUBLIC mpl::mpl)
  # add MPL_Debug definition if building in debug mode
  target_compile_definitions(${TARGET_NAME} PUBLIC 
    $<$<CONFIG:Debug>:MPL_DEBUG>)
endfunction()


add_mpl_executable(arrays arrays.cc)
add_mpl_executable(blocking blocking.cc)
add_mpl_executable(blocking_vector blocking_vector.cc)
add_mpl_executable(collective collective.cc)
add_mpl_executable(communicator communicator.cc)
add_mpl_executable(distributed_grid distributed_grid.cc)
add_mpl_executable(distributed_grid_scatter_gather distributed_grid_scatter_gather.cc)
add_mpl_executable(gather gather.cc)
add_mpl_executable(gatherv gatherv.cc)
add_mpl_executable(heat_equation_Jacobi_method heat_equation_Jacobi_method.cc)
add_mpl_executable(heat_equation_successive_over-relaxation heat_equation_successive_over-relaxation.cc)
add_mpl_executable(hello_world hello_world.cc)
add_mpl_executable(iterators iterators.cc)
add_mpl_executable(layouts layouts.cc)
add_mpl_executable(matrix_gather matrix_gather.cc)
add_mpl_executable(nonblocking nonblocking.cc)
add_mpl_executable(nonblocking_mult nonblocking_mult.cc)
add_mpl_executable(probe probe.cc)
add_mpl_executable(reduce_lcm reduce_lcm.cc)
add_mpl_executable(reduce_min_loc reduce_min_loc.cc)
add_mpl_executable(stl_container stl_container.cc)
add_mpl_executable(struct struct.cc)
add_mpl_executable(subarray subarray.cc)
add_mpl_executable(vibrating_string_mpl vibrating_string_mpl.cc)
add_mpl_executable(standard_types standard_types.cc)
add_mpl_executable(parallel_sort_mpl parallel_sort_mpl.cc)

# c only examples
add_executable(vibrating_string_mpi vibrating_string_mpi.c)
target_link_libraries(vibrating_string_mpi PUBLIC m MPI::MPI_C)
add_executable(parallel_sort_mpi parallel_sort_mpi.c)
target_link_libraries(parallel_sort_mpi PUBLIC MPI::MPI_C)

find_program(PVS_STUDIO pvs-studio)

if (PVS_STUDIO)
    include(../cmake/PVS-Studio.cmake)
    pvs_studio_add_target(TARGET PVS-Studio_analyze # ALL
            OUTPUT FORMAT errorfile
            RECURSIVE ANALYZE
            arrays blocking blocking_vector collective communicator distributed_grid
            distributed_grid_scatter_gather gather gatherv
            heat_equation_Jacobi_method heat_equation_successive_over-relaxation
            hello_world iterators layouts matrix_gather nonblocking nonblocking_mult probe reduce_lcm
            reduce_min_loc standard_types struct vibrating_string_mpi vibrating_string_mpl parallel_sort_mpl
            parallel_sort_mpi
            MODE GA:1,2,3 OP LOG target.err)
endif ()
