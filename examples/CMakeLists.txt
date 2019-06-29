cmake_minimum_required(VERSION 3.0)

project(mpl_examples)

set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED) 
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMPL_DEBUG")

include_directories("../")

add_executable(arrays arrays.cc)
add_executable(blocking blocking.cc)
add_executable(blocking_vector blocking_vector.cc)
add_executable(collective collective.cc)
add_executable(communicator communicator.cc)
add_executable(distributed_grid distributed_grid.cc)
add_executable(distributed_grid_scatter_gather distributed_grid_scatter_gather.cc)
add_executable(gather gather.cc)
add_executable(gatherv gatherv.cc)
add_executable(heat_equation_Jacobi_method heat_equation_Jacobi_method.cc)
add_executable(heat_equation_successive_over-relaxation heat_equation_successive_over-relaxation.cc)
add_executable(hello_world hello_world.cc)
add_executable(iterators iterators.cc)
add_executable(layouts layouts.cc)
add_executable(matrix_gather matrix_gather.cc)
add_executable(nonblocking nonblocking.cc)
add_executable(nonblocking_mult nonblocking_mult.cc)
add_executable(probe probe.cc)
add_executable(reduce_lcm reduce_lcm.cc)
add_executable(reduce_min_loc reduce_min_loc.cc)
add_executable(standard_types standard_types.cc)
set_property(TARGET standard_types PROPERTY CXX_STANDARD 14)
add_executable(struct struct.cc)
add_executable(subarray subarray.cc)
add_executable(vibrating_string_mpl vibrating_string_mpl.cc)
add_executable(vibrating_string_mpi vibrating_string_mpi.c)
target_link_libraries(vibrating_string_mpi m)
add_executable(parallel_sort_mpl parallel_sort_mpl.cc)
set_property(TARGET parallel_sort_mpl PROPERTY CXX_STANDARD 17)
add_executable(parallel_sort_mpi parallel_sort_mpi.c)

include(../cmake/PVS-Studio.cmake)
pvs_studio_add_target(TARGET PVS-Studio_analyze ALL
                      OUTPUT FORMAT errorfile
                      RECURSIVE ANALYZE
                      arrays blocking blocking_vector collective communicator distributed_grid
                      distributed_grid_scatter_gather gather gatherv
                      heat_equation_Jacobi_method heat_equation_successive_over-relaxation
                      hello_world iterators layouts matrix_gather nonblocking nonblocking_mult probe reduce_lcm
                      reduce_min_loc standard_types struct vibrating_string_mpi vibrating_string_mpl parallel_sort_mpl
                      parallel_sort_mpi
                      MODE GA:1,2,3 OP LOG target.err)
