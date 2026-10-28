include(CMakeFindDependencyMacro)

find_dependency(Threads)
find_dependency(MPI 4.1 REQUIRED COMPONENTS C CXX)


include(${CMAKE_CURRENT_LIST_DIR}/mplTargets.cmake)
