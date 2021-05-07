include(CMakeFindDependencyMacro)

find_dependency(Threads)
find_dependency(MPI)


include(${CMAKE_CURRENT_LIST_DIR}/mplTargets.cmake)
