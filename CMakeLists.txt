# 3.10 required for mpi targets, 3.5 required for boost targets
cmake_minimum_required(VERSION 3.10)

# FIXME I didn't see versioning info on github,
# but semantic versioning should be considered
project(mpl VERSION 0.3.0 LANGUAGES CXX C)

if(NOT DEFINED CACHE{BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "")
endif()
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# project requires c++17 to build
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_library(mpl INTERFACE)
target_include_directories(mpl
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# convention for allowing use as a subdirectory
add_library(mpl::mpl ALIAS mpl)
target_link_libraries(mpl INTERFACE Threads::Threads MPI::MPI_CXX)

option(MPL_BUILD_EXAMPLES "build the mpl examples" OFF)
if(MPL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mplConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpl)

install(DIRECTORY mpl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mpl EXPORT mplTargets)
export(EXPORT mplTargets
    NAMESPACE mpl::
    FILE mplTargets.cmake)
install(EXPORT mplTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpl
    NAMESPACE mpl::
    FILE mplTargets.cmake)

option(MPL_BUILD_DOCUMENTATION "build the mpl documentation using Doxygen and Sphinx" OFF)
add_subdirectory(doc)
