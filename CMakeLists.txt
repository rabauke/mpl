# 3.21 required for PROJECT_IS_TOP_LEVEL, 3.10 required for mpi targets, 3.5 required for boost targets
cmake_minimum_required(VERSION 3.21)

project(mpl VERSION 0.4.0 LANGUAGES CXX C)

if(NOT DEFINED CACHE{BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "")
endif()
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI 3.1 REQUIRED C CXX)

add_library(mpl INTERFACE)
target_include_directories(mpl
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(mpl INTERFACE cxx_std_17)

# convention for allowing use as a subdirectory
add_library(mpl::mpl ALIAS mpl)
target_link_libraries(mpl INTERFACE MPI::MPI_CXX)

option(MPL_BUILD_EXAMPLES "build the mpl examples" ${PROJECT_IS_TOP_LEVEL})
if(MPL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(MPL_INSTALL "Generate and install MPL target" ${PROJECT_IS_TOP_LEVEL})
if(MPL_INSTALL)
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
endif()

option(MPL_BUILD_DOCUMENTATION "build the mpl documentation using Doxygen and Sphinx" OFF)
add_subdirectory(doc)
