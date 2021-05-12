# 3.9 required for mpi targets, 3.5 required for boost targets
cmake_minimum_required(VERSION 3.9) 

# FIXME I didn't see versioning info on github,
# but semantic versioning should be considered
project(mpl VERSION 0.0.1 LANGUAGES CXX C)

option(BUILD_TESTING "" OFF)
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# project requires c++17 to build
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_library(mpl INTERFACE)
target_include_directories(mpl
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

# convention for allowing use as a subdirectory
add_library(mpl::mpl ALIAS mpl)
target_link_libraries(mpl INTERFACE Threads::Threads MPI::MPI_CXX)

option(MPL_BUILD_EXAMPLES OFF "build the mpl examples")
if(MPL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
# not prefixed because this is the default variable
# from include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mplConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpl)

install(DIRECTORY mpl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mpl EXPORT mplTargets )
export(EXPORT mplTargets
       NAMESPACE mpl::
       FILE mplTargets.cmake
       )
install(EXPORT mplTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpl
        NAMESPACE mpl::
        FILE mplTargets.cmake
        )
