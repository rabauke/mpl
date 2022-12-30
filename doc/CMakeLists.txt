set(doxygen_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
set(doxygen_binary_dir ${CMAKE_CURRENT_BINARY_DIR}/doxygen)

set(sphinx_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(sphinx_binary_dir ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

configure_file(${doxygen_source_dir}/Doxyfile.in ${doxygen_binary_dir}/Doxyfile @ONLY)
configure_file(${sphinx_source_dir}/conf.py.in ${sphinx_binary_dir}/conf.py @ONLY)

set(documentation_sources
    sphinx/auxiliary.rst
    sphinx/communicator.rst
    sphinx/conf.py.in
    sphinx/constants.rst
    sphinx/data_types.rst
    sphinx/environmental_management.rst
    sphinx/error.rst
    sphinx/examples/arrays.rst
    sphinx/examples/blocking.rst
    sphinx/examples/blocking_vector.rst
    sphinx/examples/collective.rst
    sphinx/examples/communicator.rst
    sphinx/examples/distributed_grid.rst
    sphinx/examples/distributed_grid_scatter_gather.rst
    sphinx/examples/gather.rst
    sphinx/examples/gatherv.rst
    sphinx/examples/heat_equation_Jacobi_method.rst
    sphinx/examples/heat_equation_successive_over-relaxation.rst
    sphinx/examples/hello_world.rst
    sphinx/examples/index.rst
    sphinx/examples/iterators.rst
    sphinx/examples/layouts.rst
    sphinx/examples/matrix_gather.rst
    sphinx/examples/nonblocking_mult.rst
    sphinx/examples/nonblocking.rst
    sphinx/examples/parallel_sort.rst
    sphinx/examples/probe.rst
    sphinx/examples/reduce.rst
    sphinx/examples/standard_types.rst
    sphinx/examples/stl_container.rst
    sphinx/examples/struct.rst
    sphinx/examples/subarray.rst
    sphinx/examples/vibrating_string.rst
    sphinx/grid.rst
    sphinx/group.rst
    sphinx/index.rst
    sphinx/info.rst
    sphinx/installation.rst
    sphinx/layouts.rst
    sphinx/reduction_operations.rst
    sphinx/tags.rst
    sphinx/_static/custom.css)

add_custom_command(OUTPUT ${doxygen_binary_dir}/xml/index.xml
    COMMAND doxygen Doxyfile
    WORKING_DIRECTORY ${doxygen_binary_dir}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

add_custom_command(OUTPUT ${sphinx_binary_dir}/html/index.html
    DEPENDS
    ${doxygen_binary_dir}/xml/index.xml
    ${documentation_sources}
    COMMAND sphinx-build -b html -j auto -c ${sphinx_binary_dir} ${sphinx_source_dir} ${sphinx_binary_dir}/html
    WORKING_DIRECTORY ${sphinx_binary_dir}
    COMMENT "Generating API documentation with Sphinx & Breathe"
    VERBATIM)

if(MPL_BUILD_DOCUMENTATION)
  add_custom_target(documentation ALL DEPENDS ${sphinx_binary_dir}/html/index.html)
else()
  add_custom_target(documentation DEPENDS ${sphinx_binary_dir}/html/index.html)
endif()

# Documentation installation (Sphinx output only).
include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx/html/ DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT documentation PATTERN ".*" EXCLUDE)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx/html)

set(documentation_output_dirs ${doxygen_binary_dir}/html ${doxygen_binary_dir}/xml ${sphinx_binary_dir}/html)

# Remove documentation via global clean target.
set_property(TARGET documentation APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${documentation_output_dirs})
# Custom target to remove documentation only.
add_custom_target(documentation_clean COMMAND ${CMAKE_COMMAND} -E rm -rf ${documentation_output_dirs})
