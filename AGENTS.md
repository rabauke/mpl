# MPL AI Coding Agent Guidelines

MPL is a modern, header-only C++17 message passing library for high-performance computing wrapping the MPI 4.1 standard.

## 1. Build and Test Workflow

### Configure and Build
```bash
cmake -B build -DBUILD_TESTING=ON -DMPL_BUILD_EXAMPLES=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$(nproc)
```

Existing build configurations (e.g. `cmake-build-debug-openmpi` or `cmake-build-debug-mpich`) can also be utilized directly if present.

### Targeted Test Execution
Always run targeted unit tests rather than the entire test suite during active development:
```bash
# Run a specific test via CTest
ctest --test-dir build --output-on-failure -R test_communicator_send_recv

# Or run directly via mpirun (use --oversubscribe for OpenMPI safety)
mpirun -n 2 --oversubscribe ./build/test/test_communicator_send_recv
```

## 2. Architecture & Header Index

MPL is header-only; all implementations reside under `mpl/`:

- **Environment & Lifecycle**:
  - `mpl/environment.hpp`: Global environment initialization, finalization, buffer management, processor name, timing.
  - `mpl/mpl.hpp`: Main umbrella header including the standard MPL modules.
  - `mpl/error.hpp`: Exception hierarchy (`mpl::error`, `mpl::invalid_rank`, `mpl::invalid_tag`, `mpl::io_failure`, etc.).
- **Communicators & Groups**:
  - `mpl/communicator.hpp`: Base `communicator` class and point-to-point / collective member functions.
  - `mpl/comm_group.hpp`: Process group abstractions (`mpl::group`).
  - `mpl/cartesian_communicator.hpp`: Cartesian grid topologies and shift operations.
  - `mpl/graph_communicator.hpp`: General graph topologies.
  - `mpl/distributed_graph_communicator.hpp`: Distributed graph topologies.
  - `mpl/topology_communicator.hpp`: Common base interfaces for topological communicators.
- **Data Types & Layouts**:
  - `mpl/datatype.hpp`: Type traits and mappings between C++ types and MPI basic datatypes.
  - `mpl/layout.hpp`: Memory layout abstractions (contiguous, vector, strided, indexed, subarray, struct layouts).
  - `mpl/flat_memory.hpp`: Memory block wrappers for contiguous serialization.
  - `mpl/displacements.hpp`: Array displacement helper utilities.
  - `mpl/distributed_grid.hpp`: Multi-dimensional distributed grid domain decomposition.
  - `mpl/vector.hpp`: Fixed-size multi-dimensional vector math helpers.
- **Point-to-Point & Collectives**:
  - `mpl/message.hpp`: Tagged message wrappers and matched probe constructs.
  - `mpl/request.hpp`: Non-blocking communication requests (`irequest`, `request_pool`).
  - `mpl/status.hpp`: Communication status representation (`status`).
  - `mpl/tag.hpp`: Message tags (`tag_t`, `tag_t::any()`).
  - `mpl/ranks.hpp`: Process rank collection abstractions (`ranks`).
  - `mpl/operator.hpp`: Reduction operations and user-defined reduction functors (`op`).
- **I/O & Utilities**:
  - `mpl/file.hpp`: Parallel MPI I/O file wrapper (`file`).
  - `mpl/info.hpp`: MPI info key-value pairs (`info`).
  - `mpl/command_line.hpp`: Command-line parsing helpers.
  - `mpl/utility.hpp`: Internal metaprogramming and utility helpers.

## 3. C++17 Design Invariants

- **Header-Only Simplicity**: Keep code in headers with appropriate `inline` functions and template definitions.
- **Zero-Overhead Abstractions**: Avoid heap allocations (`std::vector`, `std::string`, `new`) inside hot communication routines. Use stack buffers or flat memory views where possible.
- **RAII Resource Management**: MPI handles (`MPI_Comm`, `MPI_Datatype`, `MPI_Request`, `MPI_Op`, `MPI_File`, `MPI_Info`) must be encapsulated in RAII wrappers managing acquisition and destruction.
- **Error Handling**: Throw exceptions derived from `mpl::error` (see `mpl/error.hpp`) on invalid arguments or runtime MPI failures.
- **Modern Idioms**: Mark query methods `[[nodiscard]]` and non-throwing operations `noexcept`.

## 4. Pre-Submission Quality Checklist

Before completing any task, ensure:
1. **Clean Compilation**: Code compiles without warnings (`-Wall -Wextra -Wpedantic`).
2. **Targeted Verification**: Relevant unit tests pass under `mpirun` / `ctest`.
3. **Code Formatting**: Modified files are formatted using `clang-format -i <file>`.
4. **Documentation & Examples Sync**: Relevant `examples/` and Doxygen header comments are updated if public APIs were modified.
