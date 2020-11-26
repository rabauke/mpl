#if !(defined MPL_HPP)

#define MPL_HPP

#include <mpi.h>
#include <cstddef>

namespace mpl {

  /// \brief Wildcard value to indicate in a receive operation, e.g., \ref communicator_recv
  /// "communicator::recv", that any source is acceptable.
  /// \see tag::any
  constexpr int any_source = MPI_ANY_SOURCE;

  /// \brief Special value that can be used instead of a rank wherever a source or a
  /// destination argument is required in a call to indicate that the communication shall have
  /// no effect.
  constexpr int proc_null = MPI_PROC_NULL;

  /// \brief Special value that is used to indicate an invalid return value or function
  /// parameter in some functions.
  constexpr int undefined = MPI_UNDEFINED;

  /// \brief Special value to indicate the root process in some intercommunicator collective
  /// operations.
  constexpr int root = MPI_ROOT;

  /// \brief Special constant to indicate the start of the address range of message buffers.
  constexpr void *absolute = MPI_BOTTOM;

  /// \brief Special constant representing an upper bound on the additional space consumed when
  /// buffering messages.
  /// \see \ref communicator_bsend "communicator::bsend"
  constexpr int bsend_overhead = MPI_BSEND_OVERHEAD;

  using size_t = std::size_t;
  using ssize_t = std::ptrdiff_t;

}  // namespace mpl

#include <mpl/error.hpp>
#include <mpl/displacements.hpp>
#include <mpl/tag.hpp>
#include <mpl/ranks.hpp>
#include <mpl/flat_memory.hpp>
#include <mpl/datatype.hpp>
#include <mpl/layout.hpp>
#include <mpl/status.hpp>
#include <mpl/message.hpp>
#include <mpl/operator.hpp>
#include <mpl/request.hpp>
#include <mpl/comm_group.hpp>
#include <mpl/environment.hpp>
#include <mpl/topo_comm.hpp>
#include <mpl/cart_comm.hpp>
#include <mpl/graph_comm.hpp>
#include <mpl/dist_graph_comm.hpp>
#include <mpl/distributed_grid.hpp>

#endif
