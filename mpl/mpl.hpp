

#if !(defined MPL_HPP)

#define MPL_HPP

#include <mpi.h>
#include <cstddef>

namespace mpl {

  static constexpr int any_source = MPI_ANY_SOURCE;

  static constexpr int proc_null = MPI_PROC_NULL;

  static constexpr int undefined = MPI_UNDEFINED;

  static constexpr int root = MPI_ROOT;

  static constexpr void *absolute = MPI_BOTTOM;

  static constexpr int bsend_overheadroot = MPI_BSEND_OVERHEAD;

  using std::size_t;

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
