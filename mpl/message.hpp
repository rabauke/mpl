#if !(defined MPL_MESSAGE_HPP)

#define MPL_MESSAGE_HPP

#include <mpi.h>

namespace mpl {

  /// Status of a received message.
  using message_t = MPI_Message;

}

#endif
