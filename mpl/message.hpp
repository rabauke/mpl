#if !(defined MPL_MESSAGE_HPP)

#define MPL_MESSAGE_HPP

#include <mpi.h>


namespace mpl {

  /// Status of a received message.
  struct message_t {
    message_t() = default;

    MPI_Message message_{MPI_MESSAGE_NULL};
  };

}  // namespace mpl

#endif
