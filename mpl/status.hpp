#if !(defined MPL_STATUS_HPP)

#define MPL_STATUS_HPP

#include <mpi.h>

namespace mpl {

  class status : private MPI_Status {
  public:
    int source() const {
      return MPI_Status::MPI_SOURCE;
    }
    int tag() const {
      return MPI_Status::MPI_TAG;
    }
    int error() const {
      return MPI_Status::MPI_ERROR;
    }
    bool is_cancelled() const{
      int result;
      MPI_Test_cancelled(const_cast<MPI_Status *>(reinterpret_cast<const MPI_Status *>(this)), &result);
      return result;
    }
    bool is_canceled() const {
      return is_cancelled();
    }
    status() {
      MPI_Status::MPI_SOURCE=MPI_ANY_SOURCE;
      MPI_Status::MPI_TAG=MPI_ANY_TAG;
      MPI_Status::MPI_ERROR=MPI_SUCCESS;
    }
  };
  
}

#endif
