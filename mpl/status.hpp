#if !(defined MPL_STATUS_HPP)

#define MPL_STATUS_HPP

#include <mpi.h>

namespace mpl {

  class status : private MPI_Status {
  public:
    int source() const { return MPI_Status::MPI_SOURCE; }

    mpl::tag tag() const { return mpl::tag(MPI_Status::MPI_TAG); }

    int error() const { return MPI_Status::MPI_ERROR; }

    bool is_cancelled() const {
      int result;
      MPI_Test_cancelled(reinterpret_cast<const MPI_Status *>(this), &result);
      return result != 0;
    }

    bool is_canceled() const { return is_cancelled(); }

    template<typename T>
    int get_count() const {
      int result;
      MPI_Get_count(reinterpret_cast<const MPI_Status *>(this),
                    datatype_traits<T>::get_datatype(), &result);
      return result;
    }

    template<typename T>
    int get_count(const layout<T> &l) const {
      int result;
      MPI_Get_count(reinterpret_cast<const MPI_Status *>(this),
                    datatype_traits<layout<T>>::get_datatype(l), &result);
      return result;
    }

    status() {
      MPI_Status::MPI_SOURCE = MPI_ANY_SOURCE;
      MPI_Status::MPI_TAG = MPI_ANY_TAG;
      MPI_Status::MPI_ERROR = MPI_SUCCESS;
    }
  };

}  // namespace mpl

#endif
