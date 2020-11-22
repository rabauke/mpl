#if !(defined MPL_STATUS_HPP)

#define MPL_STATUS_HPP

#include <mpi.h>

namespace mpl {

  /// Class that represents the status of a received message.
  class status : private MPI_Status {
  public:
    /// \return source of the message
    int source() const { return MPI_Status::MPI_SOURCE; }

    /// \return tag value of the message
    mpl::tag tag() const { return mpl::tag(MPI_Status::MPI_TAG); }

    /// \return error code associated with the message
    int error() const { return MPI_Status::MPI_ERROR; }

    /// \return true if associated request has been been canceled
    bool is_cancelled() const {
      int result;
      MPI_Test_cancelled(reinterpret_cast<const MPI_Status *>(this), &result);
      return result != 0;
    }

    /// \return true if associated request has been been canceled
    bool is_canceled() const { return is_cancelled(); }

    /// \return number of top level elements of type T received in associated message
    template<typename T>
    int get_count() const {
      int result;
      MPI_Get_count(reinterpret_cast<const MPI_Status *>(this),
                    detail::datatype_traits<T>::get_datatype(), &result);
      return result;
    }

    /// \param l layout used in associated message
    /// \return number of top level elements of type T received in associated message
    template<typename T>
    int get_count(const layout<T> &l) const {
      int result;
      MPI_Get_count(reinterpret_cast<const MPI_Status *>(this),
                    detail::datatype_traits<layout<T>>::get_datatype(l), &result);
      return result;
    }

    /// default constructor initializes source and tag with wildcards given by \ref any_source
    /// and \ref tag::any and no error
    status() : MPI_Status{} {
      MPI_Status::MPI_SOURCE = MPI_ANY_SOURCE;
      MPI_Status::MPI_TAG = MPI_ANY_TAG;
      MPI_Status::MPI_ERROR = MPI_SUCCESS;
    }
  };

}  // namespace mpl

#endif
