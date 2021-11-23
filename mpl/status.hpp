#if !(defined MPL_STATUS_HPP)

#define MPL_STATUS_HPP

#include <mpi.h>

namespace mpl {

  namespace impl {
    class base_communicator;

    template<typename T>
    class base_request;

    template<typename T>
    class request_pool;
  }

  //--------------------------------------------------------------------------------------------

  /// Class that represents the status of a received message.
  class status_t : private MPI_Status {
  public:
    /// \return source of the message
    [[nodiscard]] int source() const { return MPI_Status::MPI_SOURCE; }

    /// \return tag value of the message
    [[nodiscard]] mpl::tag_t tag() const { return mpl::tag_t(MPI_Status::MPI_TAG); }

    /// \return error code associated with the message
    [[nodiscard]] int error() const { return MPI_Status::MPI_ERROR; }

    /// \return true if associated request has been been canceled
    [[nodiscard]] bool is_cancelled() const {
      int result;
      MPI_Test_cancelled(static_cast<const MPI_Status *>(this), &result);
      return result != 0;
    }

    /// \return true if associated request has been been canceled
    [[nodiscard]] bool is_canceled() const { return is_cancelled(); }

    /// \return number of top level elements of type T received in associated message
    template<typename T>
    [[nodiscard]] int get_count() const {
      int result;
      MPI_Get_count(static_cast<const MPI_Status *>(this),
                    detail::datatype_traits<T>::get_datatype(), &result);
      return result;
    }

    /// \param l layout used in associated message
    /// \return number of top level elements of type T received in associated message
    template<typename T>
    [[nodiscard]] int get_count(const layout<T> &l) const {
      int result;
      MPI_Get_count(static_cast<const MPI_Status *>(this),
                    detail::datatype_traits<layout<T>>::get_datatype(l), &result);
      return result;
    }

    /// default constructor initializes source and tag with wildcards given by \ref any_source
    /// and \ref tag_t::any and no error
    status_t() : MPI_Status{} {
      MPI_Status::MPI_SOURCE = MPI_ANY_SOURCE;
      MPI_Status::MPI_TAG = MPI_ANY_TAG;
      MPI_Status::MPI_ERROR = MPI_SUCCESS;
    }

    friend class impl::base_communicator;
    template<typename T>
    friend class impl::base_request;
    template<typename T>
    friend class impl::request_pool;
  };

}  // namespace mpl

#endif
