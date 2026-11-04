#ifndef MPL_ERROR_HPP

#define MPL_ERROR_HPP

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mpi.h>


namespace mpl {

  /// Base class for all MPL exception classes that will be thrown in case of run-time errors.
  class error : public ::std::exception {
    std::runtime_error message_;

  public:
    error() : message_{""} {
    }

    /// \param message error message that will be returned by #what method
    explicit error(const std::string &message) : message_{message} {
    }

    /// \param message error message that will be returned by #what method
    explicit error(const char *const message) : message_{message} {
    }

    /// \return character pointer to error message
    [[nodiscard]] const char *what() const noexcept override {
      return message_.what();
    }
  };

  /// Will be thrown in case of invalid rank argument.
  class invalid_rank final : public error {
  public:
    invalid_rank() : error{"invalid rank"} {
    }
  };

  /// Will be thrown in case of invalid tag argument.
  class invalid_tag final : public error {
  public:
    invalid_tag() : error{"invalid tag"} {
    }
  };

  /// Will be thrown in case of invalid size argument.
  class invalid_size final : public error {
  public:
    invalid_size() : error{"invalid size"} {
    }
  };

  /// Will be thrown in case of invalid count argument.
  class invalid_count final : public error {
  public:
    invalid_count() : error{"invalid count"} {
    }
  };

  /// Will be thrown in case of invalid count argument.
  class invalid_displacement final : public error {
  public:
    invalid_displacement() : error{"invalid displacement"} {
    }
  };

  /// Will be thrown in case of invalid layout argument.
  class invalid_layout final : public error {
  public:
    invalid_layout() : error{"invalid layout"} {
    }
  };

  /// Will be thrown in case of invalid dimension.
  class invalid_dim final : public error {
  public:
    invalid_dim() : error{"invalid dimension"} {
    }
  };

  /// Will be thrown when an error occurs while manipulating layouts.
  class invalid_datatype_bound final : public error {
  public:
    invalid_datatype_bound() : error{"invalid datatype bound"} {
    }
  };

  /// Will be thrown in case of invalid arguments.
  class invalid_argument final : public error {
  public:
    invalid_argument() : error{"invalid argument"} {
    }
  };

  namespace detail {

    /// Determines the error class of a valid MPI error code, with a conservative fallback.
    inline int error_class_of(const int code) noexcept {
      int error_class{0};
      if (MPI_Error_class(code, &error_class) == MPI_SUCCESS)
        return error_class;
      return MPI_ERR_OTHER;
    }

    /// Determines the error message of a valid MPI error code, with a numeric fallback.
    inline std::string error_message(const int code) {
      char message[MPI_MAX_ERROR_STRING + 1]{};
      int len{0};
      if (MPI_Error_string(code, message, &len) == MPI_SUCCESS and len >= 0 and
          len <= MPI_MAX_ERROR_STRING) {
        return std::string{message, static_cast<std::string::size_type>(len)};
      }
      return "MPI error code " + std::to_string(code) + " (error string inquiry failed)";
    }

  }  // namespace detail

  /// Base class for all exceptions that originate from a failed MPI call.
  ///
  /// In contrast to the other exception classes, which are thrown by MPL before MPI is
  /// called, this class carries the MPI error code and the MPI error class.
  /// Constructing an mpi_error with a valid MPI error code is safe before MPI initialization
  /// and after finalization. Failed inquiries use a conservative class and numeric message.
  class mpi_error : public error {
    int code_;
    int class_;

  public:
    /// \param mpi_code valid MPI error code returned by an MPI function
    explicit mpi_error(const int mpi_code)
        : error{detail::error_message(mpi_code)},
          code_{mpi_code},
          class_{detail::error_class_of(mpi_code)} {
    }

    /// \return MPI error code
    [[nodiscard]] int error_code() const noexcept {
      return code_;
    }

    /// \return MPI error class
    [[nodiscard]] int error_class() const noexcept {
      return class_;
    }
  };

  /// Will be thrown in case of file-related io errors.
  class io_failure final : public mpi_error {
  public:
    explicit io_failure(const int code) : mpi_error{code} {
    }
  };

  /// Will be thrown if one or more requests of a multi-request completion call failed.
  ///
  /// The lists describe outcomes reported by the failing call, not the entire pool state.
  /// Unlisted requests are not necessarily pending; null or inactive persistent slots may
  /// be reported as successfully completed with empty statuses. Statuses reported by the
  /// failing call are available from the request pool.
  class request_error final : public mpi_error {
    std::vector<int> failed_;
    std::vector<int> completed_;

  public:
    /// \param code error code of the first failed request
    /// \param failed indices of the requests that completed with an error
    /// \param completed indices of the requests that completed successfully
    request_error(const int code, std::vector<int> failed, std::vector<int> completed = {})
        : mpi_error{code}, failed_{std::move(failed)}, completed_{std::move(completed)} {
    }

    /// \return indices of the requests that completed with an error
    [[nodiscard]] const std::vector<int> &failed_requests() const noexcept {
      return failed_;
    }

    /// \return indices of the requests that completed successfully in the failed call
    [[nodiscard]] const std::vector<int> &completed_requests() const noexcept {
      return completed_;
    }
  };


  namespace detail {

    inline bool is_io_error_class(const int error_class) {
      switch (error_class) {
        case MPI_ERR_FILE:
        case MPI_ERR_IO:
        case MPI_ERR_NO_SUCH_FILE:
        case MPI_ERR_FILE_EXISTS:
        case MPI_ERR_BAD_FILE:
        case MPI_ERR_ACCESS:
        case MPI_ERR_NO_SPACE:
        case MPI_ERR_QUOTA:
        case MPI_ERR_READ_ONLY:
        case MPI_ERR_FILE_IN_USE:
        case MPI_ERR_AMODE:
        case MPI_ERR_UNSUPPORTED_DATAREP:
        case MPI_ERR_UNSUPPORTED_OPERATION:
        case MPI_ERR_DUP_DATAREP:
        case MPI_ERR_CONVERSION:
          return true;
        default:
          return false;
      }
    }

    [[noreturn]] inline void throw_mpi_error(const int code) {
      if (is_io_error_class(error_class_of(code)))
        throw io_failure{code};
      throw mpi_error{code};
    }

    /// Throws an mpl::mpi_error if an MPI call did not return MPI_SUCCESS.
    inline void check(const int return_code) {
      if (return_code != MPI_SUCCESS)
        throw_mpi_error(return_code);
    }

    /// Makes MPI return error codes instead of aborting for the given communicator.
    inline void set_errors_return(const MPI_Comm comm) {
      if (comm != MPI_COMM_NULL)
        check(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
    }

  }  // namespace detail

}  // namespace mpl

#endif
