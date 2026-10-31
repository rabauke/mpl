#ifndef MPL_ERROR_HPP

#define MPL_ERROR_HPP

#include <exception>
#include <stdexcept>


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

  /// Will be thrown in case of file-related io errors.
  class io_failure final : public error {
    static std::string get_message(const int code) {
      char message[MPI_MAX_ERROR_STRING + 1]{};

      int len{0};
      MPI_Error_string(code, message, &len);
      message[len] = '\0';
      return std::string{message};
    }

  public:
    explicit io_failure(const int code) : error{get_message(code)} {
    }
  };

}  // namespace mpl

#endif
