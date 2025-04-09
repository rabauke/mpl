#if !(defined MPL_ERROR_HPP)

#define MPL_ERROR_HPP

#include <exception>


namespace mpl {

  /// Base class for all MPL exception classes that will be thrown in case of run-time errors.
  class error : public ::std::exception {
  private:
    const char *const messsage_;

  public:
    /// \param message error message that will be returned by #what method
    explicit error(const char *const message = "unknown") : messsage_{message} {
    }

    /// \return character pointer to error message
    [[nodiscard]] const char *what() const noexcept override {
      return messsage_;
    }
  };

  /// Will be thrown in case of invalid rank argument.
  class invalid_rank : public error {
  public:
    invalid_rank() : error{"invalid rank"} {
    }
  };

  /// Will be thrown in case of invalid tag argument.
  class invalid_tag : public error {
  public:
    invalid_tag() : error{"invalid tag"} {
    }
  };

  /// Will be thrown in case of invalid size argument.
  class invalid_size : public error {
  public:
    invalid_size() : error{"invalid size"} {
    }
  };

  /// Will be thrown in case of invalid count argument.
  class invalid_count : public error {
  public:
    invalid_count() : error{"invalid count"} {
    }
  };

  /// Will be thrown in case of invalid layout argument.
  class invalid_layout : public error {
  public:
    invalid_layout() : error{"invalid layout"} {
    }
  };

  /// Will be thrown in case of invalid dimension.
  class invalid_dim : public error {
  public:
    invalid_dim() : error{"invalid dimension"} {
    }
  };

  /// Will be thrown when an error occurs while manipulating layouts.
  class invalid_datatype_bound : public error {
  public:
    invalid_datatype_bound() : error{"invalid datatype bound"} {
    }
  };

  /// Will be thrown in case of invalid arguments.
  class invalid_argument : public error {
  public:
    invalid_argument() : error{"invalid argument"} {
    }
  };

  /// Will be thrown in case of file-related io errors.
  class io_failure : public error {
    char message_[MPI_MAX_ERROR_STRING + 1]{};

  public:
    explicit io_failure(int code) {
      int len{0};
      MPI_Error_string(code, message_, &len);
      message_[len] = '\0';
    }

    /// \return character pointer to error message
    [[nodiscard]] const char *what() const noexcept override {
      return message_;
    }
  };

}  // namespace mpl

#endif
