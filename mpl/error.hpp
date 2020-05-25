#if !(defined MPL_ERROR_HPP)

#define MPL_ERROR_HPP

#include <exception>

namespace mpl {

  class error : public ::std::exception {
  protected:
    const char *const str;

  public:
    explicit error(const char *const str = "unknown") : str(str) {}

    const char *what() const noexcept override { return str; }
  };

  class invalid_rank : public error {
  public:
    invalid_rank() : error("invalid rank") {}
  };

  class invalid_tag : public error {
  public:
    invalid_tag() : error("invalid tag") {}
  };

  class invalid_size : public error {
  public:
    invalid_size() : error("invalid size") {}
  };

  class invalid_count : public error {
  public:
    invalid_count() : error("invalid count") {}
  };

  class invalid_layout : public error {
  public:
    invalid_layout() : error("invalid layout") {}
  };

  class invalid_dim : public error {
  public:
    invalid_dim() : error("invalid dimension") {}
  };

  class invalid_datatype_bound : public error {
  public:
    invalid_datatype_bound() : error("invalid datatype bound") {}
  };

  class invalid_argument : public error {
  public:
    invalid_argument() : error("invalid argument") {}
  };

}  // namespace mpl

#endif
