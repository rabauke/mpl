#if !(defined MPL_ERROR_HPP)

#define MPL_ERROR_HPP

#include <exception>

namespace mpl {

  class error : public ::std::exception {
  protected:
    const char *const str;
  public:
    explicit error(const char *const str="unknown") : str(str) {
    }

    ~error() override=default;

    virtual const char *what() const noexcept {
      return str;
    }
  };

  class invalid_rank : public error {
  public:
    invalid_rank() : error("invalid rank") {
    }

    ~invalid_rank() override=default;
  };

  class invalid_tag : public error {
  public:
    invalid_tag() : error("invalid tag") {
    }

    ~invalid_tag() override=default;
  };

  class invalid_size : public error {
  public:
    invalid_size() : error("invalid size") {
    }

    ~invalid_size() override=default;
  };

  class invalid_layout : public error {
  public:
    invalid_layout() : error("invalid layout") {
    }

    ~invalid_layout() override=default;
  };

  class invalid_dim : public error {
  public:
    invalid_dim() : error("invalid dimension") {
    }

    ~invalid_dim() override=default;
  };

  class invalid_datatype_bound : public error {
  public:
    invalid_datatype_bound() : error("invalid datatype bound") {
    }

    ~invalid_datatype_bound() override=default;
  };

}

#endif
