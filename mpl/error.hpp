#if !(defined MPL_ERROR_HPP)

#define MPL_ERROR_HPP

#include <exception>

namespace mpl {

  class error : public ::std::exception {
  protected:
    const char * const str;
  public:
    explicit error(const char * const str="unknown") : str(str) {
    }
    virtual ~error() {
    }
    virtual const char * what() const noexcept {
      return str;
    }
  };

  class invalid_rank : public error {
  public:
    explicit invalid_rank() : error("invalid rank") {
    }
    virtual ~invalid_rank() {
    }
  };
  
  class invalid_tag : public error {
  public:
    explicit invalid_tag() : error("invalid tag") {
    }
    virtual ~invalid_tag() {
    }
  };

  class invalid_size : public error {
  public:
    explicit invalid_size() : error("invalid size") {
    }
    virtual ~invalid_size() {
    }
  };

  class invalid_dim : public error {
  public:
    explicit invalid_dim() : error("invalid dimension") {
    }
    virtual ~invalid_dim() {
    }
  };
  
}

#endif
