#if !(defined MPL_TAG_HPP)

#define MPL_TAG_HPP

#include <mpl/utility.hpp>
#include <ostream>
#include <istream>

namespace mpl {

  class tag {
  private:
    int t = 0;

  public:
    tag() = default;

    template<typename T>
    tag(T t) : t(static_cast<int>(t)) {
      static_assert(detail::is_valid_tag<T>::value,
                    "not an enumeration type or underlying enumeration type too large");
    }

    explicit tag(int t) : t(t) {}

    explicit operator int() const { return t; }

    static inline tag up();

    static inline tag any();
  };

  inline bool operator==(tag t1, tag t2) {
    return static_cast<int>(t1) == static_cast<int>(t2);
  }

  inline bool operator!=(tag t1, tag t2) {
    return static_cast<int>(t1) != static_cast<int>(t2);
  }

  template<typename CharT, typename Traits>
  std::basic_ostream<CharT, Traits> &operator<<(std::basic_ostream<CharT, Traits> &os, tag t) {
    return os << static_cast<int>(t);
  }

  template<typename CharT, typename Traits>
  std::basic_istream<CharT, Traits> &operator<<(std::basic_istream<CharT, Traits> &is, tag &t) {
    int t_;
    is >> t_;
    if (is)
      t = tag(t_);
    return is;
  }

}  // namespace mpl

#endif
