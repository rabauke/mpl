#if !(defined MPL_TAG_HPP)

#define MPL_TAG_HPP

#include <mpl/utility.hpp>
#include <ostream>
#include <istream>

namespace mpl {

  /// Class for representing tag parameters in communication operations.
  class tag_t {
  private:
    int t = 0;

  public:
    tag_t() = default;

    /// Initializes tag from an enum value.  The enum's underlying type must be convertible to
    /// int without loss of precession (narrowing).
    /// \param t tag value
    template<typename T>
    tag_t(T t) : t(static_cast<int>(t)) {
      static_assert(detail::is_valid_tag_v<T>,
                    "not an enumeration type or underlying enumeration type too large");
    }

    /// Initializes tag from an int value.
    /// \param t tag value
    explicit tag_t(int t) : t(t) {}

    /// \return tag value as int
    explicit operator int() const { return t; }

    /// \return tag with largest value when converted to int
    static inline tag_t up();

    /// \return wildcard tag to be used in receive operations, e.g., \ref communicator_recv
    /// "communicator::recv", to indicate acceptance of a message with any tag value
    /// \see any_source
    static inline tag_t any();
  };

  /// \param t1 first tag to compare
  /// \param t2 second tag to compare
  /// \return true if both tags are convertible to the same int value
  inline bool operator==(tag_t t1, tag_t t2) {
    return static_cast<int>(t1) == static_cast<int>(t2);
  }

  /// \param t1 first tag to compare
  /// \param t2 second tag to compare
  /// \return true if both tags are not convertible to the same int value
  inline bool operator!=(tag_t t1, tag_t t2) {
    return static_cast<int>(t1) != static_cast<int>(t2);
  }

  /// Write tag into output stream in numerical representation.
  /// \param os output stream
  /// \param t tag to write into stream
  /// \return output stream
  template<typename CharT, typename Traits>
  std::basic_ostream<CharT, Traits> &operator<<(std::basic_ostream<CharT, Traits> &os,
                                                tag_t t) {
    return os << static_cast<int>(t);
  }

  /// Read tag given in numerical representation from input stream.
  /// \param is input stream
  /// \param t tag to read from stream
  /// \return input stream
  template<typename CharT, typename Traits>
  std::basic_istream<CharT, Traits> &operator>>(std::basic_istream<CharT, Traits> &is,
                                                tag_t &t) {
    int t_;
    is >> t_;
    if (is)
      t = tag_t(t_);
    return is;
  }

}  // namespace mpl

#endif
