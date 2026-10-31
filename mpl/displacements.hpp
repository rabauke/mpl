#ifndef MPL_DISPLACEMENTS_HPP

#define MPL_DISPLACEMENTS_HPP

#include <cstddef>
#include <vector>
#include <utility>


namespace mpl {

  /// Set of %displacements indicates the beginning of data buffers in various collective
  /// communication operations.
  /// \note Individual %displacements are always given in bytes.
  class displacements : private std::vector<ssize_t> {
    using base = std::vector<ssize_t>;

  public:
    using size_type = base::size_type;
    using value_type = base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// Constructs a set of displacements with displacement zero.
    /// \param n number of displacements
    explicit displacements(size_type n = 0) : base(n, 0) {
    }

    /// Constructs a set of displacements with given displacements.
    /// \param init initial displacements
    explicit displacements(std::initializer_list<ssize_t> init) : base(init) {
    }

    /// Copy constructor.
    /// \param other the other set of displacements to copy from
    displacements(const displacements& other) = default;
    /// Move constructor.
    /// \param other the other set of displacements to move from
    displacements(displacements&& other) = default;

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
    using base::resize;
  };

}  // namespace mpl

#endif
