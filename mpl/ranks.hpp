#if !(defined MPL_RANKS_HPP)

#define MPL_RANKS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace mpl {

  /// \brief Represents a collection of ranks.
  /// \see class \ref group
  class ranks : private std::vector<int> {
    using base = std::vector<int>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// \brief Constructs collection of ranks with all ranks having value zero.
    /// \param n initial size of the collection
    explicit ranks(size_type n = 0) : base(n, 0) {}

    /// \brief Constructs collection of ranks from a braces expression of integers.
    /// \param init list of initial values
    ranks(std::initializer_list<int> init) : base(init) {}

    /// \brief Constructs collection of ranks from another collection.
    /// \param other the other collection to copy from
    ranks(const ranks &other) = default;

    /// \brief Move-constructs collection of ranks from another collection.
    /// \param other the other collection to move from
    ranks(ranks &&other) noexcept : base(std::move(other)) {}

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    /// \brief Gives access to internal data.
    /// \return pointer to constant array
    const int *operator()() const { return base::data(); }

    /// \brief Gives access to internal data.
    /// \return pointer to array
    int *operator()() { return base::data(); }
  };

}  // namespace mpl

#endif
