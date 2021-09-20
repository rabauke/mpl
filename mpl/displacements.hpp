#if !(defined MPL_DISPLACEMENTS_HPP)

#define MPL_DISPLACEMENTS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace mpl {

  /// \brief Indicates the beginning of data buffers in various collective communication
  /// operations.
  class displacements : private std::vector<MPI_Aint> {
    using base = std::vector<MPI_Aint>;

  public:
    using size_type = base::size_type;
    using value_type = base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// \brief Constructs a set of displacements with displacement zero.
    /// \param n number of displacements
    explicit displacements(size_type n = 0) : base(n, 0) {}

    /// \brief Constructs a set of displacements with given displacements.
    /// \param init initial displacements
    explicit displacements(std::initializer_list<MPI_Aint> init) : base(init) {}

    /// \brief Copy constructor.
    /// \param other the other set of displacements to copy from
    displacements(const displacements &other) = default;
    /// \brief Move constructor.
    /// \param other the other set of displacements to move from
    displacements(displacements &&other) = default;

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
    using base::resize;

    /// \brief Get raw displacement data.
    /// \return pointer to array of displacements
    const MPI_Aint *operator()() const { return base::data(); }
  };

}  // namespace mpl

#endif
