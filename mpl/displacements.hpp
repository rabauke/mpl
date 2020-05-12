#if !(defined MPL_DISPLACEMENTS_HPP)

#define MPL_DISPLACEMENTS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace mpl {

  class displacements : private std::vector<MPI_Aint> {
    using base = std::vector<MPI_Aint>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    explicit displacements(size_type n = 0) : base(n, 0) {}

    displacements(std::initializer_list<MPI_Aint> init) : base(init) {}

    displacements(const displacements &other) = default;

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

    const MPI_Aint *operator()() const { return base::data(); }
  };

}  // namespace mpl

#endif
