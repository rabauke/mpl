#if !(defined MPL_DISPLACEMENTS_HPP)

#define MPL_DISPLACEMENTS_HPP

#include <cstddef>
#include <vector>

namespace mpl {

  class displacements : private std::vector<MPI_Aint> {
    typedef std::vector<MPI_Aint> base;
  public:
    typedef base::size_type size_type;
    explicit displacements(size_type n=0) : base(n, 0) {
    }
    displacements(std::initializer_list<MPI_Aint> init) : base(init) {
    }
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
    const MPI_Aint * operator()() const {
      return base::data();
    }
  };

}

#endif
