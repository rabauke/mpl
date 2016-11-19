#if !(defined MPL_DISPLACEMENTS_HPP)

#define MPL_DISPLACEMENTS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace mpl {

  class displacements : private std::vector<MPI_Aint> {
    typedef std::vector<MPI_Aint> base;
  public:
    typedef base::size_type size_type;
    typedef base::value_type value_type;
    typedef base::iterator iterator;
    typedef base::const_iterator const_iterator;
    explicit displacements(size_type n=0) : base(n, 0) {
    }
    displacements(std::initializer_list<MPI_Aint> init) : base(init) {
    }
    displacements(const displacements &other) : base(other) {
    }
    displacements(displacements &&other) : base(std::move(other)) {
    }
    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
    using base::resize;
    const MPI_Aint * operator()() const {
      return base::data();
    }
  };

}

#endif
