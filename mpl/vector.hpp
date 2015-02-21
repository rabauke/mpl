#if !(defined MPL_VECTOR_HPP)

#define MPL_VECTOR_HPP

#include <cstddef>
#include <vector>

namespace mpl {

  class counts : private std::vector<int> {
    typedef std::vector<int> base;
  public:
    typedef base::size_type size_type;
    explicit counts(size_type n=0) : base(n, 0) {
    }
    using base::operator[];
    using base::size;
    using base::push_back;
    const int * operator()() const {
      return base::data();
    }
  };

  //--------------------------------------------------------------------

  class displacements : private std::vector<int> {
    typedef std::vector<int> base;
  public:
    typedef base::size_type size_type;
    explicit displacements(size_type n=0) : base(n, 0) {
    }
    using base::operator[];
    using base::size;
    using base::push_back;
    const int * operator()() const {
      return base::data();
    }
  };

}

#endif
