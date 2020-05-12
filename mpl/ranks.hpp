#if !(defined MPL_RANKS_HPP)

#define MPL_RANKS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace mpl {

  class ranks : private std::vector<int> {
    using base = std::vector<int>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    explicit ranks(size_type n = 0) : base(n, 0) {}

    ranks(std::initializer_list<int> init) : base(init) {}

    ranks(const ranks &other) = default;

    ranks(ranks &&other) noexcept : base(std::move(other)) {}

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    const int *operator()() const { return base::data(); }

    int *operator()() { return base::data(); }
  };

}  // namespace mpl

#endif
