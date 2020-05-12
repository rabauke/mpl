#if !(defined MPL_DISTRIBUTED_GRID)

#define MPL_DISTRIBUTED_GRID

#include <vector>
#include <mpl/cart_comm.hpp>
#include <mpl/layout.hpp>

namespace mpl {

  template<std::size_t dim, typename T, typename A>
  class distributed_grid;

  template<std::size_t dim, typename T, typename A>
  class local_grid;

  //--------------------------------------------------------------------

  template<std::size_t dim, typename T, typename A = std::allocator<T>>
  class distributed_grid {
  public:
    using vector_type = std::vector<T, A>;
    using value_type = typename vector_type::value_type;
    using allocator_type = typename vector_type::allocator_type;
    using size_type = std::ptrdiff_t;
    using difference_type = typename vector_type::difference_type;
    using reference = typename vector_type::reference;
    using const_reference = typename vector_type::const_reference;
    using pointer = typename vector_type::pointer;
    using const_pointer = typename vector_type::const_pointer;

  private:
    std::vector<size_type> gsize_, gbegin_, gend_, size_, oend_, overlap_;
    vector_type v;
    std::vector<subarray_layout<T>> left_mirror_layout_, right_mirror_layout_,
        left_border_layout_, right_border_layout_;
    subarray_layout<T> interior_layout_;

    size_type gbegin(size_type n, int comm_size, int comm_coord) const {
      return n * comm_coord / comm_size;
    }

    size_type gend(size_type n, int comm_size, int comm_coord) const {
      return n * (comm_coord + 1) / comm_size;
    }

  public:
    class sizes {
      std::vector<size_type> size_, overlap_;

    public:
      sizes(std::initializer_list<std::pair<size_type, size_type>> list) {
        for (const std::pair<size_type, size_type> &i : list)
          add(i.first, i.second);
      }

      void add(size_type size, size_type overlap) {
        size_.push_back(size);
        overlap_.push_back(overlap);
      }

      friend class distributed_grid;
    };

    distributed_grid(const cart_communicator &C, const sizes &size)
        : gsize_(size.size_),
          gbegin_(dim),
          gend_(dim),
          size_(dim),
          oend_(dim),
          overlap_(size.overlap_) {
#if defined MPL_DEBUG
      if (C.dim() != dim or gsize_.size() != dim or overlap_.size() != dim)
        throw invalid_dim();
#endif
      auto C_size = C.dims();
      auto C_coord = C.coords();
      size_type vol = 1;
      for (std::size_t i = 0; i < dim; ++i) {
        gbegin_[i] = gbegin(gsize_[i], C_size[i], C_coord[i]);
        gend_[i] = gend(gsize_[i], C_size[i], C_coord[i]);
        size_[i] = gend_[i] - gbegin_[i];
        oend_[i] = size_[i] + 2 * overlap_[i];
        vol *= size_[i] + 2 * overlap_[i];
      }
      v.resize(vol);
      for (std::size_t i = 0; i < dim; ++i) {
        typename subarray_layout<T>::parameter par_m_l, par_m_r, par_b_l, par_b_r;
        for (std::size_t j = dim - 1; true; --j) {
          if (j == i) {
            par_m_l.add(oend_[j], overlap_[j], 0);
            par_m_r.add(oend_[j], overlap_[j], size_[j] + overlap_[j]);
            par_b_l.add(oend_[j], overlap_[j], overlap_[j]);
            par_b_r.add(oend_[j], overlap_[j], size_[j]);
          } else {
            par_m_l.add(oend_[j], size_[j], overlap_[j]);
            par_m_r.add(oend_[j], size_[j], overlap_[j]);
            par_b_l.add(oend_[j], size_[j], overlap_[j]);
            par_b_r.add(oend_[j], size_[j], overlap_[j]);
          }
          if (j == 0)
            break;
        }
        left_mirror_layout_.push_back(subarray_layout<T>(par_m_l));
        right_mirror_layout_.push_back(subarray_layout<T>(par_m_r));
        left_border_layout_.push_back(subarray_layout<T>(par_b_l));
        right_border_layout_.push_back(subarray_layout<T>(par_b_r));
        typename subarray_layout<T>::parameter par_i;
        for (std::size_t j = dim - 1; true; --j) {
          par_i.add(oend_[j], size_[j], overlap_[j]);
          if (j == 0)
            break;
        }
        interior_layout_ = subarray_layout<T>(par_i);
      }
    }

    size_type gsize(size_type d) const { return gsize_[d]; }

    size_type gbegin(size_type d) const { return gbegin_[d]; };

    size_type gend(size_type d) const { return gend_[d]; };

    size_type size(size_type d) const { return size_[d]; }

    size_type begin(size_type d) const { return overlap_[d]; };

    size_type end(size_type d) const { return size_[d] + overlap_[d]; };

    size_type obegin(size_type d) const { return 0; };

    size_type oend(size_type d) const { return oend_[d]; };

    size_type gindex(size_type d, size_type i) const { return gbegin(d) + i - begin(d); }

    reference operator()(size_type x) {
      static_assert(dim == 1, "invalid dimension");
      return v[x];
    }

    const_reference operator()(size_type x) const {
      static_assert(dim == 1, "invalid dimension");
      return v[x];
    }

    reference operator()(size_type x, size_type y) {
      static_assert(dim == 2, "invalid dimension");
      return v[x + oend_[0] * y];
    }

    const_reference operator()(size_type x, size_type y) const {
      static_assert(dim == 2, "invalid dimension");
      return v[x + oend_[0] * y];
    }

    reference operator()(size_type x, size_type y, size_type z) {
      static_assert(dim == 3, "invalid dimension");
      return v[x + oend_[0] * (y + oend_[1] * z)];
    }

    const_reference operator()(size_type x, size_type y, size_type z) const {
      static_assert(dim == 3, "invalid dimension");
      return v[x + oend_[0] * (y + oend_[1] * z)];
    }

    pointer data() { return v.data(); }

    const_pointer data() const { return v.data(); }

    const subarray_layout<T> &left_mirror_layout(size_type i) const {
      return left_mirror_layout_[i];
    }

    const subarray_layout<T> &right_mirror_layout(size_type i) const {
      return right_mirror_layout_[i];
    }

    const subarray_layout<T> &left_border_layout(size_type i) const {
      return left_border_layout_[i];
    }

    const subarray_layout<T> &right_border_layout(size_type i) const {
      return right_border_layout_[i];
    }

    const subarray_layout<T> &interior_layout() const { return interior_layout_; }

    void swap(distributed_grid<dim, T, A> &other) {
      gsize_.swap(other.gsize_);
      gbegin_.swap(other.gbegin_);
      gend_.swap(other.gend_);
      size_.swap(other.size_);
      oend_.swap(other.oend_);
      overlap_.swap(other.overlap_);
      v.swap(other.v);
      left_mirror_layout_.swap(other.left_mirror_layout_);
      right_mirror_layout_.swap(other.right_mirror_layout_);
      left_border_layout_.swap(other.left_border_layout_);
      right_border_layout_.swap(other.right_border_layout_);
      interior_layout_.swap(other.interior_layout_);
    }
  };

  //--------------------------------------------------------------------

  template<std::size_t dim, typename T, typename A = std::allocator<T>>
  class local_grid {
  public:
    using vector_type = std::vector<T, A>;
    using value_type = typename vector_type::value_type;
    using allocator_type = typename vector_type::allocator_type;
    using size_type = std::ptrdiff_t;
    using difference_type = typename vector_type::difference_type;
    using reference = typename vector_type::reference;
    using const_reference = typename vector_type::const_reference;
    using pointer = typename vector_type::pointer;
    using const_pointer = typename vector_type::const_pointer;

  private:
    std::vector<size_type> gsize_;
    vector_type v;
    layouts<T> sub_layout_;

    size_type gbegin(size_type n, int comm_size, int comm_coord) const {
      return n * comm_coord / comm_size;
    }

    size_type gend(size_type n, int comm_size, int comm_coord) const {
      return n * (comm_coord + 1) / comm_size;
    }

  public:
    class sizes {
      std::vector<size_type> size_;

    public:
      sizes(std::initializer_list<size_type> list) {
        for (const size_type &i : list)
          add(i);
      }

      void add(size_type size) { size_.push_back(size); }

      friend class local_grid;
    };

    local_grid(const cart_communicator &C, const sizes &size) : gsize_(size.size_) {
#if defined MPL_DEBUG
      if (C.dim() != dim or gsize_.size() != dim)
        throw invalid_dim();
#endif
      size_type vol = 1;
      for (std::size_t i = 0; i < dim; ++i)
        vol *= gsize_[i];
      v.resize(vol);
      auto C_size = C.dims();
      for (int i = 0, i_end = C.size(); i < i_end; ++i) {
        auto coords = C.coords(i);
        typename subarray_layout<T>::parameter par;
        for (std::size_t j = dim - 1; true; --j) {
          const auto begin = gbegin(gsize_[j], C_size[j], coords[j]);
          const auto end = gend(gsize_[j], C_size[j], coords[j]);
          par.add(gsize_[j], end - begin, begin);
          if (j == 0)
            break;
        }
        sub_layout_.push_back(subarray_layout<T>(par));
      }
    }

    size_type size(size_type d) const { return gsize_[d]; }

    size_type begin(size_type d) const { return 0; };

    size_type end(size_type d) const { return gsize_[d]; };

    const_reference operator()(size_type x) const {
      static_assert(dim == 1, "invalid dimension");
      return v[x];
    }

    reference operator()(size_type x, size_type y) {
      static_assert(dim == 2, "invalid dimension");
      return v[x + gsize_[0] * y];
    }

    const_reference operator()(size_type x, size_type y) const {
      static_assert(dim == 2, "invalid dimension");
      return v[x + gsize_[0] * y];
    }

    reference operator()(size_type x, size_type y, size_type z) {
      static_assert(dim == 3, "invalid dimension");
      return v[x + gsize_[0] * (y + gsize_[1] * z)];
    }

    const_reference operator()(size_type x, size_type y, size_type z) const {
      static_assert(dim == 3, "invalid dimension");
      return v[x + gsize_[0] * (y + gsize_[1] * z)];
    }

    pointer data() { return v.data(); }

    const_pointer data() const { return v.data(); }

    const layout<T> &sub_layout(size_type i) const { return sub_layout_[i]; }

    const layouts<T> &sub_layouts() const { return sub_layout_; }

    void swap(local_grid<dim, T, A> &other) {
      gsize_.swap(other.gsize_);
      v.swap(other.v);
      sub_layout_.swap(other.sub_layout_);
    }
  };

}  // namespace mpl

#endif
