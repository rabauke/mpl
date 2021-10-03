#if !(defined MPL_DISTRIBUTED_GRID_HPP)

#define MPL_DISTRIBUTED_GRID_HPP

#include <vector>
#include <mpl/cartesian_communicator.hpp>
#include <mpl/layout.hpp>

namespace mpl {

  template<std::size_t dim, typename T, typename A>
  class distributed_grid;

  template<std::size_t dim, typename T, typename A>
  class local_grid;

  //--------------------------------------------------------------------

  /// \brief Local portion of a distributed data grid including local overlap data.
  /// \param dim number of dimensions of the data grid, may be 1, 2, 3 or 4
  /// \tparam T data type that is hold at each grid point
  /// \tparam A memory allocator
  template<std::size_t dim, typename T, typename A = std::allocator<T>>
  class distributed_grid {
  public:
    /// \brief the underlying one-dimensional vector type
    using vector_type = std::vector<T, A>;
    /// \brief data type that is hold at each grid point
    using value_type = typename vector_type::value_type;
    /// \brief memory allocator
    using allocator_type = typename vector_type::allocator_type;
    /// \brief type used for indexing
    /// \note This is a signed integer type (unlike for STL containers).
    using size_type = std::ptrdiff_t;
    /// \brief signed integer type for iterator differences
    using difference_type = typename vector_type::difference_type;
    /// \brief reference to value_type
    using reference = typename vector_type::reference;
    /// \brief const reference to value_type
    using const_reference = typename vector_type::const_reference;
    /// \brief pointer to value_type
    using pointer = typename vector_type::pointer;
    /// \brief const pointer to value_type
    using const_pointer = typename vector_type::const_pointer;

  private:
    std::array<size_type, dim> global_size_;
    std::array<size_type, dim> global_begin_;
    std::array<size_type, dim> global_end_;
    std::array<size_type, dim> size_;
    std::array<size_type, dim> overlap_end_;
    std::array<size_type, dim> overlap_;
    vector_type v_;
    std::vector<subarray_layout<T>> left_mirror_layout_, right_mirror_layout_,
        left_border_layout_, right_border_layout_;
    subarray_layout<T> interior_layout_;

    [[nodiscard]] size_type global_begin(size_type n, int comm_size, int comm_coord) const {
      return n * comm_coord / comm_size;
    }

    [[nodiscard]] size_type global_end(size_type n, int comm_size, int comm_coord) const {
      return n * (comm_coord + 1) / comm_size;
    }

  public:
    /// \brief Pair of grid size and overlap size.
    class size_overlap_pair {
    public:
      /// \brief indicates the total/global number data pints that the data grid holds along a
      /// dimension
      size_type size{0};
      /// \brief indicates the number of overlap data points the each local process holds as a
      /// copy of data hold by adjacent processes
      size_type overlap{0};
      /// \brief Creates a size-overlap pair.
      size_overlap_pair(size_type size, size_type overlap) : size{size}, overlap{overlap} {}
    };


    /// \brief Characterizes the dimensionality, total size and overlap of a distributed data
    /// grid.
    class dimensions {
      std::array<size_overlap_pair, dim> size_overlap_;

    public:
      using value_type = size_overlap_pair;
      using reference = size_overlap_pair &;
      using const_reference = const size_overlap_pair &;
      using iterator = typename std::array<size_overlap_pair, dim>::iterator;
      using const_iterator = typename std::array<size_overlap_pair, dim>::const_iterator;

      dimensions(const size_overlap_pair &size_0) : size_overlap_{size_0} {
        static_assert(dim == 1, "invalid number of arguments");
      }
      dimensions(const size_overlap_pair &size_0, const size_overlap_pair &size_1)
          : size_overlap_{size_0, size_1} {
        static_assert(dim == 2, "invalid number of arguments");
      }
      dimensions(const size_overlap_pair &size_0, const size_overlap_pair &size_1,
                 const size_overlap_pair &size_2)
          : size_overlap_{size_0, size_1, size_2} {
        static_assert(dim == 3, "invalid number of arguments");
      }
      dimensions(const size_overlap_pair &size_0, const size_overlap_pair &size_1,
                 const size_overlap_pair &size_2, const size_overlap_pair &size_3)
          : size_overlap_{size_0, size_1, size_2, size_3} {
        static_assert(dim == 4, "invalid number of arguments");
      }

      /// \brief Determines the dimensionality.
      /// \return dimensionality (number of dimensions)
      [[nodiscard]] size_type dimensionality() const {
        return static_cast<size_type>(size_overlap_.size());
      }

      /// \brief Determines the total size of a dimension.
      /// \param dimension the rank of the dimension
      /// \return the total size of the dimension
      [[nodiscard]] size_type size(size_type dimension) const {
        return size_overlap_[dimension].size;
      }

      /// \brief Determines the overlap size of a dimension.
      /// \param dimension the rank of the dimension
      /// \return the overlap of the dimension
      [[nodiscard]] size_type overlap(size_type dimension) const {
        return size_overlap_[dimension].overlap;
      }

      /// \brief Determines then total size along a dimension and the overlap of a
      /// dimension.
      /// \param dimension the rank of the dimension
      /// \return the size and the overlap
      [[nodiscard]] const_reference operator[](size_type dimension) const {
        return size_overlap_[dimension];
      }

      /// \brief Determines then total size along a dimension and the overlap of a
      /// dimension.
      /// \param dimension the rank of the dimension
      /// \return the size and the overlap
      [[nodiscard]] reference operator[](size_type dimension) {
        return size_overlap_[dimension];
      }

      [[nodiscard]] iterator begin() { return size_overlap_.begin(); }
      [[nodiscard]] const_iterator begin() const { return size_overlap_.begin(); }
      [[nodiscard]] const_iterator cbegin() const { return size_overlap_.cbegin(); }
      [[nodiscard]] iterator end() { return size_overlap_.end(); }
      [[nodiscard]] const_iterator end() const { return size_overlap_.end(); }
      [[nodiscard]] const_iterator cend() const { return size_overlap_.xend(); }

      friend class distributed_grid;
    };

    /// \brief Creates the local portion of a distributed data grid.
    /// \param communicator Cartesian communicator that wil be employed to update overlap data
    /// between adjacent processes.
    /// \param dims size and overlap data of the global grid
    /// \note The number of dimensions of the Cartesian communicator and the size of the dims
    /// parameter must be equal.
    explicit distributed_grid(const cartesian_communicator &communicator,
                              const dimensions &dims) {
#if defined MPL_DEBUG
      if (communicator.dimensionality() != dim or global_size_.size() != dim or
          overlap_.size() != dim)
        throw invalid_dim();
#endif
      const auto c_dimensions{communicator.get_dimensions()};
      const auto c_coord{communicator.coordinates()};
      size_type volume{1};
      for (std::size_t i{0}; i < dim; ++i) {
        global_size_[i] = dims[i].size;
        overlap_[i] = dims[i].overlap;
        global_begin_[i] = global_begin(global_size_[i], c_dimensions.size(i), c_coord[i]);
        global_end_[i] = global_end(global_size_[i], c_dimensions.size(i), c_coord[i]);
        size_[i] = global_end_[i] - global_begin_[i];
        overlap_end_[i] = size_[i] + 2 * overlap_[i];
        volume *= size_[i] + 2 * overlap_[i];
      }
      v_.resize(volume);
      left_mirror_layout_.reserve(dim);
      right_mirror_layout_.reserve(dim);
      left_border_layout_.reserve(dim);
      right_border_layout_.reserve(dim);
      for (std::size_t i{0}; i < dim; ++i) {
        typename subarray_layout<T>::parameter par_m_l, par_m_r, par_b_l, par_b_r;
        for (std::size_t j{dim - 1}; true; --j) {
          if (j == i) {
            par_m_l.add(overlap_end_[j], overlap_[j], 0);
            par_m_r.add(overlap_end_[j], overlap_[j], size_[j] + overlap_[j]);
            par_b_l.add(overlap_end_[j], overlap_[j], overlap_[j]);
            par_b_r.add(overlap_end_[j], overlap_[j], size_[j]);
          } else {
            par_m_l.add(overlap_end_[j], size_[j], overlap_[j]);
            par_m_r.add(overlap_end_[j], size_[j], overlap_[j]);
            par_b_l.add(overlap_end_[j], size_[j], overlap_[j]);
            par_b_r.add(overlap_end_[j], size_[j], overlap_[j]);
          }
          if (j == 0)
            break;
        }
        left_mirror_layout_.push_back(subarray_layout<T>(par_m_l));
        right_mirror_layout_.push_back(subarray_layout<T>(par_m_r));
        left_border_layout_.push_back(subarray_layout<T>(par_b_l));
        right_border_layout_.push_back(subarray_layout<T>(par_b_r));
        typename subarray_layout<T>::parameter par_i;
        for (std::size_t j{dim - 1}; true; --j) {
          par_i.add(overlap_end_[j], size_[j], overlap_[j]);
          if (j == 0)
            break;
        }
        interior_layout_ = subarray_layout<T>(par_i);
      }
    }

    /// \brief Determines the global size of the grid along a dimension.
    /// \param d dimension
    /// \return global size
    [[nodiscard]] size_type gsize(size_type d) const { return global_size_[d]; }

    /// \brief Determines the smallest index into the global distributed grid to the local
    /// portion of the grid.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type gbegin(size_type d) const { return global_begin_[d]; };

    /// \brief Determines the smallest index into the global distributed grid that is beyond the
    /// local portion of the grid.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type gend(size_type d) const { return global_end_[d]; };

    /// \brief Determines the size of the local portion of the distributed data grid along a
    /// dimension.
    /// \param d dimension
    /// \return local grid size
    [[nodiscard]] size_type size(size_type d) const { return size_[d]; }

    /// \brief Determines the lowest index to access the local portion of the distributed data
    /// grid along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type begin(size_type d) const { return overlap_[d]; };

    /// \brief Determines the last index (plus one) to access the local portion of the
    /// distributed data grid along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type end(size_type d) const { return size_[d] + overlap_[d]; };

    /// \brief Determines the lowest index to access the local portion of the distributed grid
    /// including the overlap data along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type obegin(size_type d) const { return 0; };

    /// \brief Determines the last index (plus one) to access the local portion of the
    /// distributed grid including the overlap data along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type oend(size_type d) const { return overlap_end_[d]; };

    /// \brief Translates an index to access the local portion of the distributed data grid into
    /// an index into the global grid.
    /// \param d dimension
    /// \param i index with respect to local data
    /// \return grid index
    [[nodiscard]] size_type gindex(size_type d, size_type i) const {
      return gbegin(d) + i - begin(d);
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 1, reference> operator()(size_type i_0) {
      static_assert(dim == 1, "invalid dimension");
      return v_[i_0];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 1, const_reference> operator()(size_type i_0) const {
      static_assert(dim == 1, "invalid dimension");
      return v_[i_0];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 2, reference> operator()(size_type i_0, size_type i_1) {
      static_assert(dim == 2, "invalid dimension");
      return v_[i_0 + overlap_end_[0] * i_1];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 2, const_reference> operator()(size_type i_0,
                                                                       size_type i_1) const {
      static_assert(dim == 2, "invalid dimension");
      return v_[i_0 + overlap_end_[0] * i_1];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 3, reference> operator()(size_type i_0, size_type i_1,
                                                                 size_type i_2) {
      static_assert(dim == 3, "invalid dimension");
      return v_[i_0 + overlap_end_[0] * (i_1 + overlap_end_[1] * i_2)];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 3, const_reference> operator()(size_type i_0,
                                                                       size_type i_1,
                                                                       size_type i_2) const {
      static_assert(dim == 3, "invalid dimension");
      return v_[i_0 + overlap_end_[0] * (i_1 + overlap_end_[1] * i_2)];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \param i_3 4th dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 4, reference> operator()(size_type i_0, size_type i_1,
                                                                 size_type i_2, size_type i_3) {
      static_assert(dim == 4, "invalid dimension");
      return v_[i_0 +
                overlap_end_[0] * (i_1 + overlap_end_[1] * (i_2 + overlap_end_[2] * i_3))];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \param i_3 4th dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 4, const_reference> operator()(size_type i_0,
                                                                       size_type i_1,
                                                                       size_type i_2,
                                                                       size_type i_3) const {
      static_assert(dim == 4, "invalid dimension");
      return v_[i_0 +
                overlap_end_[0] * (i_1 + overlap_end_[1] * (i_2 + overlap_end_[2] * i_3))];
    }

    /// \brief Grands access to the flattened grid data.
    /// \return pointer to grid data
    [[nodiscard]] pointer data() { return v_.data(); }

    /// \brief Grands access to the flattened grid data.
    /// \return pointer to grid data
    [[nodiscard]] const_pointer data() const { return v_.data(); }

    /// \brief Get the memory layout for receiving data when updating data in overlap regions
    /// along a given dimension.
    /// \param d dimension
    /// \return memory layout
    /// \details The returned memory layout represents data in the left overlap region.
    [[nodiscard]] const subarray_layout<T> &left_mirror_layout(size_type d) const {
      return left_mirror_layout_[d];
    }

    /// \brief Get the memory layout for receiving data when updating data in overlap regions
    /// along a given dimension.
    /// \param d dimension
    /// \return memory layout
    /// \details The returned memory layout represents data in the right overlap region.
    [[nodiscard]] const subarray_layout<T> &right_mirror_layout(size_type d) const {
      return right_mirror_layout_[d];
    }

    /// \brief Get the memory layout for sending data when updating data in overlap regions
    /// along a given dimension.
    /// \param d dimension
    /// \return memory layout
    /// \details The returned memory layout represents data next to the left overlap region.
    const subarray_layout<T> &left_border_layout(size_type d) const {
      return left_border_layout_[d];
    }

    /// \brief Get the memory layout for sending data when updating data in overlap regions
    /// along a given dimension.
    /// \param d dimension
    /// \return memory layout
    /// \details The returned memory layout represents data next to the right overlap region.
    const subarray_layout<T> &right_border_layout(size_type d) const {
      return right_border_layout_[d];
    }

    /// \brief Get the memory layout for sending or receiving data without overlap data.
    /// \return memory layout
    /// \details The returned memory layout represents inner grid data without the overlap
    /// regions.
    const subarray_layout<T> &interior_layout() const { return interior_layout_; }

    /// \brief Swaps this distributed data grid with another.
    /// \param other other distributed data grid
    void swap(distributed_grid<dim, T, A> &other) {
      global_size_.swap(other.global_size_);
      global_begin_.swap(other.global_begin_);
      global_end_.swap(other.global_end_);
      size_.swap(other.size_);
      overlap_end_.swap(other.overlap_end_);
      overlap_.swap(other.overlap_);
      v_.swap(other.v_);
      left_mirror_layout_.swap(other.left_mirror_layout_);
      right_mirror_layout_.swap(other.right_mirror_layout_);
      left_border_layout_.swap(other.left_border_layout_);
      right_border_layout_.swap(other.right_border_layout_);
      interior_layout_.swap(other.interior_layout_);
    }
  };

  //--------------------------------------------------------------------

  /// \brief Data grid.
  /// \details This data structure is suitable for gather and scatter operation in combination
  /// with \ref distributed_grid.
  /// \param dim number of dimensions of the data grid, may be 1, 2, 3 or 4
  /// \tparam T data type that is hold at each grid point
  /// \tparam A memory allocator
  template<std::size_t dim, typename T, typename A = std::allocator<T>>
  class local_grid {
  public:
    /// \brief the underlying one-dimensional vector type
    using vector_type = std::vector<T, A>;
    /// \brief data type that is hold at each grid point
    using value_type = typename vector_type::value_type;
    /// \brief memory allocator
    using allocator_type = typename vector_type::allocator_type;
    /// \brief type used for indexing
    /// \note This is a signed integer type (unlike for STL containers).
    using size_type = std::ptrdiff_t;
    /// \brief signed integer type for iterator differences
    using difference_type = typename vector_type::difference_type;
    /// \brief reference to value_type
    using reference = typename vector_type::reference;
    /// \brief const reference to value_type
    using const_reference = typename vector_type::const_reference;
    /// \brief pointer to value_type
    using pointer = typename vector_type::pointer;
    /// \brief const pointer to value_type
    using const_pointer = typename vector_type::const_pointer;

  private:
    std::array<size_type, dim> global_size_;
    vector_type v_;
    layouts<T> sub_layouts_;

    [[nodiscard]] size_type global_begin(size_type n, int comm_size, int comm_coord) const {
      return n * comm_coord / comm_size;
    }

    [[nodiscard]] size_type global_end(size_type n, int comm_size, int comm_coord) const {
      return n * (comm_coord + 1) / comm_size;
    }

  public:
    /// \brief Characterizes the dimensionality and the total size of a local data grid.
    class dimensions {
      std::array<size_type, dim> size_;

    public:
      using value_type = size_type;
      using reference = size_type &;
      using const_reference = const size_type &;
      using iterator = typename std::array<size_type, dim>::iterator;
      using const_iterator = typename std::array<std::size_t, dim>::const_iterator;

      dimensions(const size_type &size_0) : size_{size_0} {
        static_assert(dim == 1, "invalid number of arguments");
      }
      dimensions(const size_type &size_0, const size_type &size_1) : size_{size_0, size_1} {
        static_assert(dim == 2, "invalid number of arguments");
      }
      dimensions(const size_type &size_0, const size_type &size_1, const size_type &size_2)
          : size_{size_0, size_1, size_2} {
        static_assert(dim == 3, "invalid number of arguments");
      }
      dimensions(const size_type &size_0, const size_type &size_1, const size_type &size_2,
                 const size_type &size_3)
          : size_{size_0, size_1, size_2, size_3} {
        static_assert(dim == 4, "invalid number of arguments");
      }
      /// \brief Determines the dimensionality.
      /// \return dimensionality (number of dimensions)
      [[nodiscard]] size_type dimensionality() const {
        return static_cast<size_type>(size_.size());
      }

      /// \brief Determines the total size of a dimension.
      /// \param dimension the rank of the dimension
      /// \return the total size of the dimension
      [[nodiscard]] size_type size(size_type dimension) const { return size_[dimension]; }

      /// \brief Determines then total size along a dimension of a dimension.
      /// \param dimension the rank of the dimension
      /// \return the size and the overlap
      [[nodiscard]] const_reference operator[](size_type dimension) const {
        return size_[dimension];
      }

      /// \brief Determines then total size along a dimension.
      /// \param dimension the rank of the dimension
      /// \return the size and the overlap
      [[nodiscard]] reference operator[](size_type dimension) { return size_[dimension]; }

      [[nodiscard]] iterator begin() { return size_.begin(); }
      [[nodiscard]] const_iterator begin() const { return size_.begin(); }
      [[nodiscard]] const_iterator cbegin() const { return size_.cbegin(); }
      [[nodiscard]] iterator end() { return size_.end(); }
      [[nodiscard]] const_iterator end() const { return size_.end(); }
      [[nodiscard]] const_iterator cend() const { return size_.xend(); }

      friend class local_grid;
    };

    /// \brief Creates a local data grid.
    /// \param communicator Cartesian communicator that will be employed to scatter or gather
    /// data.
    /// \param dims size of the global grid
    /// \note The number of dimensions of the Cartesian communicator and the size of the dims
    /// parameter must be equal.
    local_grid(const cartesian_communicator &communicator, const dimensions &dims)
        : global_size_(dims.size_) {
#if defined MPL_DEBUG
      if (communicator.dimensionality() != dim or global_size_.size() != dim)
        throw invalid_dim();
#endif
      size_type volume{1};
      for (std::size_t i{0}; i < dim; ++i)
        volume *= global_size_[i];
      v_.resize(volume);
      const auto c_dimensions{communicator.get_dimensions()};
      const auto c_size{communicator.size()};
      for (int i{0}, i_end{c_size}; i < i_end; ++i) {
        const auto coords{communicator.coordinates(i)};
        typename subarray_layout<T>::parameter par;
        for (std::size_t j{dim - 1}; true; --j) {
          const auto begin{global_begin(global_size_[j], c_dimensions.size(j), coords[j])};
          const auto end{global_end(global_size_[j], c_dimensions.size(j), coords[j])};
          par.add(global_size_[j], end - begin, begin);
          if (j == 0)
            break;
        }
        sub_layouts_.push_back(subarray_layout<T>(par));
      }
    }

    /// \brief Determines the size of the data grid along a dimension.
    /// \param d dimension
    /// \return local grid size
    [[nodiscard]] size_type size(size_type d) const { return global_size_[d]; }

    /// \brief Determines the lowest index to access the data grid along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type begin(size_type d) const { return 0; };

    /// \brief Determines the last index (plus one) to access the data grid along a dimension.
    /// \param d dimension
    /// \return grid index
    [[nodiscard]] size_type end(size_type d) const { return global_size_[d]; };

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 1, reference> operator()(size_type i_0) {
      static_assert(dim == 1, "invalid dimension");
      return v_[i_0];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 1, const_reference> operator()(size_type i_0) const {
      static_assert(dim == 1, "invalid dimension");
      return v_[i_0];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 2, reference> operator()(size_type i_0, size_type i_1) {
      static_assert(dim == 2, "invalid dimension");
      return v_[i_0 + global_size_[0] * i_1];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 2, const_reference> operator()(size_type i_0,
                                                                       size_type i_1) const {
      static_assert(dim == 2, "invalid dimension");
      return v_[i_0 + global_size_[0] * i_1];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 3, reference> operator()(size_type i_0, size_type i_1,
                                                                 size_type i_2) {
      static_assert(dim == 3, "invalid dimension");
      return v_[i_0 + global_size_[0] * (i_1 + global_size_[1] * i_2)];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 3, const_reference> operator()(size_type i_0,
                                                                       size_type i_1,
                                                                       size_type i_2) const {
      static_assert(dim == 3, "invalid dimension");
      return v_[i_0 + global_size_[0] * (i_1 + global_size_[1] * i_2)];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \param i_3 4th dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 4, reference> operator()(size_type i_0, size_type i_1,
                                                                 size_type i_2, size_type i_3) {
      static_assert(dim == 4, "invalid dimension");
      return v_[i_0 +
                global_size_[0] * (i_1 + global_size_[1] * (i_2 + global_size_[2] * i_3))];
    }

    /// \brief Element access.
    /// \param i_0 1st dimension index
    /// \param i_1 2nd dimension index
    /// \param i_2 3rd dimension index
    /// \param i_3 4th dimension index
    /// \return grid data element
    template<std::size_t d = dim>
    [[nodiscard]] std::enable_if_t<d == 4, const_reference> operator()(size_type i_0,
                                                                       size_type i_1,
                                                                       size_type i_2,
                                                                       size_type i_3) const {
      static_assert(dim == 4, "invalid dimension");
      return v_[i_0 +
                global_size_[0] * (i_1 + global_size_[1] * (i_2 + global_size_[2] * i_3))];
    }

    /// \brief Grands access to the flattened grid data.
    /// \return pointer to grid data
    [[nodiscard]] pointer data() { return v_.data(); }

    /// \brief Grands access to the flattened grid data.
    /// \return pointer to grid data
    [[nodiscard]] const_pointer data() const { return v_.data(); }

    /// \brief Get layouts for scatting and gathering of the grid data.
    /// \return set of layouts
    /// \details If there is a local_grid and a distributed_grid that have been created with the
    /// same Cartesian communicator argument and if both grids have the same total size then
    /// the i-th returned layout is suitable to send a sub-set of data from the local_grid to
    /// the distributed grid at the process with rank i in the Cartesian communicator.
    [[nodiscard]] const layouts<T> &sub_layouts() const { return sub_layouts_; }

    void swap(local_grid<dim, T, A> &other) {
      global_size_.swap(other.global_size_);
      v_.swap(other.v_);
      sub_layouts_.swap(other.sub_layouts_);
    }
  };

}  // namespace mpl

#endif
