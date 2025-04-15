#if !(defined MPL_LAYOUT_HPP)

#define MPL_LAYOUT_HPP

#include <mpi.h>
#include <cstddef>
#include <iterator>
#include <initializer_list>
#include <type_traits>
#include <limits>
#include <utility>
#include <algorithm>
#include <vector>


namespace mpl {

  template<typename T>
  class layout;

  template<typename T>
  class null_layout;

  template<typename T>
  class empty_layout;

  template<typename T>
  class contiguous_layout;

  template<typename T>
  class vector_layout;

  template<typename T>
  class strided_vector_layout;

  template<typename T>
  class indexed_layout;

  template<typename T>
  class hindexed_layout;

  template<typename T>
  class indexed_block_layout;

  template<typename T>
  class hindexed_block_layout;

  template<typename T>
  class iterator_layout;

  template<typename T>
  class subarray_layout;

  class heterogeneous_layout;

  template<typename T>
  class absolute_data;

  template<typename T>
  absolute_data<T *> make_absolute(T *x, const layout<T> &l);

  template<typename T>
  absolute_data<const T *> make_absolute(const T *x, const layout<T> &l);

  template<typename T>
  class contiguous_layouts;

  namespace impl {

    class base_communicator;
    class topology_communicator;

  }  // namespace impl

  //--------------------------------------------------------------------

  /// Base class for a family of classes that describe where objects are located in
  /// memory when several objects of the same type T are exchanged in a single message.
  /// \tparam T type of the objects that the layout refers to (the base element type)
  template<typename T>
  class layout {
  private:
    MPI_Datatype type_{MPI_DATATYPE_NULL};

  protected:
    explicit layout(MPI_Datatype new_type) : type_{new_type} {
      if (type_ != MPI_DATATYPE_NULL)
        MPI_Type_commit(&type_);
    }

  public:
    /// Default constructor creates a layout of zero objects.
    layout() = default;

    /// Copy constructor creates a new layout that describes the same memory layout as
    /// the other one.
    /// \param l the layout to copy from
    layout(const layout &l) {
      if (l.type_ != MPI_DATATYPE_NULL)
        MPI_Type_dup(l.type_, &type_);
    }

    /// Move constructor creates a new layout that describes the same memory layout as
    /// the other one.
    /// \param l the layout to move from
    layout(layout &&l) noexcept : type_{l.type_} {
      l.type_ = MPI_DATATYPE_NULL;
    }

    /// Copy assignment operator creates a new layout that describes the same memory
    /// layout as the other one.
    /// \param l the layout to copy from
    layout &operator=(const layout &l) {
      if (this != &l) {
        if (type_ != MPI_DATATYPE_NULL)
          MPI_Type_free(&type_);
        if (l.type_ != MPI_DATATYPE_NULL)
          MPI_Type_dup(l.type_, &type_);
        else
          type_ = MPI_DATATYPE_NULL;
      }
      return *this;
    }

    /// Move assignment operator creates a new layout that describes the same memory
    /// layout as the other one.
    /// \param l the layout to move from
    layout &operator=(layout &&l) noexcept {
      if (type_ != MPI_DATATYPE_NULL)
        MPI_Type_free(&type_);
      type_ = l.type_;
      l.type_ = MPI_DATATYPE_NULL;
      return *this;
    }

    /// Get the underlying MPI handle of the data type.
    /// \return MPI handle of the data type
    /// \note This function returns a non-owning handle to the underlying MPI data type, which
    /// may be useful when refactoring legacy MPI applications to MPL.
    /// \warning The handle must not be used to modify the MPI data type that the handle points
    /// to.
    [[nodiscard]] MPI_Datatype native_handle() const {
      return type_;
    }

    /// Get the byte extent of the layout.
    /// \return the extent in bytes
    /// \note The extent of a layout correspondents to the extent of the underlying MPI
    /// datatype.  See MPI documentation for details.
    /// \see \c extent
    [[nodiscard]] ssize_t byte_extent() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_;
    }

    /// Get the byte lower bound of the layout.
    /// \return the lower bound in bytes
    /// \note The lower bound of a layout correspondents to the lower bound of the underlying
    /// MPI datatype.  See MPI documentation for details.
    /// \see \c byte_upper_bound, \c lower_bound
    [[nodiscard]] ssize_t byte_lower_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return lb_;
    }

    /// Get the byte upper bound of the layout.
    /// \return the upper bound in bytes
    /// \note The upper bound of a layout correspondents to the upper bound of the underlying
    /// MPI datatype.  See MPI documentation for details.
    /// \see \c byte_lower_bound, \c upper_bound
    [[nodiscard]] ssize_t byte_upper_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_ - lb_;
    }

    /// Get the extent of the layout.
    /// \return the extent
    /// \note The extent of a layout correspondents to the extent of the underlying MPI
    /// datatype.  See MPI documentation for details.
    /// \see \c byte_extent
    [[nodiscard]] ssize_t extent() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{byte_extent()};
      if (static_cast<ssize_t>(res / sizeof(T) * sizeof(T)) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Get the lower bound of the layout.
    /// \return the lower bound
    /// \note The lower bound of a layout correspondents to the lower bound of the underlying
    /// MPI datatype.  See MPI documentation for details.
    /// \see \c byte_lower_bound, \c upper_bound
    [[nodiscard]] ssize_t lower_bound() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{byte_lower_bound()};
      if (static_cast<ssize_t>(res / sizeof(T) * sizeof(T)) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Get the upper bound of the layout.
    /// \return the upper bound in bytes
    /// \note The upper bound of a layout correspondents to the upper bound of the underlying
    /// MPI datatype.  See MPI documentation for details.
    /// \see \c byte_upper_bound, \c lower_bound
    [[nodiscard]] ssize_t upper_bound() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{byte_upper_bound()};
      if (static_cast<ssize_t>(res / sizeof(T) * sizeof(T)) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Get the true byte extent of the layout.
    /// \return the true extent in bytes
    /// \note The true extent of a layout correspondents to the extent of the underlying MPI
    /// datatype.  See MPI documentation for details.
    /// \see \c true_extent
    [[nodiscard]] ssize_t true_byte_extent() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_;
    }

    /// Get the true byte lower bound of the layout.
    /// \return the true lower bound in bytes
    /// \note The true lower bound of a layout correspondents to the lower bound of the
    /// underlying MPI datatype.  See MPI documentation for details.
    /// \see \c true_byte_upper_bound, \c true_lower_bound
    [[nodiscard]] ssize_t true_byte_lower_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return lb_;
    }

    /// Get the true byte upper bound of the layout.
    /// \return the true upper bound in bytes
    /// \note The true upper bound of a layout correspondents to the upper bound of the
    /// underlying MPI datatype.  See MPI documentation for details.
    /// \see \c true_byte_lower_bound, \c true_upper_bound
    [[nodiscard]] ssize_t true_byte_upper_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type_, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_ - lb_;
    }

    /// Get the true extent of the layout.
    /// \return the true extent
    /// \note The true extent of a layout correspondents to the extent of the underlying MPI
    /// datatype.  See MPI documentation for details.
    /// \see \c true_byte_extent
    [[nodiscard]] ssize_t true_extent() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{true_byte_extent()};
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Get the true lower bound of the layout.
    /// \return the true lower bound
    /// \note The true lower bound of a layout correspondents to the lower bound of the
    /// underlying MPI datatype.  See MPI documentation for details.
    /// \see \c true_byte_lower_bound, \c true_upper_bound
    [[nodiscard]] ssize_t true_lower_bound() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{true_byte_lower_bound()};
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Get the true upper bound of the layout.
    /// \return the true upper bound in bytes
    /// \note The true upper bound of a layout correspondents to the upper bound of the
    /// underlying MPI datatype.  See MPI documentation for details.
    /// \see \c true_byte_upper_bound, \c true_lower_bound
    [[nodiscard]] ssize_t true_upper_bound() const {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      const ssize_t res{true_byte_upper_bound()};
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    /// Resize the layout.
    /// \param lb the layout's new true byte lower bound
    /// \param extent the layout's new true byte extent
    void byte_resize(ssize_t lb, ssize_t extent) {
      if (type_ != MPI_DATATYPE_NULL) {
        MPI_Datatype newtype;
        MPI_Type_create_resized(type_, lb, extent, &newtype);
        MPI_Type_commit(&newtype);
        MPI_Type_free(&type_);
        type_ = newtype;
      }
    }

    /// Resize the layout.
    /// \param lb the layout's new true lower bound
    /// \param extent the layout's new true extent
    void resize(ssize_t lb, ssize_t extent) {
      static_assert(not std::is_void_v<T>, "layout type must be for non-void type");
      byte_resize(static_cast<ssize_t>(sizeof(T)) * lb,
                  static_cast<ssize_t>(sizeof(T)) * extent);
    }

    /// Swap with other layout.
    /// \param l other layout
    void swap(layout &l) noexcept {
      std::swap(type_, l.type_);
    }

    /// Destroy layout.
    ~layout() {
      if (type_ != MPI_DATATYPE_NULL)
        MPI_Type_free(&type_);
    }

    friend class detail::datatype_traits<layout>;

    friend class null_layout<T>;

    friend class empty_layout<T>;

    friend class contiguous_layout<T>;

    friend class vector_layout<T>;

    friend class strided_vector_layout<T>;

    friend class indexed_layout<T>;

    friend class hindexed_layout<T>;

    friend class indexed_block_layout<T>;

    friend class hindexed_block_layout<T>;

    friend class iterator_layout<T>;

    friend class subarray_layout<T>;

    friend class heterogeneous_layout;

    friend absolute_data<T *> make_absolute<>(T *, const layout &);

    friend absolute_data<const T *> make_absolute<>(const T *, const layout &);
  };

  //--------------------------------------------------------------------

  /// Layout with zero elements.
  /// \tparam T base element type
  /// \note This type corresponds to the MPI datatype \c MPI_DATATYPE_NULL.  The template
  /// parameter \c T is required for syntactical reasons but does not affect the class' behaviour.
  /// \see inherits all member methods of \c layout
  template<typename T>
  class null_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// default constructor
    null_layout() noexcept : layout<T>(MPI_DATATYPE_NULL) {
    }

    /// copy constructor
    null_layout([[maybe_unused]] const null_layout &l) noexcept : null_layout() {
    }

    null_layout([[maybe_unused]] null_layout &&l) noexcept : null_layout() {
    }

    /// swap two instances of null_layout
    /// \note This a no-op, as all instances of \c null_layout are equal.
    void swap([[maybe_unused]] null_layout &other) noexcept {
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout with zero elements.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class empty_layout : public layout<T> {
    using layout<T>::type_;

    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

  public:
    /// default constructor
    empty_layout() : layout<T>(build()) {
    }

    /// copy constructor
    /// \param l layout to copy from
    empty_layout(const empty_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    empty_layout(empty_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    empty_layout &operator=(const empty_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    empty_layout &operator=(empty_layout &&l) noexcept = default;

    /// exchanges two empty layouts
    /// \param other the layout to swap with
    void swap(empty_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
  };

  //--------------------------------------------------------------------

  /// Layout representing contiguous storage several objects.
  /// \tparam T base element type
  /// \note Both types \c contiguous_layout and \c vector_layout represent contiguous
  /// storage.  \c contiguous_layout implements some additional bookkeeping as one important
  /// difference between both classes.  The class \c vector_layout is slightly more flexible
  /// and should be used to represent contiguous storage unless the MPL library _requires_ the
  /// usage of \c contiguous_layout, e.g., in <tt>communicator::reduce</tt>.
  /// \see inherits all member methods of \c layout, \c vector_layout, \c contiguous_layouts
  template<typename T>
  class contiguous_layout : public layout<T> {
    using layout<T>::type_;

    static MPI_Datatype build(
        size_t count, MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      if (count <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        MPI_Type_contiguous(static_cast<int>(count), old_type, &new_type);
      } else {
        const size_t modulus{std::numeric_limits<int>::max()};
        const size_t count_1{count / modulus};
        const size_t count_0{count - count_1 * modulus};
        MPI_Count lb, extent;
        MPI_Type_get_extent_x(old_type, &lb, &extent);
        MPI_Datatype type_modulus;
        MPI_Type_contiguous(static_cast<int>(modulus), old_type, &type_modulus);
        std::vector<int> block_lengths{static_cast<int>(count_0), static_cast<int>(count_1)};
#if defined MPL_DEBUG
        if (count_0 * extent > static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()))
          throw invalid_size();
#endif
        std::vector<MPI_Aint> displacements{0, static_cast<MPI_Aint>(count_0 * extent)};
        std::vector<MPI_Datatype> types{old_type, type_modulus};
        MPI_Type_create_struct(2, block_lengths.data(), displacements.data(), types.data(),
                               &new_type);
      }
      return new_type;
    }

    size_t count_;

    [[nodiscard]] size_t size() const {
      return count_;
    }

  public:
    /// constructs layout for contiguous storage several objects of type T
    /// \param count number of objects
    explicit contiguous_layout(size_t count = 0) : layout<T>(build(count)), count_(count) {
    }

    /// constructs layout for data with memory layout that is a homogenous sequence of
    /// some other contiguous layout
    /// \param count number of layouts in sequence
    /// \param l the layout of a single element
    explicit contiguous_layout(size_t count, const contiguous_layout &l)
        : layout<T>(build(count, l.type_)), count_(l.count_ * count) {
    }

    /// copy constructor
    /// \param l layout to copy from
    contiguous_layout(const contiguous_layout &l) : layout<T>(l), count_(l.count_) {
    }

    /// move constructor
    /// \param l layout to move from
    contiguous_layout(contiguous_layout &&l) noexcept
        : layout<T>(std::move(l)), count_(l.count_) {
      l.count_ = 0;
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    contiguous_layout &operator=(const contiguous_layout &l) {
      if (&l != this) {
        layout<T>::operator=(l);
        count_ = l.count_;
      }
      return *this;
    }

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    contiguous_layout &operator=(contiguous_layout &&l) noexcept {
      if (this != &l) {
        layout<T>::operator=(std::move(l));
        count_ = l.count_;
        l.count_ = 0;
      }
      return *this;
    }

    /// exchanges two contiguous layouts
    /// \param other the layout to swap with
    void swap(contiguous_layout&other) noexcept {
      std::swap(type_, other.type_);
      std::swap(count_, other.count_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;

    friend class impl::base_communicator;
    friend class communicator;
    friend class inter_communicator;
    friend class contiguous_layouts<T>;
  };

  //--------------------------------------------------------------------

  /// Layout representing contiguous storage several objects.
  /// \tparam T base element type
  /// \note Both types \ref contiguous_layout and \c vector_layout represent contiguous
  /// storage.  \c contiguous_layout implements some additional bookkeeping as one important
  /// difference between both classes.  The class \c vector_layout is slightly more flexible
  /// and should be used to represent contiguous storage unless the MPL library _requires_ the
  /// usage of \c contiguous_layout, e.g., in <tt>communicator::reduce</tt>.
  /// \see inherits all member methods of \c layout, \c contiguous_layout
  template<typename T>
  class vector_layout : public layout<T> {
    using layout<T>::type_;

    static MPI_Datatype build(
        size_t count, MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      if (count <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        MPI_Type_contiguous(static_cast<int>(count), old_type, &new_type);
      } else {
        const size_t modulus{std::numeric_limits<int>::max()};
        const size_t count_1{count / modulus};
        const size_t count_0{count - count_1 * modulus};
        MPI_Count lb, extent;
        MPI_Type_get_extent_x(old_type, &lb, &extent);
        MPI_Datatype type_modulus;
        MPI_Type_contiguous(static_cast<int>(modulus), old_type, &type_modulus);
        std::vector<int> block_lengths{static_cast<int>(count_0), static_cast<int>(count_1)};
#if defined MPL_DEBUG
        if (count_0 * extent > static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()))
          throw invalid_size();
#endif
        std::vector<MPI_Aint> displacements{0, static_cast<MPI_Aint>(count_0 * extent)};
        std::vector<MPI_Datatype> types{old_type, type_modulus};
        MPI_Type_create_struct(2, block_lengths.data(), displacements.data(), types.data(),
                               &new_type);
      }
      return new_type;
    }

  public:
    /// constructs layout for contiguous storage several objects of type T
    /// \param count number of objects
    explicit vector_layout(size_t count = 0) : layout<T>(build(count)) {
    }

    /// constructs layout for data with memory layout that is a homogenous sequence of some
    /// other layout
    /// \param count number of layouts in sequence
    /// \param l the layout of a single element
    explicit vector_layout(size_t count, const layout<T> &l)
        : layout<T>(build(count, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    vector_layout(const vector_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    vector_layout(vector_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    vector_layout &operator=(const vector_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    vector_layout &operator=(vector_layout &&l) noexcept = default;

    /// exchanges two contiguous layouts
    /// \param other the layout to swap with
    void swap(vector_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing uniform storage of several objects with a possibly non-unit stride
  /// between consecutive elements.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class strided_vector_layout : public layout<T> {
    using layout<T>::type_;

    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        int count, int blocklength, int stride,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, stride, old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    strided_vector_layout() : layout<T>(build()) {
    }

    /// constructs a layout with several strided objects of type T
    /// \param count the number of blocks (non-negative)
    /// \param blocklength number of data elements in each block (non-negative)
    /// \param stride number or elements between start of each block
    explicit strided_vector_layout(int count, int blocklength, int stride)
        : layout<T>(build(count, blocklength, stride)) {
    }

    /// constructs a layout with several strided objects of some other layout
    /// \param count the number of blocks (non-negative)
    /// \param blocklength number of data elements in each block (non-negative)
    /// \param stride number or elements between start of each block, each element having the
    /// size given by the extend of the layout given by parameter \c l
    /// \param l the layout of a single element in each block
    explicit strided_vector_layout(int count, int blocklength, int stride, const layout<T> &l)
        : layout<T>(build(count, blocklength, stride, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    strided_vector_layout(const strided_vector_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    strided_vector_layout(strided_vector_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    strided_vector_layout &operator=(const strided_vector_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    strided_vector_layout &operator=(strided_vector_layout &&l) noexcept = default;

    /// exchanges two contiguous layouts
    /// \param other the layout to swap with
    void swap(strided_vector_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing data in a sequence of consecutive homogenous blocks of varying
  /// lengths.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class indexed_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize an indexed layout.
    class parameter {
      std::vector<int> blocklengths, displacements;

    public:
      /// creates parameters for an indexed layout representing an empty sequence
      parameter() = default;

      /// converts a container into an indexed layout parameter
      /// \tparam List_T container type, must work with range-based for loops, value type must
      /// have two elements (the block length and the displacement), each convertible to int
      /// \param list container
      template<typename List_T>
      parameter(const List_T &list) {
        using std::get;
        for (const auto &i : list)
          add(get<0>(i), get<1>(i));
      }

      /// converts an initializer list into a parameters
      /// \param list initializer list with two-element tuples of block lengths and
      /// displacements
      parameter(std::initializer_list<std::tuple<int, int>> list) {
        using std::get;
        for (const std::tuple<int, int> &i : list)
          add(get<0>(i), get<1>(i));
      }

      /// add an additional block
      /// \param blocklength block length
      /// \param displacement displacement (relative to the beginning of the first block)
      void add(int blocklength, int displacement) {
        blocklengths.push_back(blocklength);
        displacements.push_back(displacement);
      }

      friend class indexed_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_indexed(par.displacements.size(), par.blocklengths.data(),
                       par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    indexed_layout() : layout<T>(build()) {
    }

    /// constructs indexed layout for data of type \c T
    /// \param par parameter containing information about the layout
    explicit indexed_layout(const parameter &par) : layout<T>(build(par)) {
    }

    /// constructs indexed layout for data with some other layout
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    explicit indexed_layout(const parameter &par, const layout<T> &l)
        : layout<T>(build(par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    indexed_layout(const indexed_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    indexed_layout(indexed_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    indexed_layout &operator=(const indexed_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    indexed_layout &operator=(indexed_layout &&l) noexcept = default;

    /// exchanges two indexed layouts
    /// \param other the layout to swap with
    void swap(indexed_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing data in a sequence of consecutive homogenous blocks of varying
  /// lengths.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class hindexed_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize a heterogeneously indexed
    /// layout.
    class parameter {
      std::vector<int> blocklengths;
      std::vector<MPI_Aint> displacements;

    public:
      /// creates parameters for an empty sequence
      parameter() = default;

      /// converts a container into a hindexed layout parameter
      /// \tparam List_T container type, must work with range-based for loops, value type must
      /// have two elements (the block length and the displacement), each convertible to int
      /// \param list container
      template<typename List_T>
      explicit parameter(const List_T &list) {
        using std::get;
        for (const auto &i : list)
          add(get<0>(i), get<1>(i));
      }

      /// converts an initializer list into a hindexed layout parameter
      /// \param list initializer list with two-element tuples of block lengths and
      /// displacements
      parameter(std::initializer_list<std::tuple<int, ssize_t>> list) {
        using std::get;
        for (const std::tuple<int, ssize_t> &i : list)
          add(get<0>(i), get<1>(i));
      }

      /// add an additional block
      /// \param blocklength block length
      /// \param displacement displacement (relative to the beginning of the first block)
      void add(int blocklength, ssize_t displacement) {
        static_assert(
            std::numeric_limits<MPI_Aint>::min() <= std::numeric_limits<ssize_t>::min() and
                std::numeric_limits<MPI_Aint>::max() >= std::numeric_limits<ssize_t>::max(),
            "MPI implementation with unusual MPI_Aint");
        blocklengths.push_back(blocklength);
        displacements.push_back(displacement);
      }

      friend class hindexed_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_hindexed(par.displacements.size(), par.blocklengths.data(),
                               par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    hindexed_layout() : layout<T>(build()) {
    }

    /// constructs heterogeneously indexed layout for data of type \c T
    /// \param par parameter containing information about the layout
    /// \note displacements are given in bytes
    explicit hindexed_layout(const parameter &par) : layout<T>(build(par)) {
    }

    /// constructs heterogeneously indexed layout for data with some other layout
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    /// \note displacements are given bytes
    explicit hindexed_layout(const parameter &par, const layout<T> &l)
        : layout<T>(build(par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    hindexed_layout(const hindexed_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    hindexed_layout(hindexed_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    hindexed_layout &operator=(const hindexed_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    hindexed_layout &operator=(hindexed_layout &&l) noexcept = default;

    /// exchanges two indexed layouts
    /// \param other the layout to swap with
    void swap(hindexed_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing data in a sequence of consecutive homogenous blocks of uniform
  /// lengths.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class indexed_block_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize an indexed layout.
    class parameter {
      std::vector<int> displacements;

    public:
      /// creates parameters for an indexed layout representing an empty sequence
      parameter() = default;

      /// converts a container into an indexed block layout parameter
      /// \tparam List_T container type, must work with range-based for loops, value type must
      /// have a single element (the displacement), convertible to int
      /// \param list container
      template<typename List_T>
      explicit parameter(const List_T &list) {
        for (const auto &i : list)
          add(i);
      }

      /// converts an initializer list into an indexed block layout parameter
      /// \param list initializer list with integers representing displacements
      parameter(std::initializer_list<int> list) {
        for (int i : list)
          add(i);
      }

      /// add an additional block
      /// \param displacement displacement (relative to the beginning of the first block)
      void add(int displacement) {
        displacements.push_back(displacement);
      }

      friend class indexed_block_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        int blocklengths, const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_indexed_block(par.displacements.size(), blocklengths,
                                    par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    indexed_block_layout() : layout<T>(build()) {
    }

    /// constructs indexed layout for data of type T
    /// \param blocklength the length of each block
    /// \param par parameter containing information about the layout
    /// \note displacements are given in multiples of the extent of \c T
    explicit indexed_block_layout(int blocklength, const parameter &par)
        : layout<T>(build(blocklength, par)) {
    }

    /// constructs indexed layout for data with some other layout
    /// \param blocklength the length of each block
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    /// \note displacements are given in multiples of the extent of \c l
    explicit indexed_block_layout(int blocklength, const parameter &par, const layout<T> &l)
        : layout<T>(build(blocklength, par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    indexed_block_layout(const indexed_block_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    indexed_block_layout(indexed_block_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    indexed_block_layout &operator=(const indexed_block_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    indexed_block_layout &operator=(indexed_block_layout &&l) noexcept = default;

    /// exchanges two indexed layouts
    /// \param other the layout to swap with
    void swap(indexed_block_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing data in a sequence of consecutive homogenous blocks of uniform
  /// lengths.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class hindexed_block_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize a heterogeneously indexed
    /// layout.
    class parameter {
      std::vector<MPI_Aint> displacements;

    public:
      /// creates an hindexed block layout parameter for an empty sequence
      parameter() = default;

      /// converts a container into a parameters
      /// \tparam List_T container type, must work with range-based for loops, value type must
      /// have a single element (the displacement), convertible to int
      /// \param list container
      template<typename List_T>
      explicit parameter(const List_T &list) {
        for (const auto &i : list)
          add(i);
      }

      /// converts an initializer list into an hindexed block layout parameter
      /// \param list initializer list with integers representing the displacements
      parameter(std::initializer_list<ssize_t> list) {
        for (const ssize_t &i : list)
          add(i);
      }

      /// add an additional block
      /// \param displacement displacement (relative to the beginning of the first block)
      void add(ssize_t displacement) {
        static_assert(
            std::numeric_limits<MPI_Aint>::min() <= std::numeric_limits<ssize_t>::min() and
                std::numeric_limits<MPI_Aint>::max() >= std::numeric_limits<ssize_t>::max(),
            "MPI implementation with unusual MPI_Aint");
        displacements.push_back(displacement);
      }

      friend class hindexed_block_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        int blocklengths, const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_hindexed_block(par.displacements.size(), blocklengths,
                                     par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    hindexed_block_layout() : layout<T>(build()) {
    }

    /// constructs heterogeneously indexed layout for data of type T
    /// \param blocklength the length of each block
    /// \param par parameter containing information about the layout
    /// \note displacements are given in bytes
    explicit hindexed_block_layout(int blocklength, const parameter &par)
        : layout<T>(build(blocklength, par)) {
    }

    /// constructs heterogeneously indexed layout for data with some other layout
    /// \param blocklength the length of each block
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    /// \note displacements are given bytes
    explicit hindexed_block_layout(int blocklength, const parameter &par, const layout<T> &l)
        : layout<T>(build(blocklength, par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    hindexed_block_layout(const hindexed_block_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    hindexed_block_layout(hindexed_block_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    hindexed_block_layout &operator=(const hindexed_block_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    hindexed_block_layout &operator=(hindexed_block_layout &&l) noexcept = default;

    /// exchanges two indexed layouts
    /// \param other the layout to swap with
    void swap(hindexed_block_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Layout representing data at non-consecutive memory locations, which can be
  /// addressed via an iterator.
  /// \tparam T base element type
  /// \note Iterators that have been used to create objects of this type must not become invalid
  /// during the object's life time.  Iterator must allow read access.
  /// \see inherits all member methods of \c layout
  template<typename T>
  class iterator_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize an iterator layout.
    class parameter {
      std::vector<MPI_Aint> displacements;
      std::vector<int> blocklengths;

      template<typename value_T>
      void add(value_T &base, value_T *&i, MPI_Count extent_) {
        add(reinterpret_cast<char *>(&i) - reinterpret_cast<char *>(&base), extent_);
      }

      template<typename value_T>
      void add(const value_T &base, const value_T &i, MPI_Count extent_) {
        add(reinterpret_cast<const char *>(&i) - reinterpret_cast<const char *>(&base),
            extent_);
      }

      void add(MPI_Aint displacement, MPI_Count extent_) {
        if ((not displacements.empty()) and
            displacements.back() + blocklengths.back() * extent_ == displacement and
            blocklengths.back() < std::numeric_limits<int>::max())
          blocklengths.back()++;
        else {
          displacements.push_back(displacement);
          blocklengths.push_back(1);
        }
      }

    public:
      /// creates parameters for an iterator layout representing an empty sequence
      parameter() = default;

      /// converts an iterator pair into an iterator layout  parameter
      /// \tparam itert_T iterator type
      /// \param first iterator to the first element
      /// \param last iterator pointing after the last element
      template<typename iter_T>
      parameter(iter_T first, iter_T last) {
        MPI_Count lb_, extent_;
        MPI_Type_get_extent_x(detail::datatype_traits<T>::get_datatype(), &lb_, &extent_);
        if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
          throw invalid_datatype_bound();
        for (iter_T i = first; i != last; ++i)
          add(*first, *i, extent_);
      }

      friend class iterator_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
#if defined MPL_DEBUG
      if (par.displacements.size() >
          static_cast<decltype(par.displacements.size())>(std::numeric_limits<int>::max()))
        throw invalid_size();
#endif
      MPI_Type_create_hindexed(par.displacements.size(), par.blocklengths.data(),
                               par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    iterator_layout() : layout<T>(build()) {
    }

    /// constructs iterator layout for data of type T
    /// \tparam iter_T iterator type
    /// \param first iterator to the first element
    /// \param last iterator pointing after the last element
    template<typename iter_T>
    explicit iterator_layout(iter_T first, iter_T last)
        : layout<T>(build(parameter(first, last))) {
    }

    /// constructs iterator layout for data of type T
    /// \param par parameter containing information about the layout
    explicit iterator_layout(const parameter &par) : layout<T>(build(par)) {
    }

    /// constructs iterator layout for data with some other layout
    /// \tparam itert_T iterator type
    /// \param first iterator to the first element
    /// \param last iterator pointing after the last element
    /// \param l the layout of a single element
    template<typename iter_T>
    explicit iterator_layout(iter_T first, iter_T last, const layout<T> &l)
        : layout<T>(build(parameter(first, last), l.type_)) {
    }

    /// constructs iterator layout for data with some other layout
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    explicit iterator_layout(const parameter &par, const layout<T> &l)
        : layout<T>(build(par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    iterator_layout(const iterator_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    iterator_layout(iterator_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    iterator_layout &operator=(const iterator_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    iterator_layout &operator=(iterator_layout &&l) noexcept = default;

    /// exchanges two iterator layouts
    /// \param other the layout to swap with
    void swap(iterator_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Represents order of elements in a multi dimensional array
  /// \see \c subarray_layout
  enum class array_orders {
    /// row-major order, also known as lexographical access order or as C order
    C_order = MPI_ORDER_C,
    /// column-major order, also known as colexographical access order or as Fortran
    /// order
    Fortran_order = MPI_ORDER_FORTRAN
  };

  /// Layout representing data of a subset multi dimensional array.
  /// \tparam T base element type
  /// \see inherits all member methods of \c layout
  template<typename T>
  class subarray_layout : public layout<T> {
    using layout<T>::type_;

  public:
    /// Class representing the parameters to characterize a subarray layout.
    class parameter {
      std::vector<int> sizes, subsizes, starts;
      array_orders order_ = array_orders::C_order;

    public:
      /// creates parameters for a subarray layout representing an empty subarray
      parameter() = default;

      /// creates parameters for a subarray layout
      /// \tparam List_T container type holding three-element values, must support range-based
      /// for loops
      /// \param V container type holding three-element values
      /// \note Each element of parameter V describes the subarray in one dimension.  The first
      /// component of each value gives the total size of the dimension, the second one equals
      /// the size of the subarray, the third value gives the index of the first element of the
      /// subarray.  Array order is C order.
      template<typename List_T>
      explicit parameter(const List_T &V) {
        for (const auto &i : V)
          add(std::get<0>(i), std::get<1>(i), std::get<2>(i));
      }

      /// creates parameters for a subarray layout
      /// \param list list describing the subarray geometry
      /// \note Each element of parameter list describes the subarray in one dimension.  The
      /// first component of each value gives the total size of the dimension, the second one
      /// equals the size of the subarray, the third value gives the index of the first element
      /// of the subarray.  Array order is C order.
      parameter(std::initializer_list<std::array<int, 3>> list) {
        for (const std::array<int, 3> &i : list)
          add(i[0], i[1], i[2]);
      }

      /// add parameters for an additional array dimension
      /// \param size total size of the array along the current dimension
      /// \param subsize size of the subarray
      /// \param start the index of the first element of the subarray
      void add(int size, int subsize, int start) {
        sizes.push_back(size);
        subsizes.push_back(subsize);
        starts.push_back(start);
      }

      /// set the array order
      /// \param new_order the array order
      void order(array_orders new_order) {
        order_ = new_order;
      }

      /// get the array order
      /// \return array order
      [[nodiscard]] array_orders order() const {
        return order_;
      }

      friend class subarray_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, detail::datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(
        const parameter &par,
        MPI_Datatype old_type = detail::datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      int total_size = 1;
      for (std::vector<int>::size_type i = 0; i < par.sizes.size(); ++i)
        total_size *= par.subsizes[i];
      if (total_size > 0)
        MPI_Type_create_subarray(par.sizes.size(), par.sizes.data(), par.subsizes.data(),
                                 par.starts.data(), static_cast<int>(par.order()), old_type,
                                 &new_type);
      else
        new_type = build();
      return new_type;
    }

  public:
    /// constructs a layout with no data
    subarray_layout() : layout<T>(build()) {
    }

    /// constructs subarray layout for data of type T
    /// \param par parameter containing information about the layout
    explicit subarray_layout(const parameter &par) : layout<T>(build(par)) {
    }

    /// constructs subarray layout for data with some other layout
    /// \param par parameter containing information about the layout
    /// \param l the layout of a single element
    explicit subarray_layout(const parameter &par, const layout<T> &l)
        : layout<T>(build(par, l.type_)) {
    }

    /// copy constructor
    /// \param l layout to copy from
    subarray_layout(const subarray_layout &l) : layout<T>(l) {
    }

    /// move constructor
    /// \param l layout to move from
    subarray_layout(subarray_layout &&l) noexcept : layout<T>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    subarray_layout &operator=(const subarray_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    subarray_layout &operator=(subarray_layout &&l) noexcept = default;

    /// exchanges two subarray layouts
    /// \param other the layout to swap with
    void swap(subarray_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;
  };

  //--------------------------------------------------------------------

  /// Helper class for \c heterogeneous_layout,
  /// <tt>make_absolute(T *x, const layout<T> &l)</tt> and
  /// <tt>make_absolute(const T *x, const layout<T> &l)</tt>.
  /// \tparam T data type
  template<typename T>
  class absolute_data;

  /// Helper class for \c heterogeneous_layout,
  /// <tt>make_absolute(T *x, const layout<T> &l)</tt> and
  /// <tt>make_absolute(const T *x, const layout<T> &l)</tt>.
  /// \tparam T data type
  template<typename T>
  class absolute_data<T *> {
    T *address_{nullptr};
    MPI_Datatype datatype_;

    explicit absolute_data(T *address, MPI_Datatype datatype)
        : address_{address}, datatype_{datatype} {
    }

    friend class heterogeneous_layout;

    friend absolute_data make_absolute<>(T *, const layout<T> &);
  };


  /// Helper class for \c heterogeneous_layout,
  /// <tt>make_absolute(T *x, const layout<T> &l)</tt> and
  /// <tt>make_absolute(const T *x, const layout<T> &l)</tt>.
  /// \tparam T data type
  template<typename T>
  class absolute_data<const T *> {
    const T *address_{nullptr};
    MPI_Datatype datatype_;

    explicit absolute_data(const T *address, MPI_Datatype datatype)
        : address_{address}, datatype_{datatype} {
    }

    friend class heterogeneous_layout;

    friend absolute_data make_absolute<>(const T *, const layout<T> &);
  };


  /// Layout representing heterogeneous data at specific memory locations.
  /// \see inherits all member methods of the class layout
  /// \note \c absolute must be used as send/receive buffer argument when sending or receiving
  /// data of heterogeneous layout.
  class heterogeneous_layout : public layout<void> {
    using layout<void>::type_;

  public:
    /// Class representing the parameters to characterize a heterogeneous layout.
    class parameter {
      std::vector<int> block_lengths_;
      std::vector<MPI_Aint> displacements_;
      std::vector<MPI_Datatype> types_;

      void add() const {
      }

    public:
      /// creates parameters for a heterogeneous layout representing an empty data set
      parameter() = default;

      /// creates parameters for a heterogeneous layout
      /// \tparam Ts types of the heterogeneous data
      /// \param xs heterogeneous data elements (parameter pack)
      template<typename... Ts>
      explicit parameter(const Ts &...xs) {
        add(xs...);
      }

      /// adds one or more values to a heterogeneous layout
      /// \tparam T type of the first heterogeneous data element
      /// \param x the first data element
      /// \tparam Ts types of further heterogeneous data elements
      /// \param xs further data elements (parameter pack)
      template<typename T, typename... Ts>
      void add(const T &x, const Ts &...xs) {
        block_lengths_.push_back(1);
        displacements_.push_back(reinterpret_cast<MPI_Aint>(&x));
        types_.push_back(detail::datatype_traits<T>::get_datatype());
        add(xs...);
      }

      /// adds one or more values to a heterogeneous layout
      /// \tparam T type of the first heterogeneous data element
      /// \param x the first data element given by its absolute memory address, return value of
      /// \c make_absolute
      /// \tparam Ts types of further heterogeneous data elements
      /// \param xs further data elements (parameter pack)
      template<typename T, typename... Ts>
      void add(const absolute_data<T *> &x, const Ts &...xs) {
        block_lengths_.push_back(1);
        displacements_.push_back(reinterpret_cast<MPI_Aint>(x.address_));
        types_.push_back(x.datatype_);
        add(xs...);
      }

      friend class heterogeneous_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type{};
      MPI_Type_contiguous(0, detail::datatype_traits<char>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par) {
      MPI_Datatype new_type{};
      MPI_Type_create_struct(static_cast<int>(par.block_lengths_.size()),
                             par.block_lengths_.data(), par.displacements_.data(),
                             par.types_.data(), &new_type);
      return new_type;
    }

  public:
    /// constructs a layout with no data
    heterogeneous_layout() : layout<void>(build()) {
    }

    /// constructs heterogeneous layout
    /// \param par parameter containing information about the layout
    explicit heterogeneous_layout(const parameter &par) : layout<void>(build(par)) {
    }

    /// constructs heterogeneous layout
    /// \tparam T type of the first heterogeneous data element
    /// \param x the first data element
    /// \tparam Ts types of further heterogeneous data elements
    /// \param xs further data elements (parameter pack)
    template<typename T, typename... Ts>
    explicit heterogeneous_layout(const T &x, const Ts &...xs)
        : layout<void>(build(parameter(x, xs...))) {
    }

    /// copy constructor
    /// \param l layout to copy from
    heterogeneous_layout(const heterogeneous_layout &l) = default;

    /// move constructor
    /// \param l layout to move from
    heterogeneous_layout(heterogeneous_layout &&l) noexcept : layout<void>(std::move(l)) {
    }

    /// copy assignment operator
    /// \param l layout to copy from
    /// \return reference to this object
    heterogeneous_layout &operator=(const heterogeneous_layout &l) = default;

    /// move assignment operator
    /// \param l layout to move from
    /// \return reference to this object
    heterogeneous_layout &operator=(heterogeneous_layout &&l) noexcept = default;

    /// exchanges two heterogeneous layouts
    /// \param other the layout to swap with
    void swap(heterogeneous_layout &other) noexcept {
      std::swap(type_, other.type_);
    }

    using layout<void>::byte_extent;
    using layout<void>::byte_lower_bound;
    using layout<void>::byte_upper_bound;
    using layout<void>::byte_resize;
    using layout<void>::extent;
    using layout<void>::lower_bound;
    using layout<void>::upper_bound;
    using layout<void>::resize;
  };


  /// Helper function for the class heterogeneous_layout.
  /// \tparam T data type
  /// \param x address of data
  /// \param l layout of data at address x
  /// \see <tt>make_absolute(T *x, const layout<T> &l)</tt> function and
  /// \c heterogeneous_layout class
  template<typename T>
  absolute_data<T *> make_absolute(T *x, const layout<T> &l) {
    return absolute_data<T *>{x, l.type_};
  }

  /// Helper function for the class heterogeneous_layout.
  /// \tparam T data type
  /// \param x address of data
  /// \param l layout of data at address x
  /// \see <tt>make_absolute(const T *x, const layout<T> &l)</tt> function and
  /// \c heterogeneous_layout class
  template<typename T>
  absolute_data<const T *> make_absolute(const T *x, const layout<T> &l) {
    return absolute_data<const T *>{x, l.type_};
  }

  //--------------------------------------------------------------------

  namespace detail {

    template<typename T>
    struct datatype_traits<layout<T>> {
      static MPI_Datatype get_datatype(const layout<T> &l) {
        return l.type_;
      }
    };

  }  // namespace detail

  //--------------------------------------------------------------------

  /// container for storing layouts
  /// \tparam T base element type of the layouts
  template<typename T>
  class layouts : private std::vector<layout<T>> {
    using base = std::vector<layout<T>>;

  public:
    /// type for index access
    using typename base::size_type;

    /// constructs a layout container with no data
    layouts() : base() {
    }

    /// constructs a layout container of empty layouts
    /// \param n number of initial layouts in container
    explicit layouts(size_type n) : base(n, empty_layout<T>()) {
    }

    /// constructs a layout container
    /// \param n number of initial layouts in container
    /// \param l layout used to initialize the layout container
    explicit layouts(size_type n, const layout<T> &l) : base(n, l) {
    }

    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    friend class impl::base_communicator;
    friend class impl::topology_communicator;
    friend class communicator;

  private:
    const layout<T> *operator()() const {
      return base::data();
    }
  };

  //--------------------------------------------------------------------

  /// container for storing contiguous layouts
  /// \tparam T base element type of the contiguous layouts
  template<typename T>
  class contiguous_layouts : private std::vector<contiguous_layout<T>> {
    using base = std::vector<contiguous_layout<T>>;
    mutable std::vector<int> s;

  public:
    /// type for index access
    using typename base::size_type;

    /// constructs a layout container with no data
    contiguous_layouts() : base() {
    }

    /// constructs a layout container of empty layouts
    /// \param n number of initial layouts in container
    explicit contiguous_layouts(size_type n) : base(n, contiguous_layout<T>()), s() {
    }

    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    friend class impl::base_communicator;
    friend class impl::topology_communicator;

  private:
    const contiguous_layout<T> *operator()() const {
      return base::data();
    }

    const int *sizes() const {
      s.clear();
      s.reserve(size());
      std::transform(begin(), end(), std::back_inserter(s),
                     [](const contiguous_layout<T> &l) { return l.size(); });
      return s.data();
    }
  };

}  // namespace mpl

#endif
