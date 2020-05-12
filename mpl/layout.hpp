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
  std::pair<T *, MPI_Datatype> make_absolute(T *, const layout<T> &);

  template<typename T>
  std::pair<const T *, MPI_Datatype> make_absolute(const T *, const layout<T> &);

  template<typename T>
  class contiguous_layouts;

  //--------------------------------------------------------------------

  template<typename T>
  class layout {
  private:
    MPI_Datatype type;

  protected:
    explicit layout(MPI_Datatype new_type) : type(new_type) {
      if (type != MPI_DATATYPE_NULL)
        MPI_Type_commit(&type);
    }

  public:
    layout() : type(MPI_DATATYPE_NULL) {}

    layout(const layout &l) {
      if (l.type != MPI_DATATYPE_NULL)
        MPI_Type_dup(l.type, &type);
      else
        type = MPI_DATATYPE_NULL;
    }

    layout(layout &&l) noexcept {
      type = l.type;
      l.type = MPI_DATATYPE_NULL;
    }

    layout &operator=(const layout &l) {
      if (this != &l) {
        if (type != MPI_DATATYPE_NULL)
          MPI_Type_free(&type);
        if (l.type != MPI_DATATYPE_NULL)
          MPI_Type_dup(l.type, &type);
        else
          type = MPI_DATATYPE_NULL;
      }
      return *this;
    }

    layout &operator=(layout &&l) noexcept {
      if (this != &l) {
        if (type != MPI_DATATYPE_NULL)
          MPI_Type_free(&type);
        type = l.type;
        l.type = MPI_DATATYPE_NULL;
      }
      return *this;
    }

    std::ptrdiff_t byte_extent() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_;
    }

    std::ptrdiff_t byte_lower_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return lb_;
    }

    std::ptrdiff_t byte_upper_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_ - lb_;
    }

    std::ptrdiff_t extent() const {
      std::ptrdiff_t res = byte_extent();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    std::ptrdiff_t lower_bound() const {
      std::ptrdiff_t res = byte_lower_bound();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    std::ptrdiff_t upper_bound() const {
      std::ptrdiff_t res = byte_upper_bound();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    std::ptrdiff_t true_byte_extent() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_;
    }

    std::ptrdiff_t true_byte_lower_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return lb_;
    }

    std::ptrdiff_t true_byte_upper_bound() const {
      MPI_Count lb_, extent_;
      MPI_Type_get_true_extent_x(type, &lb_, &extent_);
      if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
        throw invalid_datatype_bound();
      return extent_ - lb_;
    }

    std::ptrdiff_t true_extent() const {
      std::ptrdiff_t res = true_byte_extent();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    std::ptrdiff_t true_lower_bound() const {
      std::ptrdiff_t res = true_byte_lower_bound();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    std::ptrdiff_t true_upper_bound() const {
      std::ptrdiff_t res = true_byte_upper_bound();
      if (res / sizeof(T) * sizeof(T) != res)
        throw invalid_datatype_bound();
      return res / sizeof(T);
    }

    void byte_resize(std::ptrdiff_t lb, std::ptrdiff_t extent) {
      if (type != MPI_DATATYPE_NULL) {
        MPI_Datatype newtype;
        MPI_Type_create_resized(type, lb, extent, &newtype);
        MPI_Type_commit(&newtype);
        MPI_Type_free(&type);
        type = newtype;
      }
    }

    void resize(std::ptrdiff_t lb, std::ptrdiff_t extent) {
      if (type != MPI_DATATYPE_NULL) {
        MPI_Datatype newtype;
        MPI_Type_create_resized(type, lb * sizeof(T), extent * sizeof(T), &newtype);
        MPI_Type_commit(&newtype);
        MPI_Type_free(&type);
        type = newtype;
      }
    }

    void swap(layout &l) { std::swap(type, l.type); }

    ~layout() {
      if (type != MPI_DATATYPE_NULL)
        MPI_Type_free(&type);
    }

    friend class datatype_traits<layout<T>>;

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

    friend std::pair<T *, MPI_Datatype> make_absolute<>(T *, const layout<T> &);

    friend std::pair<const T *, MPI_Datatype> make_absolute<>(const T *, const layout<T> &);
  };

  //--------------------------------------------------------------------

  template<typename T>
  class null_layout : public layout<T> {
    using layout<T>::type;

  public:
    null_layout() : layout<T>(MPI_DATATYPE_NULL) {}

    void swap(null_layout<T> &other) {}

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

  template<typename T>
  class empty_layout : public layout<T> {
    using layout<T>::type;

    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

  public:
    empty_layout() : layout<T>(build()) {}

    empty_layout(const empty_layout &l) : layout<T>(l) {}

    empty_layout(empty_layout &&l) noexcept : layout<T>(std::move(l)) {}

    empty_layout<T> &operator=(const empty_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    empty_layout<T> &operator=(empty_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(empty_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class contiguous_layout : public layout<T> {
    using layout<T>::type;

    static MPI_Datatype build(size_t count,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      if (count <= std::numeric_limits<int>::max()) {
        MPI_Type_contiguous(static_cast<int>(count), old_type, &new_type);
      } else {
        const size_t modulus = std::numeric_limits<int>::max();
        const size_t count1 = count / modulus;
        const size_t count0 = count - count1 * modulus;
        MPI_Count lb, extent;
        MPI_Type_get_extent_x(old_type, &lb, &extent);
        MPI_Datatype type_modulus;
        MPI_Type_contiguous(static_cast<int>(modulus), old_type, &type_modulus);
        std::vector<int> block_lengths{static_cast<int>(count0), static_cast<int>(count1)};
#if defined MPL_DEBUG
        if (count0 * extent > std::numeric_limits<MPI_Aint>::max())
          throw invalid_size();
#endif
        std::vector<MPI_Aint> displacements{0, static_cast<MPI_Aint>(count0 * extent)};
        std::vector<MPI_Datatype> types{old_type, type_modulus};
        MPI_Type_create_struct(2, block_lengths.data(), displacements.data(), types.data(),
                               &new_type);
      }
      return new_type;
    }

    size_t count;

    size_t size() const { return count; }

  public:
    explicit contiguous_layout(size_t c = 0) : layout<T>(build(c)), count(c) {}

    explicit contiguous_layout(size_t c, const contiguous_layout<T> &other)
        : layout<T>(build(c, other.type)), count(other.c * c) {}

    contiguous_layout(const contiguous_layout<T> &l) : layout<T>(l), count(l.count) {}

    contiguous_layout(contiguous_layout &&l) noexcept
        : layout<T>(std::move(l)), count(l.count) {
      l.count = 0;
    }

    contiguous_layout<T> &operator=(const contiguous_layout<T> &l) {
      layout<T>::operator=(l);
      count = l.count;
      return *this;
    }

    contiguous_layout<T> &operator=(contiguous_layout<T> &&l) noexcept {
      if (this != &l) {
        layout<T>::operator=(std::move(l));
        count = l.count;
        l.count = 0;
      }
      return *this;
    }

    void swap(contiguous_layout<T> &other) {
      std::swap(type, other.type);
      std::swap(count, other.count);
    }

    using layout<T>::byte_extent;
    using layout<T>::byte_lower_bound;
    using layout<T>::byte_upper_bound;
    using layout<T>::byte_resize;
    using layout<T>::extent;
    using layout<T>::lower_bound;
    using layout<T>::upper_bound;
    using layout<T>::resize;

    friend class communicator;
    friend class contiguous_layouts<T>;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class vector_layout : public layout<T> {
    using layout<T>::type;

    static MPI_Datatype build(size_t count,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      if (count <= std::numeric_limits<int>::max()) {
        MPI_Type_contiguous(static_cast<int>(count), old_type, &new_type);
      } else {
        const size_t modulus = std::numeric_limits<int>::max();
        const size_t count1 = count / modulus;
        const size_t count0 = count - count1 * modulus;
        MPI_Count lb, extent;
        MPI_Type_get_extent_x(old_type, &lb, &extent);
        MPI_Datatype type_modulus;
        MPI_Type_contiguous(static_cast<int>(modulus), old_type, &type_modulus);
        std::vector<int> block_lengths{static_cast<int>(count0), static_cast<int>(count1)};
#if defined MPL_DEBUG
        if (count0 * extent > std::numeric_limits<MPI_Aint>::max())
          throw invalid_size();
#endif
        std::vector<MPI_Aint> displacements{0, static_cast<MPI_Aint>(count0 * extent)};
        std::vector<MPI_Datatype> types{old_type, type_modulus};
        MPI_Type_create_struct(2, block_lengths.data(), displacements.data(), types.data(),
                               &new_type);
      }
      return new_type;
    }

  public:
    explicit vector_layout(size_t c = 0) : layout<T>(build(c)) {}

    explicit vector_layout(size_t c, const layout<T> &other)
        : layout<T>(build(c, other.type)) {}

    explicit vector_layout(size_t c, const vector_layout<T> &other)
        : layout<T>(build(c, other.type)) {}

    vector_layout(const vector_layout<T> &l) : layout<T>(l) {}

    vector_layout(vector_layout &&l) noexcept : layout<T>(std::move(l)) {}

    vector_layout<T> &operator=(const vector_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    vector_layout<T> &operator=(vector_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(vector_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class strided_vector_layout : public layout<T> {
    using layout<T>::type;

    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(int count, int blocklength, int stride,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, stride, old_type, &new_type);
      return new_type;
    }

  public:
    strided_vector_layout() : layout<T>(build()) {}

    explicit strided_vector_layout(int count, int blocklength, int stride)
        : layout<T>(build(count, blocklength, stride)) {}

    explicit strided_vector_layout(int count, int blocklength, int stride,
                                   const layout<T> &other)
        : layout<T>(build(count, blocklength, stride, other.type)) {}

    strided_vector_layout(const strided_vector_layout<T> &l) : layout<T>(l) {}

    strided_vector_layout(strided_vector_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    strided_vector_layout<T> &operator=(const strided_vector_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    strided_vector_layout<T> &operator=(strided_vector_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(strided_vector_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class indexed_layout : public layout<T> {
    using layout<T>::type;

  public:
    class parameter {
      std::vector<int> blocklengths, displacements;

    public:
      parameter() = default;

      template<typename List_T>
      parameter(const List_T &V) {
        for (const auto &i : V)
          add(i[0], i[1]);
      }

      parameter(std::initializer_list<std::array<int, 2>> list) {
        for (const std::array<int, 2> &i : list)
          add(i[0], i[1]);
      }

      void add(int blocklength, int displacement) {
        blocklengths.push_back(blocklength);
        displacements.push_back(displacement);
      }

      friend class indexed_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_indexed(par.displacements.size(), par.blocklengths.data(),
                       par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    indexed_layout() : layout<T>(build()) {}

    explicit indexed_layout(const parameter &par) : layout<T>(build(par)) {}

    explicit indexed_layout(const parameter &par, const layout<T> &other)
        : layout<T>(build(par, other.type)) {}

    indexed_layout(const indexed_layout<T> &l) : layout<T>(l) {}

    indexed_layout(indexed_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    indexed_layout<T> &operator=(const indexed_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    indexed_layout<T> &operator=(indexed_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(indexed_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class hindexed_layout : public layout<T> {
    using layout<T>::type;

  public:
    class parameter {
      std::vector<int> blocklengths;
      std::vector<MPI_Aint> displacements;

    public:
      parameter() = default;

      template<typename List_T>
      parameter(const List_T &V) {
        for (const auto &i : V)
          add(i.first, i.second);
      }

      parameter(std::initializer_list<std::pair<int, MPI_Aint>> list) {
        for (const std::pair<int, MPI_Aint> &i : list)
          add(i.first, i.second);
      }

      void add(int blocklength, MPI_Aint displacement) {
        blocklengths.push_back(blocklength);
        displacements.push_back(displacement);
      }

      friend class hindexed_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_hindexed(par.displacements.size(), par.blocklengths.data(),
                               par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    hindexed_layout() : layout<T>(build()) {}

    explicit hindexed_layout(const parameter &par) : layout<T>(build(par)) {}

    explicit hindexed_layout(const parameter &par, const layout<T> &other)
        : layout<T>(build(par, other.type)) {}

    hindexed_layout(const hindexed_layout<T> &l) : layout<T>(l) {}

    hindexed_layout(hindexed_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    hindexed_layout<T> &operator=(const hindexed_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    hindexed_layout<T> &operator=(hindexed_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(hindexed_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class indexed_block_layout : public layout<T> {
    using layout<T>::type;

  public:
    class parameter {
      std::vector<int> displacements;

    public:
      parameter() = default;

      template<typename List_T>
      parameter(const List_T &V) {
        for (const auto &i : V)
          add(i);
      }

      parameter(std::initializer_list<int> list) {
        for (int i : list)
          add(i);
      }

      void add(int displacement) { displacements.push_back(displacement); }

      friend class indexed_block_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(int blocklengths, const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_indexed_block(par.displacements.size(), blocklengths,
                                    par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    indexed_block_layout() : layout<T>(build()) {}

    explicit indexed_block_layout(int blocklengths, const parameter &par)
        : layout<T>(build(blocklengths, par)) {}

    explicit indexed_block_layout(int blocklengths, const parameter &par,
                                  const layout<T> &other)
        : layout<T>(build(blocklengths, par, other.type)) {}

    indexed_block_layout(const indexed_block_layout<T> &l) : layout<T>(l) {}

    indexed_block_layout(indexed_block_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    indexed_block_layout<T> &operator=(const indexed_block_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    indexed_block_layout<T> &operator=(indexed_block_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(indexed_block_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class hindexed_block_layout : public layout<T> {
    using layout<T>::type;

  public:
    class parameter {
      std::vector<MPI_Aint> displacements;

    public:
      parameter() = default;

      template<typename List_T>
      parameter(const List_T &V) {
        for (const auto &i : V)
          add(i);
      }

      parameter(std::initializer_list<MPI_Aint> list) {
        for (int i : list)
          add(i);
      }

      void add(MPI_Aint displacement) { displacements.push_back(displacement); }

      friend class hindexed_block_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(int blocklengths, const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_hindexed_block(par.displacements.size(), blocklengths,
                                     par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    hindexed_block_layout() : layout<T>(build()) {}

    explicit hindexed_block_layout(int blocklengths, const parameter &par)
        : layout<T>(build(blocklengths, par)) {}

    explicit hindexed_block_layout(int blocklengths, const parameter &par,
                                   const layout<T> &other)
        : layout<T>(build(blocklengths, par, other.type)) {}

    hindexed_block_layout(const hindexed_block_layout<T> &l) : layout<T>(l) {}

    hindexed_block_layout(hindexed_block_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    hindexed_block_layout<T> &operator=(const hindexed_block_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    hindexed_block_layout<T> &operator=(hindexed_block_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(hindexed_block_layout<T> &other) { std::swap(type, other.type); }

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

  template<typename T>
  class iterator_layout : public layout<T> {
    using layout<T>::type;

  public:
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
      parameter() = default;

      template<typename iter_T>
      parameter(iter_T first, iter_T end) {
        MPI_Count lb_, extent_;
        MPI_Type_get_extent_x(datatype_traits<T>::get_datatype(), &lb_, &extent_);
        if (lb_ == MPI_UNDEFINED or extent_ == MPI_UNDEFINED)
          throw invalid_datatype_bound();
        for (iter_T i = first; i != end; ++i)
          add(*first, *i, extent_);
      }

      friend class iterator_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
#if defined MPL_DEBUG
      if (par.displacements.size() > std::numeric_limits<int>::max())
        throw invalid_size();
#endif
      MPI_Type_create_hindexed(par.displacements.size(), par.blocklengths.data(),
                               par.displacements.data(), old_type, &new_type);
      return new_type;
    }

  public:
    iterator_layout() : layout<T>(build()) {}

    template<typename iter_T>
    explicit iterator_layout(iter_T first, iter_T end)
        : layout<T>(build(parameter(first, end))) {}

    explicit iterator_layout(const parameter &par) : layout<T>(build(par)) {}

    template<typename iter_T>
    explicit iterator_layout(iter_T first, iter_T end, const layout<T> &other)
        : layout<T>(build(parameter(first, end), other.type)) {}

    explicit iterator_layout(const parameter &par, const layout<T> &other)
        : layout<T>(build(par, other.type)) {}

    iterator_layout(const iterator_layout<T> &l) : layout<T>(l) {}

    iterator_layout(iterator_layout<T> &&l) noexcept : layout<T>(std::move(l)) {}

    iterator_layout<T> &operator=(const iterator_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    iterator_layout<T> &operator=(iterator_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(iterator_layout<T> &other) { std::swap(type, other.type); }

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

  enum class array_orders { C_order = MPI_ORDER_C, Fortran_order = MPI_ORDER_FORTRAN };

  template<typename T>
  class subarray_layout : public layout<T> {
    using layout<T>::type;

  public:
    class parameter {
      std::vector<int> sizes, subsizes, starts;
      array_orders order_ = array_orders::C_order;

    public:
      parameter() = default;

      template<typename List_T>
      parameter(const List_T &V) {
        for (const auto &i : V)
          add(i[0], i[1], i[2]);
      }

      parameter(std::initializer_list<std::array<int, 3>> list) {
        for (const std::array<int, 3> &i : list)
          add(i[0], i[1], i[2]);
      }

      void add(int size, int subsize, int start) {
        sizes.push_back(size);
        subsizes.push_back(subsize);
        starts.push_back(start);
      }

      void order(array_orders new_order) { order_ = new_order; }

      array_orders order() const { return order_; }

      friend class subarray_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par,
                              MPI_Datatype old_type = datatype_traits<T>::get_datatype()) {
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
    subarray_layout() : layout<T>(build()) {}

    explicit subarray_layout(const parameter &par) : layout<T>(build(par)) {}

    explicit subarray_layout(const parameter &par, const layout<T> &other)
        : layout<T>(build(par, other.type)) {}

    subarray_layout(const subarray_layout<T> &l) : layout<T>(l) {}

    subarray_layout(subarray_layout<T> &&l) : layout<T>(std::move(l)) {}

    subarray_layout<T> &operator=(const subarray_layout<T> &l) {
      layout<T>::operator=(l);
      return *this;
    }

    subarray_layout<T> &operator=(subarray_layout<T> &&l) noexcept {
      layout<T>::operator=(std::move(l));
      return *this;
    }

    void swap(subarray_layout<T> &other) { std::swap(type, other.type); }

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

  class heterogeneous_layout : public layout<void> {
    using layout<void>::type;

  public:
    class parameter {
      std::vector<int> blocklengths;
      std::vector<MPI_Aint> displacements;
      std::vector<MPI_Datatype> types;

      void add() const {}

    public:
      parameter() = default;

      template<typename... Ts>
      parameter(const Ts &... xs) {
        add(xs...);
      }

      template<typename T, typename... Ts>
      void add(const T &x, const Ts &... xs) {
        blocklengths.push_back(1);
        displacements.push_back(reinterpret_cast<MPI_Aint>(&x));
        types.push_back(datatype_traits<T>::get_datatype());
        add(xs...);
      }

      template<typename T, typename... Ts>
      void add(const std::pair<T *, MPI_Datatype> &x, const Ts &... xs) {
        blocklengths.push_back(1);
        displacements.push_back(reinterpret_cast<MPI_Aint>(x.first));
        types.push_back(x.second);
        add(xs...);
      }

      friend class heterogeneous_layout;
    };

  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<char>::get_datatype(), &new_type);
      return new_type;
    }

    static MPI_Datatype build(const parameter &par) {
      MPI_Datatype new_type;
      MPI_Type_create_struct(static_cast<int>(par.blocklengths.size()), par.blocklengths.data(),
                             par.displacements.data(), par.types.data(), &new_type);
      return new_type;
    }

  public:
    heterogeneous_layout() : layout<void>(build()) {}

    explicit heterogeneous_layout(const parameter &par) : layout<void>(build(par)) {}

    template<typename T, typename... Ts>
    explicit heterogeneous_layout(const T &x, const Ts &... xs)
        : layout<void>(build(parameter(x, xs...))) {}

    heterogeneous_layout(const heterogeneous_layout &l) : layout<void>(l) {}

    heterogeneous_layout(heterogeneous_layout &&l) : layout<void>(std::move(l)) {}

    heterogeneous_layout &operator=(const heterogeneous_layout &l) {
      layout<void>::operator=(l);
      return *this;
    }

    heterogeneous_layout &operator=(heterogeneous_layout &&l) noexcept {
      layout<void>::operator=(std::move(l));
      return *this;
    }

    void swap(heterogeneous_layout &other) { std::swap(type, other.type); }

    using layout<void>::byte_extent;
    using layout<void>::byte_lower_bound;
    using layout<void>::byte_upper_bound;
    using layout<void>::byte_resize;
    using layout<void>::extent;
    using layout<void>::lower_bound;
    using layout<void>::upper_bound;
    using layout<void>::resize;
  };


  template<typename T>
  inline std::pair<T *, MPI_Datatype> make_absolute(T *x, const layout<T> &l) {
    return std::make_pair(x, l.type);
  }

  template<typename T>
  inline std::pair<const T *, MPI_Datatype> make_absolute(const T *x, const layout<T> &l) {
    return std::make_pair(x, l.type);
  }

  //--------------------------------------------------------------------

  template<typename T>
  struct datatype_traits<layout<T>> {
    static MPI_Datatype get_datatype(const layout<T> &l) { return l.type; }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class layouts : private std::vector<layout<T>> {
    using base = std::vector<layout<T>>;

  public:
    using typename base::size_type;

    explicit layouts(size_type n = 0) : base(n, empty_layout<T>()) {}

    explicit layouts(size_type n, const layout<T> &l) : base(n, l) {}

    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    const layout<T> *operator()() const { return base::data(); }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class contiguous_layouts : private std::vector<contiguous_layout<T>> {
    using base = std::vector<contiguous_layout<T>>;
    mutable std::vector<int> s;

  public:
    using typename base::size_type;

    explicit contiguous_layouts(size_type n = 0) : base(n, contiguous_layout<T>()), s() {}

    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;

    const contiguous_layout<T> *operator()() const { return base::data(); }

    const int *sizes() const {
      s.clear();
      s.reserve(size());
      for (const auto &i : *this)
        s.push_back(i.size());
      return s.data();
    }
  };

}  // namespace mpl

#endif
