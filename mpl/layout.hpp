#if !(defined MPL_LAYOUT_HPP)

#define MPL_LAYOUT_HPP

#include <mpi.h>
#include <iterator>
#include <initializer_list>
#include <type_traits>

namespace mpl {
  
  template<typename T>
  class layout {
  protected:
    MPI_Datatype type;
    explicit layout(MPI_Datatype new_type) : type(new_type) {
    }
  public:
    layout() : type(MPI_DATATYPE_NULL) {
    }
    layout(const layout &l) {
      if (l.type!=MPI_DATATYPE_NULL)
	MPI_Type_dup(l.type, &type);
      else 
	type=MPI_DATATYPE_NULL;
    }
    layout & operator=(const layout &l) {
      if (this!=&l) {
	if (type!=MPI_DATATYPE_NULL)
	  MPI_Type_free(&type);
	if (l.type!=MPI_DATATYPE_NULL)
	  MPI_Type_dup(l.type, &type);
	else
	  type=MPI_DATATYPE_NULL;
      }
      return *this;
    }
    ~layout() {
      if (type!=MPI_DATATYPE_NULL)
	MPI_Type_free(&type);
    }
    friend struct datatype_traits<layout<T> >;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class null_layout : public layout<T> {
    using layout<T>::type;
  public:
    null_layout() : layout<T>::layout(MPI_DATATYPE_NULL) {
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class empty_layout : public layout<T> {
    using layout<T>::type;
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    int count;
  public:
    explicit empty_layout() : 
      layout<T>::layout(build()), count() {
      MPI_Type_commit(&type);
    }
    empty_layout(const empty_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    empty_layout & operator=(const empty_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    int size() const {
      return count;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class contiguous_layout : public layout<T> {
    using layout<T>::type;
    static MPI_Datatype build(int count) {
      MPI_Datatype new_type;
      MPI_Type_contiguous(count, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    int count;
  public:
    explicit contiguous_layout(int c=0) : 
      layout<T>::layout(build(c)), count(c) {
      MPI_Type_commit(&type);
    }
    contiguous_layout(const contiguous_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    contiguous_layout & operator=(const contiguous_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    int size() const {
      return count;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class vector_layout : public layout<T> {
    using layout<T>::type;
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    static MPI_Datatype build(int count, int blocklength, int stride) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, stride, datatype_traits<T>::get_datatype(),
		      &new_type);
      return new_type;
    }
  public:
    vector_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    explicit vector_layout(int count, int blocklength, int stride) : 
      layout<T>::layout(build(count, blocklength, stride)) {
      MPI_Type_commit(&type);
    }
    vector_layout(const vector_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    vector_layout & operator=(const vector_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class indexed_block_layout : public layout<T> {
    using layout<T>::type;
    template<typename I>
    static MPI_Datatype build(int blocklength, I i1, I i2) {
      static_assert(std::is_same<int, typename std::iterator_traits<I>::value_type>::value, "iterator value_type musst be int");
      detail::flat_memory_in<int, I> displacements(i1, i2);
      MPI_Datatype new_type;
      MPI_Type_create_indexed_block(displacements.size(), blocklength, 
				    displacements.data(), 
				    datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }
  public:
    indexed_block_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    template<typename I>
    explicit indexed_block_layout(int blocklength, I i1, I i2) : 
      layout<T>::layout(build(blocklength, i1, i2)) {
      MPI_Type_commit(&type);
    }
    template<typename I>
    explicit indexed_block_layout(int blocklength, std::initializer_list<I> i) : 
      layout<T>::layout(build(blocklength, i.begin(), i.end())) {
      MPI_Type_commit(&type);
    }
    indexed_block_layout(const indexed_block_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    indexed_block_layout & operator=(const indexed_block_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class indexed_layout : public layout<T> {
    using layout<T>::type;
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    template<typename I, typename J>
    static MPI_Datatype build(I i1, I i2, J j1, J j2) {
      static_assert(std::is_same<int, typename std::iterator_traits<I>::value_type>::value, "iterator value_type musst be int");
      static_assert(std::is_same<int, typename std::iterator_traits<J>::value_type>::value, "iterator value_type musst be int");
      detail::flat_memory_in<int, I> blocklengths(i1, i2);
      detail::flat_memory_in<int, J> displacements(j1, j2);
      // TODO: implement error handling for blocklengths.size()!=displacements.size()
      MPI_Datatype new_type;
      MPI_Type_indexed(displacements.size(), blocklengths.data(), displacements.data(), 
		       datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }
  public:
    indexed_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    template<typename I, typename J>
    explicit indexed_layout(I i1, I i2, J j1, J j2) : 
      layout<T>::layout(build(i1, i2, j1, j2)) {
      MPI_Type_commit(&type);
    }
    template<typename I, typename J>
    explicit indexed_layout(I i1, I i2, std::initializer_list<J> j) : 
      layout<T>::layout(build(i1, i2, j.begin(), j.end())) {
      MPI_Type_commit(&type);
    }
    template<typename I, typename J>
    explicit indexed_layout(std::initializer_list<I> i, J j1, J j2) : 
      layout<T>::layout(build(i.begin(), i.end(), j1, j2)) {
      MPI_Type_commit(&type);
    }
    template<typename I, typename J>
    explicit indexed_layout(std::initializer_list<I> i, std::initializer_list<J> j) : 
      layout<T>::layout(build(i.begin(), i.end(), j.begin(), j.end())) {
      MPI_Type_commit(&type);
    }
    indexed_layout(const indexed_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    indexed_layout & operator=(const indexed_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
  };

  //--------------------------------------------------------------------

  template<typename T, int n>
  class subarray_layout : public layout<T> {
  public:
    typedef enum { C_order=MPI_ORDER_C, Fortran_order=MPI_ORDER_FORTRAN } order_type;
  private:
    using layout<T>::type;
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    static MPI_Datatype build(const vector<int, n> &sizes,
			      const vector<int, n> &subsizes,
			      const vector<int, n> &starts, 
			      order_type order) {
      MPI_Datatype new_type;
      int total_size=1;
      for (int i=0; i<n; ++i)
	total_size*=subsizes[i];
      if (total_size>0)
	MPI_Type_create_subarray(n, &sizes[0], &subsizes[0], &starts[0],
				 order,
				 datatype_traits<T>::get_datatype(), &new_type);
      else 
	new_type=build();
      return new_type;
    }
  public:
    subarray_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    subarray_layout(const vector<int, n> &sizes,
		    const vector<int, n> &subsizes,
		    const vector<int, n> &starts, 
		    order_type order=C_order) : 
      layout<T>::layout(build(sizes, subsizes, starts, order)) {
      MPI_Type_commit(&type);
    }		   
    subarray_layout(const subarray_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    subarray_layout & operator=(const subarray_layout &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  struct datatype_traits<layout<T> > {
    static MPI_Datatype get_datatype(const layout<T> &l) {
      return l.type;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class layouts : private std::vector<layout<T> > {
    typedef std::vector<layout<T> > base;
  public:
    typedef typename base::size_type size_type;
    explicit layouts(size_type n=0) : base(n, null_layout<T>()) {
    }
    using base::operator[];
    using base::size;
    using base::push_back;
    const layout<T> * operator()() const {
      return base::data();
    }
  };

}

#endif
