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
      MPI_Type_commit(&type);
    }
    ~layout() {
      MPI_Type_free(&type);
    }
    layout & operator=(const layout &)=delete;
    friend struct datatype_traits<layout<T> >;
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
    explicit contiguous_layout(int c) : 
      layout<T>::layout(build(c)), count(c) {
    }
    int size() const {
      return count;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class vector_layout : public layout<T> {
    using layout<T>::type;
    static MPI_Datatype build(int count, int blocklength, int stride) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, stride, datatype_traits<T>::get_datatype(),
		      &new_type);
      return new_type;
    }
  public:
    vector_layout(int count, int blocklength, int stride) : 
      layout<T>::layout(build(count, blocklength, stride)) {
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
    template<typename I>
    indexed_block_layout(int blocklength, I i1, I i2) : 
      layout<T>::layout(build(blocklength, i1, i2)) {
    }
    template<typename I>
    indexed_block_layout(int blocklength, std::initializer_list<I> i) : 
      layout<T>::layout(build(blocklength, i.begin(), i.end())) {
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class indexed_layout : public layout<T> {
    using layout<T>::type;
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
    template<typename I, typename J>
    indexed_layout(I i1, I i2, J j1, J j2) : 
      layout<T>::layout(build(i1, i2, j1, j2)) {
    }
    template<typename I, typename J>
    indexed_layout(I i1, I i2, std::initializer_list<J> j) : 
      layout<T>::layout(build(i1, i2, j.begin(), j.end())) {
    }
    template<typename I, typename J>
    indexed_layout(std::initializer_list<I> i, J j1, J j2) : 
      layout<T>::layout(build(i.begin(), i.end(), j1, j2)) {
    }
    template<typename I, typename J>
    indexed_layout(std::initializer_list<I> i, std::initializer_list<J> j) : 
      layout<T>::layout(build(i.begin(), i.end(), j.begin(), j.end())) {
    }
  };

  //--------------------------------------------------------------------

  template<typename T, int n>
  class subarray_layout : public layout<T> {
  public:
    typedef enum { C_order=MPI_ORDER_C, Fortran_order=MPI_ORDER_FORTRAN } order_type;
  private:
    using layout<T>::type;
    static MPI_Datatype build(const vector<int, n> &sizes,
			      const vector<int, n> &subsizes,
			      const vector<int, n> &starts, 
			      order_type order) {
      MPI_Datatype new_type;
      MPI_Type_create_subarray(n, const_cast<int *>(&sizes[0]), 
			       const_cast<int *>(&subsizes[0]), 
			       const_cast<int *>(&starts[0]),
			       order,
			       datatype_traits<T>::get_datatype(), &new_type);
      return new_type;
    }
  public:
    subarray_layout(const vector<int, n> &sizes,
		    const vector<int, n> &subsizes,
		    const vector<int, n> &starts, 
		    order_type order) : 
      layout<T>::layout(build(sizes, subsizes, starts, order)) {
    }		   
  };

  //--------------------------------------------------------------------

  template<typename T>
  struct datatype_traits<layout<T> > {
    static MPI_Datatype get_datatype(const layout<T> &l) {
      return l.type;
    }
  };

}

#endif
