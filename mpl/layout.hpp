#if !(defined MPL_LAYOUT_HPP)

#define MPL_LAYOUT_HPP

#include <mpi.h>
#include <cstddef>
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
    void resize(std::ptrdiff_t lb, std::ptrdiff_t extent) {
      if (type!=MPI_DATATYPE_NULL) {
	MPI_Datatype newtype;
	MPI_Type_create_resized(type, lb, extent, &newtype);
	MPI_Type_free(&type);
	type=newtype;
      }
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
    empty_layout() : 
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
    static MPI_Datatype build(int count, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_contiguous(count, old_type, &new_type);
      return new_type;
    }
    int count;
  public:
    explicit contiguous_layout(int c=0) : 
      layout<T>::layout(build(c)), count(c) {
      MPI_Type_commit(&type);
    }
    explicit contiguous_layout(int c, const layout<T> &other) : 
      layout<T>::layout(build(c, other.type)), count(c) {
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
    static MPI_Datatype build(int count, int blocklength, int stride, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, stride, old_type, &new_type);
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
    explicit vector_layout(int count, int blocklength, int stride, const layout<T> &other) : 
      layout<T>::layout(build(count, blocklength, stride, other.type)) {
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
  class indexed_layout : public layout<T> {
    using layout<T>::type;
  public:
    class parameter {
      std::vector<int> blocklengths, displacements;
    public:
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
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    static MPI_Datatype build(const parameter &par, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_indexed(par.displacements.size(), par.blocklengths.data(), par.displacements.data(), 
		       old_type, &new_type);
      return new_type;
    }
  public:
    indexed_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    explicit indexed_layout(const parameter &par) :
      layout<T>::layout(build(par)) {
      MPI_Type_commit(&type);
    }
    explicit indexed_layout(const parameter &par, const layout<T> &other) :
      layout<T>::layout(build(par, other.type)) {
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

  template<typename T>
  class indexed_block_layout : public layout<T> {
    using layout<T>::type;
  public:
    class parameter {
      std::vector<int> displacements;
    public:
      parameter(std::initializer_list<int> list) {
	for (int i : list) 
	  add(i);
      }
      void add(int displacement) {
	displacements.push_back(displacement);
      }
      friend class indexed_block_layout;
    };
  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    static MPI_Datatype build(int blocklengths, const parameter &par, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_create_indexed_block(par.displacements.size(), blocklengths, par.displacements.data(), 
				    old_type, &new_type);
      return new_type;
    }
  public:
    indexed_block_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    explicit indexed_block_layout(int blocklengths, const parameter &par) :
      layout<T>::layout(build(blocklengths, par)) {
      MPI_Type_commit(&type);
    }
    explicit indexed_block_layout(int blocklengths, const parameter &par, const layout<T> &other) :
      layout<T>::layout(build(blocklengths, par, other.type)) {
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
  class subarray_layout : public layout<T> {
    using layout<T>::type;
  public:
    typedef enum { C_order=MPI_ORDER_C, Fortran_order=MPI_ORDER_FORTRAN } order_type;
    class parameter {
      std::vector<int> sizes, subsizes, starts;
      order_type order_=C_order;
    public:
      parameter(std::initializer_list<std::array<int, 3>> list) {
	for (const std::array<int, 3> &i : list) 
	  add(i[0], i[1], i[2]);
      }
      void add(int size, int subsize, int start) {
	sizes.push_back(size);
	subsizes.push_back(subsize);
	starts.push_back(start);
      }
      void order(order_type new_order) {
	order_=new_order;
      }
      order_type order() const {
	return order_;
      }
      friend class subarray_layout;
    };
  private:
    static MPI_Datatype build() {
      MPI_Datatype new_type;
      MPI_Type_contiguous(0, datatype_traits<T>::get_datatype(),
 			  &new_type);
      return new_type;
    }
    static MPI_Datatype build(const parameter &par, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      int total_size=1;
      for (std::vector<int>::size_type i=0; i<par.sizes.size(); ++i)
	total_size*=par.subsizes[i];
      if (total_size>0)
	MPI_Type_create_subarray(par.sizes.size(), par.sizes.data(), par.subsizes.data(), par.starts.data(),
				 par.order(),
				 old_type, &new_type);
      else 
	new_type=build();
      return new_type;
    }
  public:
    subarray_layout() : layout<T>::layout(build()) {
      MPI_Type_commit(&type);
    }
    explicit subarray_layout(const parameter &par) : 
      layout<T>::layout(build(par)) {
      MPI_Type_commit(&type);
    }		   
    explicit subarray_layout(const parameter &par, const layout<T> &other) : 
      layout<T>::layout(build(par, other.type)) {
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
