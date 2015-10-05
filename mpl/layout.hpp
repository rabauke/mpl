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
      if (type!=MPI_DATATYPE_NULL)
	MPI_Type_commit(&type);
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
    layout(layout &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
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
    layout & operator=(layout &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      return *this;
    }
    void resize(std::ptrdiff_t lb, std::ptrdiff_t extent) {
      if (type!=MPI_DATATYPE_NULL) {
	MPI_Datatype newtype;
	MPI_Type_create_resized(type, lb, extent, &newtype);
	MPI_Type_commit(&newtype);
	MPI_Type_free(&type);
	type=newtype;
      }
    }
    void swap(layout &l) {
      std::swap(type, l.type);
    }
    ~layout() {
      if (type!=MPI_DATATYPE_NULL)
	MPI_Type_free(&type);
    }
    friend struct datatype_traits<layout<T>>;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class null_layout : public layout<T> {
    using layout<T>::type;
  public:
    null_layout() : layout<T>::layout(MPI_DATATYPE_NULL) {
    }
    void swap(null_layout<T> &other) {
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
  public:
    empty_layout() : 
      layout<T>::layout(build()) {
    }
    empty_layout(const empty_layout &l) {
      MPI_Type_dup(l.type, &type);
    }
    empty_layout<T> & operator=(const empty_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    void swap(empty_layout<T> &other) {
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class contiguous_layout : public layout<T> {
    using layout<T>::type;
    using layout<T>::resize;
    static MPI_Datatype build(int count, 
			      MPI_Datatype old_type=datatype_traits<T>::get_datatype()) {
      MPI_Datatype new_type;
      MPI_Type_contiguous(count, old_type, &new_type);
      return new_type;
    }
    const int count;
    const bool simple;
    int size() const {
      return count;
    }
    bool is_simple() const {
      return simple;
    }
  public:
    explicit contiguous_layout(int c=0) : 
      layout<T>::layout(build(c)), count(c), simple(true) {
    }
    explicit contiguous_layout(int c, const layout<T> &other) : 
      layout<T>::layout(build(c, other.type)), count(c), simple(false) {
    }
    explicit contiguous_layout(int c, const contiguous_layout<T> &other) : 
      layout<T>::layout(build(c, other.type)), count(other.simple ? other.c*c : c), simple(other.simple) {
    }
    contiguous_layout(const contiguous_layout<T> &l) : count(l.count), simple(l.simple) {
      MPI_Type_dup(l.type, &type);
    }
    contiguous_layout(contiguous_layout &&l) : count(l.count), simple(l.simple) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      l.size=0;
      l.simple=false;
    }
    contiguous_layout<T> & operator=(const contiguous_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    contiguous_layout<T> & operator=(contiguous_layout<T> &&l) {
      type=l.type;
      size=l.size;
      simple=l.simple;
      l.type=MPI_DATATYPE_NULL;
      l.size=0;
      l.simple=false;
      return *this;
    }
    void swap(contiguous_layout<T> &other) {
      std::swap(type, other.type);
      std::swap(count, other.count);
      std::swap(simple, other.simple);
    }    
    friend class communicator;
  };

  //--------------------------------------------------------------------

  template<typename T>
  class vector_layout : public layout<T> {
    using layout<T>::type;
    using layout<T>::resize;
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
    }
    explicit vector_layout(int count, int blocklength, int stride) : 
      layout<T>::layout(build(count, blocklength, stride)) {
    }
    explicit vector_layout(int count, int blocklength, int stride, const layout<T> &other) : 
      layout<T>::layout(build(count, blocklength, stride, other.type)) {
    }
    vector_layout(const vector_layout<T> &l) {
      MPI_Type_dup(l.type, &type);
    }
    vector_layout(vector_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
    }
    vector_layout<T> & operator=(const vector_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    vector_layout<T> & operator=(vector_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      return *this;
    }      
    void swap(vector_layout<T> &other) {
      std::swap(type, other.type);
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class indexed_layout : public layout<T> {
    using layout<T>::type;
    using layout<T>::resize;
  public:
    class parameter {
      std::vector<int> blocklengths, displacements;
    public:
      parameter() {
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
    }
    explicit indexed_layout(const parameter &par) :
      layout<T>::layout(build(par)) {
    }
    explicit indexed_layout(const parameter &par, const layout<T> &other) :
      layout<T>::layout(build(par, other.type)) {
    }
    indexed_layout(const indexed_layout<T> &l) {
      MPI_Type_dup(l.type, &type);
    }
    indexed_layout(indexed_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
    }
    indexed_layout<T> & operator=(const indexed_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    indexed_layout<T> & operator=(indexed_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      return *this;
    }      
    void swap(indexed_layout<T> &other) {
      std::swap(type, other.type);
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class indexed_block_layout : public layout<T> {
    using layout<T>::type;
    using layout<T>::resize;
  public:
    class parameter {
      std::vector<int> displacements;
    public:
      parameter() {
      }
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
    }
    explicit indexed_block_layout(int blocklengths, const parameter &par) :
      layout<T>::layout(build(blocklengths, par)) {
    }
    explicit indexed_block_layout(int blocklengths, const parameter &par, const layout<T> &other) :
      layout<T>::layout(build(blocklengths, par, other.type)) {
    }
    indexed_block_layout(const indexed_block_layout<T> &l) {
      MPI_Type_dup(l.type, &type);
    }
    indexed_block_layout(indexed_block_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
    }
    indexed_block_layout<T> & operator=(const indexed_block_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    indexed_block_layout<T> & operator=(indexed_block_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      return *this;
    }      
    void swap(indexed_block_layout<T> &other) {
      std::swap(type, other.type);
    }
  };

  //--------------------------------------------------------------------

  enum class array_orders { C_order=MPI_ORDER_C, Fortran_order=MPI_ORDER_FORTRAN }; 

  template<typename T>
  class subarray_layout : public layout<T> {
    using layout<T>::type;
    using layout<T>::resize;
  public:
    class parameter {
      std::vector<int> sizes, subsizes, starts;
      array_orders order_=array_orders::C_order;
    public:
      parameter() {
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
      void order(array_orders new_order) {
	order_=new_order;
      }
      array_orders order() const {
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
				 static_cast<int>(par.order()),
				 old_type, &new_type);
      else 
	new_type=build();
      return new_type;
    }
  public:
    subarray_layout() : layout<T>::layout(build()) {
    }
    explicit subarray_layout(const parameter &par) : 
      layout<T>::layout(build(par)) {
    }		   
    explicit subarray_layout(const parameter &par, const layout<T> &other) : 
      layout<T>::layout(build(par, other.type)) {
    }		   
    subarray_layout(const subarray_layout<T> &l) {
      MPI_Type_dup(l.type, &type);
    }
    subarray_layout(subarray_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
    }
    subarray_layout<T> & operator=(const subarray_layout<T> &l) {
      if (this!=&l) {
	MPI_Type_free(&type);
	MPI_Type_dup(l.type, &type);
      }
      return *this;
    }
    subarray_layout<T> & operator=(subarray_layout<T> &&l) {
      type=l.type;
      l.type=MPI_DATATYPE_NULL;
      return *this;
    }      
    void swap(subarray_layout<T> &other) {
      std::swap(type, other.type);
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  struct datatype_traits<layout<T>> {
    static MPI_Datatype get_datatype(const layout<T> &l) {
      return l.type;
    }
  };

  //--------------------------------------------------------------------

  template<typename T>
  class layouts : private std::vector<layout<T>> {
    typedef std::vector<layout<T>> base;
  public:
    typedef typename base::size_type size_type;
    explicit layouts(size_type n=0) : base(n, empty_layout<T>()) {
    }
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
    const layout<T> * operator()() const {
      return base::data();
    }
  };

}

#endif
