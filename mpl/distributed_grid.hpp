#if !(defined MPL_DISTRIBUTED_GRID)

#define MPL_DISTRIBUTED_GRID

#include <vector>
#include <mpl/cart_comm.hpp>

namespace mpl {
  
  template<std::size_t dim, typename T, typename A=std::allocator<T> >
  class distributed_grid {
  public:
    typedef std::vector<T, A> vector_type;
    typedef typename vector_type::value_type value_type;
    typedef typename vector_type::allocator_type allocator_type;
    typedef typename vector_type::size_type size_type;
    typedef typename vector_type::difference_type difference_type;
    typedef typename vector_type::reference reference;
    typedef typename vector_type::const_reference const_reference;
    typedef typename vector_type::pointer pointer;
    typedef typename vector_type::const_pointer const_pointer;
  private:
    const cart_communicator &C;
    std::vector<size_type> gsize_, gbegin_, gend_, size_, oend_, overlap_;
    vector_type v;
    std::vector<int> C_size, C_coord;
    std::vector<subarray_layout<T> > left_mirror_layout, right_mirror_layout,
       left_border_layout, right_border_layout;

    size_type gbegin(size_type n, int comm_size, int comm_coord) const {
      return n*comm_coord/comm_size;
    }
    size_type gend(size_type n, int comm_size, int comm_coord) const {
      return n*(comm_coord+1)/comm_size;
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

    distributed_grid(const cart_communicator &C, const sizes &size) : 
      C(C), gsize_(size.size_), gbegin_(dim), gend_(dim), size_(dim), oend_(dim), overlap_(size.overlap_) {
#if defined MPL_DEBUG
      if (C.dim()!=dim or gsize_.size()!=dim or overlap_.size()!=dim)
	throw invalid_dim();
#endif
      C_size=C.dims();
      C_coord=C.coords();
      size_type vol=1;
      for (std::size_t i=0; i<dim; ++i) {
	gbegin_[i]=gbegin(gsize_[i], C_size[i], C_coord[i]);
       	gend_[i]=gend(gsize_[i], C_size[i], C_coord[i]);
       	size_[i]=gend_[i]-gbegin_[i];
	oend_[i]=size_[i]+2*overlap_[i];
        vol*=size_[i]+2*overlap_[i];
      }
      v.resize(vol);
      for (std::size_t i=0; i<dim; ++i) {
	typename subarray_layout<T>::parameter par_m_l, par_m_r, par_b_l, par_b_r;
	for (std::size_t j=dim-1; true; --j) {
	  if (j==i) {
	    par_m_l.add(oend_[j], overlap_[j], 0);
	    par_m_r.add(oend_[j], overlap_[j], size_[j]+overlap_[j]);
	    par_b_l.add(oend_[j], overlap_[j], overlap_[j]);
	    par_b_r.add(oend_[j], overlap_[j], size_[j]);
	  } else {
	    par_m_l.add(oend_[j], size_[j], overlap_[j]);
	    par_m_r.add(oend_[j], size_[j], overlap_[j]);
	    par_b_l.add(oend_[j], size_[j], overlap_[j]);
	    par_b_r.add(oend_[j], size_[j], overlap_[j]);
	  }
	  if (j==0)
	    break;
	}
	left_mirror_layout.push_back(subarray_layout<T>(par_m_l));
	right_mirror_layout.push_back(subarray_layout<T>(par_m_r));
	left_border_layout.push_back(subarray_layout<T>(par_b_l));
	right_border_layout.push_back(subarray_layout<T>(par_b_r));
      }
    }
    size_type gsize(size_type d) const {
      return gsize_[d];
    }
    size_type gbegin(size_type d) const {
      return gbegin_[d];
    };
    size_type gend(size_type d) const {
      return gend_[d];
    };
    size_type size(size_type d) const {
      return size_[d];
    }
    size_type begin(size_type d) const {
      return overlap_[d];
    };
    size_type end(size_type d) const {
      return size_[d]+overlap_[d];
    };
    size_type obegin(size_type d) const {
      return 0;
    };
    size_type oend(size_type d) const {
      return oend_[d];
    };
    size_type gindex(size_type d, size_type i) const {
      return gbegin(d)+i-begin(d);
    }
    reference operator()(size_type x) {
      static_assert(dim==1, "invalid dimension");
      return v[x];
    }
    const_reference operator()(size_type x) const {
      static_assert(dim==1, "invalid dimension");
      return v[x];
    }
    reference operator()(size_type x, size_type y) {
      static_assert(dim==2, "invalid dimension");
      return v[x+oend_[0]*y];
    }
    const_reference operator()(size_type x, size_type y) const {
      static_assert(dim==2, "invalid dimension");
      return v[x+oend_[0]*y];
    }
    reference operator()(size_type x, size_type y, size_type z) {
      static_assert(dim==3, "invalid dimension");
      return v[x+oend_[0]*(y+oend_[1]*z)];
    }
    const_reference operator()(size_type x, size_type y, size_type z) const {
      static_assert(dim==3, "invalid dimension");
      return v[x+oend_[0]*(y+oend_[1]*z)];
    }
    pointer data() {
      return v.data();
    }
    const_pointer data() const {
      return v.data();
    }
    void update_overlap() {
      shift_ranks ranks;
      for (std::size_t i=0; i<dim; ++i) {
	irequest_pool r;
	// send to left
	ranks=C.shift(i, -1);
	// C.sendrecv(data(), left_border_layout[i], ranks.dest, 0,
	// 	   data(), right_mirror_layout[i], ranks.source, 0);
	r.push(C.isend(data(), left_border_layout[i], ranks.dest));
	r.push(C.irecv(data(), right_mirror_layout[i], ranks.source));
	// send to right
	ranks=C.shift(i, +1);
	// C.sendrecv(data(), right_border_layout[i], ranks.dest, 0,
	// 	   data(), left_mirror_layout[i], ranks.source, 0);
	r.push(C.isend(data(), right_border_layout[i], ranks.dest));
	r.push(C.irecv(data(), left_mirror_layout[i], ranks.source));
	r.waitall();
      }
    }
  };
  
}

#endif
