#if !(defined MPL_CART_COMM_HPP)

#define MPL_CART_COMM_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <tuple>

namespace mpl {

  struct shift_ranks {
    int source, dest;
  };
  
  //--------------------------------------------------------------------

  class cart_communicator : public communicator {
  public:
    class parameter {
      std::vector<int> dims_, periodic_;
    public:
      typedef std::vector<int>::size_type size_type;
      parameter(std::initializer_list<std::pair<int, bool>> list) {
	for (const std::pair<int, bool> &i : list) 
	  add(i.first, i.second);
      }
      void add(int dim, bool p) {
	dims_.push_back(dim);
	periodic_.push_back(p);
      }
      int dims(size_type i) const {
	return dims_[i];
      }
      bool periodic(size_type i) const {
	return periodic_[i];
      }
      friend class cart_communicator;
      friend parameter dims_create(int, parameter);
    };
    cart_communicator(const communicator &old_comm,
		      const parameter &par,
		      bool reorder=true) {
      MPI_Cart_create(old_comm.comm, par.dims_.size(), par.dims_.data(), par.periodic_.data(), reorder, &comm);
    }
    cart_communicator(const cart_communicator &old_comm,
		      const std::vector<int> &remain_dims) {
      MPI_Cart_sub(old_comm.comm, remain_dims.data(), &comm);
    }
    int dim() const {
      int ndims;
      MPI_Cartdim_get(comm, &ndims);
      return ndims;
    }
    int rank(const std::vector<int> &coords) const {
      int rank_;
      MPI_Cart_rank(comm, coords.data(), &rank_);
      return rank_;
    }
    std::vector<int> coords(int rank) const {
      std::vector<int> coords_(dim());
      MPI_Cart_coords(comm, rank, coords_.size(), coords_.data());
      return coords_;
    }
    std::vector<int> coords() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return coords_;
    }
    std::vector<int> dims() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return dims_;
    }
    std::vector<int> periodic() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return periodic_;
    }
    shift_ranks shift(int direction, int disp) const {
      int rank_source, rank_dest;
      MPI_Cart_shift(comm, direction, disp, &rank_source, &rank_dest);
      return {rank_source, rank_dest};
    }
  };

  //--------------------------------------------------------------------

  cart_communicator::parameter dims_create(int size, cart_communicator::parameter par) {
    MPI_Dims_create(size, par.dims_.size(), par.dims_.data());
    return par;
  }

}

#endif
