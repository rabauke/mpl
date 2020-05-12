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

  class cart_communicator : public detail::topo_communicator {
  public:
    enum class periodicity { periodic, nonperiodic };
    static constexpr periodicity periodic = periodicity::periodic;
    static constexpr periodicity nonperiodic = periodicity::nonperiodic;

    class coords_type : private std::vector<int> {
      using base = std::vector<int>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::data;
      using base::operator=;
      using base::operator[];
    };

    class periodicities_type : private std::vector<periodicity> {
      using base = std::vector<periodicity>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::data;
      using base::operator=;
      using base::operator[];
    };

    class sizes {
      std::vector<int> dims_, periodic_;

    public:
      using size_type = std::vector<int>::size_type;

      sizes(std::initializer_list<std::pair<int, periodicity>> list) {
        for (const auto &i : list)
          add(i.first, i.second);
      }

      void add(int dim, periodicity p) {
        dims_.push_back(dim);
        periodic_.push_back(p == periodicity::periodic);
      }

      int dims(size_type i) const { return dims_[i]; }

      bool periodic(size_type i) const { return periodic_[i] != 0; }

      friend class cart_communicator;

      friend sizes dims_create(int, sizes);
    };

    cart_communicator() = default;

    explicit cart_communicator(const communicator &old_comm, const sizes &par,
                               bool reorder = true) {
      MPI_Cart_create(old_comm.comm, par.dims_.size(), par.dims_.data(), par.periodic_.data(),
                      reorder, &comm);
    }

    explicit cart_communicator(const cart_communicator &old_comm,
                               const coords_type &remain_dims) {
#if defined MPL_DEBUG
      if (remain_dims.size() != old_comm.dim())
        throw invalid_size();
#endif
      MPI_Cart_sub(old_comm.comm, remain_dims.data(), &comm);
    }

    cart_communicator(cart_communicator &&other) noexcept {
      comm = other.comm;
      other.comm = MPI_COMM_SELF;
    }

    void operator=(const cart_communicator &) = delete;

    cart_communicator &operator=(cart_communicator &&other) noexcept {
      if (this != &other) {
        int result1, result2;
        MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
        MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
        if (result1 != MPI_IDENT and result2 != MPI_IDENT)
          MPI_Comm_free(&comm);
        comm = other.comm;
        other.comm = MPI_COMM_SELF;
      }
      return *this;
    }

    int dim() const {
      int ndims;
      MPI_Cartdim_get(comm, &ndims);
      return ndims;
    }

    using communicator::rank;

    int rank(const coords_type &c) const {
      int rank_;
      MPI_Cart_rank(comm, c.data(), &rank_);
      return rank_;
    }

    coords_type coords(int rank) const {
      coords_type coords_(dim());
      MPI_Cart_coords(comm, rank, coords_.size(), coords_.data());
      return coords_;
    }

    coords_type coords() const {
      int ndims(dim());
      coords_type dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return coords_;
    }

    coords_type dims() const {
      int ndims(dim());
      coords_type dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return dims_;
    }

    periodicities_type is_periodic() const {
      int ndims(dim());
      coords_type dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      periodicities_type periodic(ndims);
      for (int i = 0; i < ndims; ++i)
        periodic[i] = periodic_[i] ? periodicity::periodic : periodicity::nonperiodic;
      return periodic;
    }

    shift_ranks shift(int direction, int disp) const {
      int rank_source, rank_dest;
      MPI_Cart_shift(comm, direction, disp, &rank_source, &rank_dest);
      return {rank_source, rank_dest};
    }
  };

  //--------------------------------------------------------------------

  inline cart_communicator::sizes dims_create(int size, cart_communicator::sizes par) {
    if (MPI_Dims_create(size, par.dims_.size(), par.dims_.data()) != MPI_SUCCESS)
      throw invalid_dim();
    return par;
  }

}  // namespace mpl

#endif
