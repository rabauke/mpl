#if !(defined MPL_CARTESIAN_COMMUNICATOR_HPP)

#define MPL_CARTESIAN_COMMUNICATOR_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <tuple>

namespace mpl {

  /// \brief Helper class to represent source and destination ranks within a Cartesian
  /// communicator.
  /// \see cartesian_communicator::shift
  struct shift_ranks {
    int source{0};
    int destination{0};
  };

  //--------------------------------------------------------------------

  class cartesian_communicator : public impl::topo_communicator {
  public:
    enum class periodicity { non_periodic, periodic };
    static constexpr periodicity non_periodic = periodicity::non_periodic;
    static constexpr periodicity periodic = periodicity::periodic;

    enum class included_tag : int { excluded = 0, included = 1 };
    static constexpr included_tag excluded = included_tag::excluded;
    static constexpr included_tag included = included_tag::included;


    class coordinate_type : private std::vector<int> {
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
      using base::push_back;
      using base::operator=;
      using base::operator[];
    };


    class periodicity_vector : private std::vector<periodicity> {
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
      using base::push_back;
      using base::operator=;
      using base::operator[];
    };


    class included_vector : private std::vector<included_tag> {
      using base = std::vector<included_tag>;

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
      using base::push_back;
      using base::operator=;
      using base::operator[];
    };


    class dimensions {
      std::vector<int> dims_, periodic_;

    public:
      using size_type = std::vector<int>::size_type;

      dimensions(std::initializer_list<std::tuple<int, periodicity>> list) {
        for (const auto &i : list)
          add(std::get<int>(i), std::get<periodicity>(i));
      }

      void add(int dim, periodicity p) {
        dims_.push_back(dim);
        periodic_.push_back(p == periodicity::periodic);
      }

      [[nodiscard]] size_type size() const { return dims_.size(); }
      [[nodiscard]] int size(size_type i) const { return dims_[i]; }
      [[nodiscard]] bool is_periodic(size_type i) const { return periodic_[i] != 0; }

      friend class cartesian_communicator;
      friend dimensions dims_create(int, dimensions);
    };


    cartesian_communicator() = default;

    explicit cartesian_communicator(const communicator &old_comm, const dimensions &par,
                                    bool reorder = true) {
      MPI_Cart_create(old_comm.comm_, par.dims_.size(), par.dims_.data(), par.periodic_.data(),
                      reorder, &comm_);
    }

    explicit cartesian_communicator(const cartesian_communicator &old_comm,
                                    const included_vector &is_included) {
#if defined MPL_DEBUG
      if (is_included.size() != old_comm.dimensionality())
        throw invalid_size();
#endif
      MPI_Cart_sub(old_comm.comm_, reinterpret_cast<const int *>(is_included.data()), &comm_);
    }

    cartesian_communicator(cartesian_communicator &&other) noexcept {
      comm_ = other.comm_;
      other.comm_ = MPI_COMM_SELF;
    }

    void operator=(const cartesian_communicator &) = delete;

    cartesian_communicator &operator=(cartesian_communicator &&other) noexcept {
      if (this != &other) {
        int result_1{0}, result_2{0};
        MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
        MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
        if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
          MPI_Comm_free(&comm_);
        comm_ = other.comm_;
        other.comm_ = MPI_COMM_SELF;
      }
      return *this;
    }

    [[nodiscard]] int dimensionality() const {
      int t_dimensionality{0};
      MPI_Cartdim_get(this->comm_, &t_dimensionality);
      return t_dimensionality;
    }

    using communicator::rank;

    [[nodiscard]] int rank(const coordinate_type &c) const {
      int t_rank{0};
      MPI_Cart_rank(this->comm_, c.data(), &t_rank);
      return t_rank;
    }

    [[nodiscard]] coordinate_type coordinate(int rank) const {
      coordinate_type coordinates(dimensionality());
      MPI_Cart_coords(comm_, rank, coordinates.size(), coordinates.data());
      return coordinates;
    }

    [[nodiscard]] coordinate_type coordinate() const {
      const int t_dimensionality{dimensionality()};
      coordinate_type dimensions(t_dimensionality), is_periodic_as_int(t_dimensionality),
          coordinate(t_dimensionality);
      MPI_Cart_get(comm_, t_dimensionality, dimensions.data(), is_periodic_as_int.data(),
                   coordinate.data());
      return coordinate;
    }

    [[nodiscard]] coordinate_type dimension() const {
      const int t_dimensionality{dimensionality()};
      coordinate_type dimensions(t_dimensionality), is_periodic_as_int(t_dimensionality),
          coordinates(t_dimensionality);
      MPI_Cart_get(comm_, t_dimensionality, dimensions.data(), is_periodic_as_int.data(),
                   coordinates.data());
      return dimensions;
    }

    [[nodiscard]] periodicity_vector is_periodic() const {
      const int t_dimensionality{dimensionality()};
      coordinate_type dimensions(t_dimensionality), is_periodic_as_int(t_dimensionality),
          coordinates(t_dimensionality);
      MPI_Cart_get(comm_, t_dimensionality, dimensions.data(), is_periodic_as_int.data(),
                   coordinates.data());
      periodicity_vector is_periodic(t_dimensionality);
      for (int i = 0; i < t_dimensionality; ++i)
        is_periodic[i] =
            is_periodic_as_int[i] ? periodicity::periodic : periodicity::non_periodic;
      return is_periodic;
    }

    [[nodiscard]] shift_ranks shift(int direction, int displacement) const {
      shift_ranks ranks;
      MPI_Cart_shift(this->comm_, direction, displacement, &ranks.source, &ranks.destination);
      return ranks;
    }
  };

  //--------------------------------------------------------------------

  inline cartesian_communicator::dimensions dims_create(
      int size, cartesian_communicator::dimensions par) {
    if (MPI_Dims_create(size, static_cast<int>(par.dims_.size()), par.dims_.data()) !=
        MPI_SUCCESS)
      throw invalid_dim();
    return par;
  }

}  // namespace mpl

#endif
