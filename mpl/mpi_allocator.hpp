#ifndef MPL_MPI_ALLOCATOR_HPP

#define MPL_MPI_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <mpi.h>
#include <mpl/error.hpp>


namespace mpl {

  namespace detail {

    class mpi_allocator_info {
      MPI_Info info_{MPI_INFO_NULL};

    public:
      mpi_allocator_info() = default;

      mpi_allocator_info(const mpi_allocator_info &) = delete;

      ~mpi_allocator_info() {
        if (info_ != MPI_INFO_NULL)
          MPI_Info_free(&info_);
      }

      void set_minimum_alignment(std::size_t alignment) {
        check(MPI_Info_create(&info_));
        const std::string value{std::to_string(alignment)};
        check(MPI_Info_set(info_, "mpi_minimum_memory_alignment", value.c_str()));
      }

      [[nodiscard]] MPI_Info native_handle() const noexcept {
        return info_;
      }
    };

  }  // namespace detail

  /// Allocates storage through MPI's memory allocation routines.
  template<typename T>
  class mpi_allocator {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;

    template<typename U>
    struct rebind {
      using other = mpi_allocator<U>;
    };

    constexpr mpi_allocator() noexcept = default;

    template<typename U>
    constexpr mpi_allocator(const mpi_allocator<U> &) noexcept {
    }

    [[nodiscard]] T *allocate(size_type n) {
      if (n > max_size())
        throw std::bad_array_new_length{};
      if (n == 0)
        return nullptr;

      detail::mpi_allocator_info allocation_info;
      if constexpr (alignof(T) > alignof(std::max_align_t))
        allocation_info.set_minimum_alignment(alignof(T));

      void *buffer{nullptr};
      const MPI_Aint bytes{static_cast<MPI_Aint>(n * sizeof(T))};
      detail::check(MPI_Alloc_mem(bytes, allocation_info.native_handle(), &buffer));
      return static_cast<T *>(buffer);
    }

    void deallocate(T *pointer, size_type) noexcept {
      if (pointer != nullptr)
        static_cast<void>(MPI_Free_mem(pointer));
    }

    [[nodiscard]] static constexpr size_type max_size() noexcept {
      constexpr std::uintmax_t size_max{
          static_cast<std::uintmax_t>(std::numeric_limits<size_type>::max())};
      constexpr std::uintmax_t mpi_aint_max{
          static_cast<std::uintmax_t>(std::numeric_limits<MPI_Aint>::max())};
      constexpr std::uintmax_t max_bytes{size_max < mpi_aint_max ? size_max : mpi_aint_max};
      return static_cast<size_type>(max_bytes / sizeof(T));
    }
  };

  template<typename T, typename U>
  [[nodiscard]] constexpr bool operator==(const mpi_allocator<T> &,
                                          const mpi_allocator<U> &) noexcept {
    return true;
  }

  template<typename T, typename U>
  [[nodiscard]] constexpr bool operator!=(const mpi_allocator<T> &,
                                          const mpi_allocator<U> &) noexcept {
    return false;
  }

}  // namespace mpl

#endif  // MPL_MPI_ALLOCATOR_HPP
