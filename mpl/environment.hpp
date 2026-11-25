#ifndef MPL_ENVIRONMENT_HPP

#define MPL_ENVIRONMENT_HPP

#include <array>
#include <string>
#include <memory>
#include <vector>
#include <mpi.h>
#include <mpl/error.hpp>


namespace mpl {

  /// Represents the various levels of thread support that the underlying MPI
  /// implementation may provide.
  enum class threading_modes {
    /// the application is single-threaded
    single = MPI_THREAD_SINGLE,
    /// the application is multi-threaded, however, all MPL calls will be issued from the main
    /// thread only
    funneled = MPI_THREAD_FUNNELED,
    /// the application is multi-threaded and any thread may issue MPL calls, however,
    /// different threads will never issue MPL calls at the same time
    serialized = MPI_THREAD_SERIALIZED,
    /// the application is multi-threaded, any thread may issue MPI calls and different threads
    /// may issue MPL calls at the same time
    multiple = MPI_THREAD_MULTIPLE
  };

  namespace environment {

    namespace detail {

      class env {
        class initializer {
          int thread_mode_{MPI_THREAD_SINGLE};
          bool is_externally_initialized_{false};

        public:
          initializer() {
            int is_initialized{0};
            mpl::detail::check(MPI_Initialized(&is_initialized));
            is_externally_initialized_ = is_initialized != 0;
            if (is_externally_initialized_)
              mpl::detail::check(MPI_Query_thread(&thread_mode_));
            else
              mpl::detail::check(
                  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &thread_mode_));
          }

          ~initializer() {
            if (not is_externally_initialized_)
              MPI_Finalize();
          }

          [[nodiscard]] threading_modes thread_mode() const {
            switch (thread_mode_) {
              case MPI_THREAD_SINGLE:
                return threading_modes::single;
              case MPI_THREAD_FUNNELED:
                return threading_modes::funneled;
              case MPI_THREAD_SERIALIZED:
                return threading_modes::serialized;
              case MPI_THREAD_MULTIPLE:
                return threading_modes::multiple;
              default:
                return threading_modes::single;  // make compiler happy
            }
          }
        };

        initializer init;
        mpl::communicator comm_world_, comm_self_;

      public:
        env() : comm_world_{MPI_COMM_WORLD}, comm_self_{MPI_COMM_SELF} {
          int size;
          mpl::detail::check(MPI_Comm_size(MPI_COMM_WORLD, &size));
        }

        env(const env &) = delete;

        env &operator=(const env &) = delete;

        [[nodiscard]] int tag_up() const {
          void *p;
          int flag;
          mpl::detail::check(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &p, &flag));
          return *static_cast<int *>(p);
        }

        [[nodiscard]] threading_modes threading_mode() const {
          return init.thread_mode();
        }

        [[nodiscard]] bool is_thread_main() const {
          int res;
          mpl::detail::check(MPI_Is_thread_main(&res));
          return static_cast<bool>(res);
        }

        [[nodiscard]] bool wtime_is_global() const {
          void *p;
          int flag;
          mpl::detail::check(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &p, &flag));
          return *static_cast<int *>(p);
        }

        [[nodiscard]] const communicator &comm_world() const {
          return comm_world_;
        }

        [[nodiscard]] const communicator &comm_self() const {
          return comm_self_;
        }

        [[nodiscard]] std::string processor_name() const {
          std::array<char, MPI_MAX_PROCESSOR_NAME> name;
          int len;
          mpl::detail::check(MPI_Get_processor_name(name.data(), &len));
          return name.data();
        }

        [[nodiscard]] std::string library_version() const {
          std::array<char, MPI_MAX_LIBRARY_VERSION_STRING> version;
          int len;
          mpl::detail::check(MPI_Get_library_version(version.data(), &len));
          return {version.data(), static_cast<std::string::size_type>(len)};
        }

        [[nodiscard]] double wtime() const {
          return MPI_Wtime();
        }

        [[nodiscard]] double wtick() const {
          return MPI_Wtick();
        }

        void buffer_attach(void *buff, ssize_type size) const {
          mpl::detail::check(MPI_Buffer_attach_c(buff, size));
        }

        [[nodiscard]] std::pair<void *, ssize_type> buffer_detach() const {
          void *buff;
          MPI_Count size;
          mpl::detail::check(MPI_Buffer_detach_c(&buff, &size));
          return {buff, size};
        }

        void buffer_flush() const {
          mpl::detail::check(MPI_Buffer_flush());
        }

        [[nodiscard]] irequest buffer_iflush() const {
          MPI_Request req;
          mpl::detail::check(MPI_Buffer_iflush(&req));
          return mpl::impl::base_irequest{req};
        }
      };

      //----------------------------------------------------------------

      [[nodiscard]] inline const env &get_env() {
        static env the_env;
        return the_env;
      }

    }  // namespace detail

    //------------------------------------------------------------------

    /// Determines the highest level of thread support that is provided by the underlying
    /// MPI implementation.
    /// \return supported threading level
    [[nodiscard]] inline threading_modes threading_mode() {
      return detail::get_env().threading_mode();
    }

    /// Determines if the current thread is the main thread, i.e., the thread that has
    /// initialized the MPI environment of the underlying MPI implementation.
    /// \return true if the current thread is the main thread
    [[nodiscard]] inline bool is_thread_main() {
      return detail::get_env().is_thread_main();
    }

    /// Determines if time values given by <tt>\ref wtime</tt> are synchronized with each other
    /// for all processes of the communicator given in <tt>\ref comm_world</tt>.
    /// \return true if times are synchronized
    /// \see <tt>\ref wtime</tt>
    [[nodiscard]] inline bool wtime_is_global() {
      return detail::get_env().wtime_is_global();
    }

    /// Provides access to a predefined communicator that allows communication with
    /// all processes.
    /// \return communicator to communicate with any other process
    [[nodiscard]] inline const communicator &comm_world() {
      return detail::get_env().comm_world();
    }

    /// Provides access to a predefined communicator that includes only the calling
    /// process itself.
    /// \return communicator including only the process itself
    [[nodiscard]] inline const communicator &comm_self() {
      return detail::get_env().comm_self();
    }

    /// Gives a unique specifier, the processor name, for the actual (physical) node.
    /// \return name of the node
    /// \note The name is determined by the underlying MPI implementation, i.e., it is
    /// implementation defined and may be different for different MPI implementations.
    [[nodiscard]] inline std::string processor_name() {
      return detail::get_env().processor_name();
    }

    /// Gets the version string of the MPI library in use.
    /// \return version string reported by the MPI implementation
    [[nodiscard]] inline std::string library_version() {
      return detail::get_env().library_version();
    }

    /// Get time.
    /// \return number of seconds of elapsed wall-clock time since some time in the past
    [[nodiscard]] inline double wtime() {
      return detail::get_env().wtime();
    }

    /// Get resolution of time given by \c wtime.
    /// \return resolution of \c wtime in seconds.
    /// \see \c wtime
    [[nodiscard]] inline double wtick() {
      return detail::get_env().wtick();
    }

    /// Provides a process-wide buffer in the user's memory for buffering outgoing messages.
    /// \param buff pointer to user-provided buffer
    /// \param size size of the buffer in bytes, must be non-negative
    /// \see \c buffer_detach
    inline void buffer_attach(void *buff, MPI_Count size) {
      return detail::get_env().buffer_attach(buff, size);
    }

    /// Detaches the process-wide buffer currently associated with MPL.
    /// \return pair representing the buffer location and size, i.e., the parameters provided to
    /// <tt>\ref buffer_attach</tt>
    /// \see \c buffer_attach
    inline std::pair<void *, MPI_Count> buffer_detach() {
      return detail::get_env().buffer_detach();
    }

    /// Completes all outstanding buffered sends using the process-wide buffer.
    /// \note Complete this operation before detaching or reusing the buffer storage.
    inline void buffer_flush() {
      detail::get_env().buffer_flush();
    }

    /// Starts completion of all outstanding buffered sends using the process-wide buffer.
    /// \return request that completes when all buffered sends have completed
    /// \note Wait for or test the request before detaching or reusing the buffer storage.
    [[nodiscard]] inline irequest buffer_iflush() {
      return detail::get_env().buffer_iflush();
    }

  }  // namespace environment

  //--------------------------------------------------------------------

  [[nodiscard]] tag_t tag_t::up() {
    return tag_t(environment::detail::get_env().tag_up());
  }

  [[nodiscard]] tag_t tag_t::any() {
    return tag_t(MPI_ANY_TAG);
  }

  //--------------------------------------------------------------------

  /// Buffer manager for buffered send operations.
  /// \note There must be not more than one instance of the class \c bsend_buffer at any time
  /// per process.
  class bsend_buffer {
    void *buff_;

  public:
    /// deleted default constructor
    bsend_buffer() = delete;

    /// deleted copy constructor
    /// \param other buffer manager to copy from
    bsend_buffer(const bsend_buffer &other) = delete;

    /// deleted move constructor
    /// \param other buffer manager to move from
    bsend_buffer(bsend_buffer &&other) = delete;

    /// allocates buffer with specific size using a default-constructed allocator
    /// \param size buffer size in bytes
    /// \note The size given should be the sum of the sizes of all outstanding buffered send
    /// operations that will be sent during the lifetime of the \c bsend_buffer object, plus
    /// <tt>\ref bsend_overhead</tt> for each buffered send operation.  Use
    /// \c communicator::bsend_size to calculate the required buffer size.
    /// \see \c communicator::bsend and \c communicator::ibsend
    explicit bsend_buffer(ssize_type size) {
      std::unique_ptr<void, void (*)(void *)> owner{operator new(size), &operator delete };
      environment::buffer_attach(owner.get(), size);
      buff_ = owner.release();
    }

    /// waits for uncompleted message transfers and frees the buffer
    /// \note A blocking communication operation is performed when an object of type
    /// \c bsend_buffer goes out of scope.
    ~bsend_buffer() {
      try {
        environment::buffer_detach();
      } catch (...) {
        // must not throw
      }
      operator delete(buff_);
    }

    /// deleted copy assignment operator
    /// \param other buffer manager to copy-assign from
    void operator=(const bsend_buffer &other) = delete;

    /// deleted move assignment operator
    /// \param other buffer manager to move-assign from
    void operator=(bsend_buffer &&other) = delete;
  };

  /// Buffer manager for buffered sends on a communicator.
  /// \note The communicator must outlive the guard, and a communicator may have at most one
  /// attached buffer at a time.
  class communicator_buffer {
    const communicator &comm_;
    void *buff_{nullptr};

  public:
    /// deleted default constructor
    communicator_buffer() = delete;

    /// Allocates and attaches a buffer to the given communicator.
    /// \param comm communicator to associate with the allocated buffer
    /// \param size non-negative buffer size in bytes; use \c communicator::bsend_size to calculate
    /// the required size
    /// \note The buffer is detached before its allocation is released. If detachment fails, the
    /// destructor does not throw and leaves the allocation untouched to avoid freeing storage
    /// that MPI may still use; this can leak the allocation.
    explicit communicator_buffer(const communicator &comm, ssize_type size) : comm_{comm} {
      std::unique_ptr<void, void (*)(void *)> owner{operator new(size), &operator delete };
      comm_.attach_buffer(owner.get(), size);
      buff_ = owner.release();
    }

    /// Detaches the buffer before releasing its allocation.
    ~communicator_buffer() noexcept {
      try {
        (void)comm_.detach_buffer();
      } catch (...) {
        // Keep the buffer alive if detachment failed.
        return;
      }
      operator delete(buff_);
    }

    /// deleted copy constructor
    communicator_buffer(const communicator_buffer &) = delete;

    /// deleted move constructor
    communicator_buffer(communicator_buffer &&) = delete;

    /// deleted copy assignment operator
    void operator=(const communicator_buffer &) = delete;

    /// deleted move assignment operator
    void operator=(communicator_buffer &&) = delete;
  };

}  // namespace mpl

#endif
