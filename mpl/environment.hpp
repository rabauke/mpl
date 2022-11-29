#if !(defined MPL_ENVIRONMENT_HPP)

#define MPL_ENVIRONMENT_HPP

#include <string>
#include <memory>
#include <vector>
#include <mpi.h>

namespace mpl {

  /// Represents the various levels of thread support that the underlying MPI
  /// implementation may provide.
  enum class threading_modes {
    /// the application is single-threaded
    single = MPI_THREAD_SINGLE,
    /// the application is multi-threaded, however all MPL calls will be issued from the main
    /// thread only
    funneled = MPI_THREAD_FUNNELED,
    /// the application is multi-threaded and any thread may issue MPL calls, however
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

        public:
          initializer() {
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &thread_mode_);
          }

          ~initializer() { MPI_Finalize(); }

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
            }
            return threading_modes::single;  // make compiler happy
          }
        };

        initializer init;
        mpl::communicator comm_world_, comm_self_;

      public:
        env() : init{}, comm_world_{MPI_COMM_WORLD}, comm_self_{MPI_COMM_SELF} {
          int size;
          MPI_Comm_size(MPI_COMM_WORLD, &size);
        }

        env(const env &) = delete;

        env &operator=(const env &) = delete;

        [[nodiscard]] int tag_up() const {
          void *p;
          int flag;
          MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &p, &flag);
          return *static_cast<int *>(p);
        }

        [[nodiscard]] threading_modes threading_mode() const { return init.thread_mode(); }

        [[nodiscard]] bool is_thread_main() const {
          int res;
          MPI_Is_thread_main(&res);
          return static_cast<bool>(res);
        }

        [[nodiscard]] bool wtime_is_global() const {
          void *p;
          int flag;
          MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &p, &flag);
          return *static_cast<int *>(p);
        }

        [[nodiscard]] const communicator &comm_world() const { return comm_world_; }

        [[nodiscard]] const communicator &comm_self() const { return comm_self_; }

        [[nodiscard]] std::string processor_name() const {
          char name[MPI_MAX_PROCESSOR_NAME];
          int len;
          MPI_Get_processor_name(name, &len);
          return name;
        }

        [[nodiscard]] double wtime() const { return MPI_Wtime(); }

        [[nodiscard]] double wtick() const { return MPI_Wtick(); }

        void buffer_attach(void *buff, int size) const { MPI_Buffer_attach(buff, size); }

        [[nodiscard]] std::pair<void *, int> buffer_detach() const {
          void *buff;
          int size;
          MPI_Buffer_detach(&buff, &size);
          return {buff, size};
        }
      };

      //----------------------------------------------------------------

      inline const env &get_env() {
        static env the_env;
        return the_env;
      }

    }  // namespace detail

    //------------------------------------------------------------------

    /// Determines the highest level of thread support that is provided by the underlying
    /// MPI implementation.
    /// \return supported threading level
    inline threading_modes threading_mode() { return detail::get_env().threading_mode(); }

    /// Determines if the current thread is the main thread, i.e., the thread that has
    /// initialized the MPI environment of the underlying MPI implementation.
    /// \return true if current thread is the main thread
    inline bool is_thread_main() { return detail::get_env().is_thread_main(); }

    /// Determines if time values given by <tt>\ref wtime</tt> are synchronized with each other
    /// for all processes of the communicator given in <tt>\ref comm_world</tt>.
    /// \return true if times are synchronized
    /// \see <tt>\ref wtime</tt>
    inline bool wtime_is_global() { return detail::get_env().wtime_is_global(); }

    /// Provides access to a predefined communicator that allows communication with
    /// all processes.
    /// \return communicator to communicate with any other process
    inline const communicator &comm_world() { return detail::get_env().comm_world(); }

    /// Provides access to a predefined communicator that includes only the calling
    /// process itself.
    /// \return communicator including only the precess itself
    inline const communicator &comm_self() { return detail::get_env().comm_self(); }

    /// Gives a unique specifier, the processor name, for the actual (physical) node.
    /// \return name of the node
    /// \note The name is determined by the underlying MPI implementation, i.e., it is
    /// implementation defined and may be different for different MPI implementations.
    inline std::string processor_name() { return detail::get_env().processor_name(); }

    /// Get time.
    /// \return number of seconds of elapsed wall-clock time since some time in the past
    inline double wtime() { return detail::get_env().wtime(); }

    /// Get resolution of time given by \c wtime.
    /// \return resolution of \c wtime in seconds.
    /// \see \c wtime
    inline double wtick() { return detail::get_env().wtick(); }

    /// Provides to MPL a buffer in the user's memory to be used for buffering outgoing
    /// messages.
    /// \param buff pointer to user-provided buffer
    /// \param size size of the buffer in bytes, must be non-negative
    /// \see \c buffer_detach
    inline void buffer_attach(void *buff, int size) {
      return detail::get_env().buffer_attach(buff, size);
    }

    /// Detach the buffer currently associated with MPL.
    /// \return pair representing the buffer location and size, i.e., the parameters provided to
    /// <tt>\ref buffer_attach</tt>
    /// \see \c buffer_attach
    inline std::pair<void *, int> buffer_detach() { return detail::get_env().buffer_detach(); }

  }  // namespace environment

  //--------------------------------------------------------------------

  tag_t tag_t::up() { return tag_t(environment::detail::get_env().tag_up()); }

  tag_t tag_t::any() { return tag_t(MPI_ANY_TAG); }

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
    explicit bsend_buffer(int size) : buff_{operator new(size)} {
      environment::buffer_attach(buff_, size);
    }

    /// waits for uncompleted message transfers and frees the buffer
    /// \note A blocking communication operation is performed when an object of type
    /// \c bsend_buffer goes out of scope.
    ~bsend_buffer() {
      environment::buffer_detach();
      operator delete(buff_);
    }

    /// deleted copy assignment operator
    /// \param other buffer manager to copy-assign from
    void operator=(const bsend_buffer &other) = delete;

    /// deleted move assignment operator
    /// \param other buffer manager to move-assign from
    void operator=(bsend_buffer &&other) = delete;
  };

}  // namespace mpl

#endif
