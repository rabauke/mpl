#if !(defined MPL_ENVIRONMENT_HPP)

#define MPL_ENVIRONMENT_HPP

#include <string>
#include <memory>
#include <vector>
#include <mpi.h>

namespace mpl {

  /// \brief Represents the various levels of thread support that the underlying MPI
  /// implementation may provide.
  enum class threading_modes {
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
  };

  namespace environment {

    namespace detail {

      class env {
        class initializer {
          int thread_mode_;

        public:
          initializer() {
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &thread_mode_);
          }

          ~initializer() { MPI_Finalize(); }

          threading_modes thread_mode() const {
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
        env() : init(), comm_world_(MPI_COMM_WORLD), comm_self_(MPI_COMM_SELF) {
          int size;
          MPI_Comm_size(MPI_COMM_WORLD, &size);
        }

        env(const env &) = delete;

        env &operator=(const env &) = delete;

        int tag_up() const {
          void *p;
          int flag;
          MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &p, &flag);
          return *reinterpret_cast<int *>(p);
        }

        threading_modes threading_mode() const { return init.thread_mode(); }

        bool is_thread_main() const {
          int res;
          MPI_Is_thread_main(&res);
          return static_cast<bool>(res);
        }

        bool wtime_is_global() const {
          void *p;
          int flag;
          MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &p, &flag);
          return *reinterpret_cast<int *>(p);
        }

        const communicator &comm_world() const { return comm_world_; }

        const communicator &comm_self() const { return comm_self_; }

        std::string processor_name() const {
          char name[MPI_MAX_PROCESSOR_NAME];
          int len;
          MPI_Get_processor_name(name, &len);
          return std::string(name);
        }

        double wtime() const { return MPI_Wtime(); }

        double wtick() const { return MPI_Wtick(); }

        void buffer_attach(void *buff, int size) const { MPI_Buffer_attach(buff, size); }

        std::pair<void *, int> buffer_detach() const {
          void *buff;
          int size;
          MPI_Buffer_detach(&buff, &size);
          return std::make_pair(buff, size);
        }
      };

      //----------------------------------------------------------------

      inline const env &get_env() {
        static env the_env;
        return the_env;
      }

    }  // namespace detail

    //------------------------------------------------------------------

    /// \brief Determines the highest level of thread support that is provided by the underlying
    /// MPI implementation.
    /// \return supported threading level
    inline threading_modes threading_mode() { return detail::get_env().threading_mode(); }

    /// \brief Determines if the current thread is the main thread, i.e., the thread that has
    /// initialized the MPI environment of the underlying MPI implementation.
    /// \return true if current thread is the main thread
    inline bool is_thread_main() { return detail::get_env().is_thread_main(); }

    /// \brief Determines if time values given by \ref wtime are synchronized with each other
    /// for all processes of the communicator given in \ref comm_world.
    /// \return true if times are  synchronized
    /// \see \ref wtime
    inline bool wtime_is_global() { return detail::get_env().wtime_is_global(); }

    /// \brief Provides access to a predefined communicator that allows communication with
    /// all processes.
    /// \return communicator to communicate with any other process
    inline const communicator &comm_world() { return detail::get_env().comm_world(); }

    /// \brief Provides access to a predefined communicator that includes only the calling
    /// process itself.
    /// \return communicator including only the precess itself
    inline const communicator &comm_self() { return detail::get_env().comm_self(); }

    /// \brief Gives a unique specifier, the processor name, for the actual (physical) node.
    /// \return name of the node
    /// \note The name is determined by the underlying MPI implementation, i.e., it is
    /// implementation defined and may be different for different MPI implementations.
    inline std::string processor_name() { return detail::get_env().processor_name(); }

    /// \brief Get time.
    /// \return number of seconds of elapsed wall-clock time since some time in the past
    inline double wtime() { return detail::get_env().wtime(); }

    /// \brief Get resolution of time given by \ref wtime.
    /// \return resolution of \ref wtime in seconds.
    /// \see \ref wtime
    inline double wtick() { return detail::get_env().wtick(); }

    /// \brief Provides to MPL a buffer in the user's memory to be used for buffering outgoing
    /// messages.
    /// \param buff pointer to user-provided buffer
    /// \param size size of the buffer in bytes, must be non-negative
    /// \see \ref buffer_detach
    inline void buffer_attach(void *buff, int size) {
      return detail::get_env().buffer_attach(buff, size);
    }

    /// \brief Detach the buffer currently associated with MPL.
    /// \return pair representing the buffer location and size, i.e., the parameters provided to
    /// \ref buffer_attach
    /// \see \ref buffer_attach
    inline std::pair<void *, int> buffer_detach() { return detail::get_env().buffer_detach(); }

  }  // namespace environment

  //--------------------------------------------------------------------

  tag tag::up() { return tag(environment::detail::get_env().tag_up()); }

  tag tag::any() { return tag(MPI_ANY_TAG); }

  //--------------------------------------------------------------------

  /// \brief Buffer manager for buffered  send operations.
  /// \param A allocator for allocating buffer memory
  template<typename A = std::allocator<char>>
  class bsend_buffer {
    int size;
    A alloc;
    char *buff;

  public:
    /// allocates buffer with specific size using a default-constructed allocator
    /// \param size buffer size in bytes
    /// \note The size given should be the sum of the sizes of all outstanding buffered send
    /// operations will be sent during the lifetime of the \ref bsend_buffer object, plus
    /// \ref bsend_overhead for each buffered send operation.
    /// \see communicator_bsend
    explicit bsend_buffer(int size) : size(size), alloc(), buff(alloc.allocate(size)) {
      environment::buffer_attach(buff, size);
    }

    /// allocates buffer with specific size using the provided allocator
    /// \param size buffer size in bytes
    /// \param alloc allocator
    /// \note The size given should be the sum of the sizes of all outstanding buffered send
    /// operations will be sent during the lifetime of the \ref bsend_buffer object, plus
    /// \ref bsend_overhead for each buffered send operation.
    /// \see communicator_bsend
    explicit bsend_buffer(int size, A alloc)
        : size(size), alloc(alloc), buff(alloc.allocate(size)) {
      environment::buffer_attach(buff, size);
    }

    /// frees the buffer
    ~bsend_buffer() {
      environment::buffer_detach();
      alloc.deallocate(buff, size);
    }
  };

}  // namespace mpl

#endif
