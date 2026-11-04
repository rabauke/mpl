#ifndef MPI_ERROR_INJECTION_HPP
#define MPI_ERROR_INJECTION_HPP

#include <mpi.h>

namespace injection {

  enum class operation {
    none,
    dup,
    set_handler,
    struct_create,
    struct_commit,
    resize,
    resized_commit,
    layout_commit,
    buffer_attach,
    error_class,
    error_string,
    error_string_bad_length,
    cancel,
    file_group,
    file_arg,
    file_io,
    datatype_create,
    info_set,
    group_size
  };

  struct counts {
    int acquired{0};
    int released{0};
    int duplicate_releases{0};
  };

  void reset();
  void track(MPI_Comm comm);
  counts events();
  counts datatype_events();
  counts buffer_events();
  int cancel_calls();

  class failure {
  public:
    explicit failure(operation op);
    ~failure();
    failure(const failure &) = delete;
    failure &operator=(const failure &) = delete;
  };

}  // namespace injection

#endif
