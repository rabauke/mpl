#include "mpi_error_injection.hpp"
#include <cstdlib>
#include <new>
#include <vector>

namespace {
  injection::operation armed{injection::operation::none};
  struct resource {
    MPI_Comm handle;
    bool released;
  };
  std::vector<resource> resources;
  int duplicate_releases{0};
  struct datatype_resource {
    MPI_Datatype handle;
    bool released;
  };
  std::vector<datatype_resource> datatypes;
  int invalid_datatype_releases{0};
  int datatype_commits{0};
  void *tracked_buffer{nullptr};
  int buffer_attaches{0};
  int buffer_deletes{0};
  int buffer_detaches{0};
  int cancellations{0};

  bool fail(injection::operation op) {
    if (armed != op)
      return false;
    armed = injection::operation::none;
    return true;
  }

  void acquired(MPI_Comm comm) {
    if (comm != MPI_COMM_NULL)
      resources.push_back({comm, false});
  }
}  // namespace

namespace injection {
  void reset() {
    armed = operation::none;
    resources.clear();
    duplicate_releases = 0;
    datatypes.clear();
    invalid_datatype_releases = 0;
    datatype_commits = 0;
    tracked_buffer = nullptr;
    buffer_attaches = 0;
    buffer_deletes = 0;
    buffer_detaches = 0;
    cancellations = 0;
  }

  void track(MPI_Comm comm) {
    acquired(comm);
  }

  counts events() {
    int released{0};
    for (const auto &resource : resources)
      released += resource.released;
    return {static_cast<int>(resources.size()), released, duplicate_releases};
  }

  counts datatype_events() {
    int released{0};
    for (const auto &resource : datatypes)
      released += resource.released;
    return {static_cast<int>(datatypes.size()), released, invalid_datatype_releases};
  }

  counts buffer_events() {
    return {buffer_attaches, buffer_deletes, buffer_detaches};
  }

  int cancel_calls() {
    return cancellations;
  }

  failure::failure(operation op) {
    armed = op;
  }
  failure::~failure() {
    armed = operation::none;
  }
}  // namespace injection

extern "C" int MPI_Error_class(int code, int *error_class) {
  if (fail(injection::operation::error_class))
    return MPI_ERR_OTHER;
  return PMPI_Error_class(code, error_class);
}

extern "C" int MPI_Error_string(int code, char *message, int *length) {
  if (fail(injection::operation::error_string))
    return MPI_ERR_OTHER;
  if (fail(injection::operation::error_string_bad_length)) {
    *length = MPI_MAX_ERROR_STRING + 1;
    return MPI_SUCCESS;
  }
  return PMPI_Error_string(code, message, length);
}

extern "C" int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  if (fail(injection::operation::dup))
    return MPI_ERR_OTHER;
  const int result{PMPI_Comm_dup(comm, newcomm)};
  if (result == MPI_SUCCESS)
    acquired(*newcomm);
  return result;
}

extern "C" int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
  const int result{PMPI_Comm_split(comm, color, key, newcomm)};
  if (result == MPI_SUCCESS)
    acquired(*newcomm);
  return result;
}

extern "C" int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler handler) {
  if (fail(injection::operation::set_handler))
    return MPI_ERR_OTHER;
  return PMPI_Comm_set_errhandler(comm, handler);
}

extern "C" int MPI_Comm_free(MPI_Comm *comm) {
  for (auto it{resources.rbegin()}; it != resources.rend(); ++it) {
    if (it->handle == *comm) {
      if (it->released)
        ++duplicate_releases;
      else
        it->released = true;
      break;
    }
  }
  return PMPI_Comm_free(comm);
}

extern "C" int MPI_Type_create_struct_c(MPI_Count count, const MPI_Count blocklengths[],
                                        const MPI_Count displacements[],
                                        const MPI_Datatype types[], MPI_Datatype *newtype) {
  if (fail(injection::operation::struct_create)) {
    *newtype = MPI_INT;
    return MPI_ERR_OTHER;
  }
  const int result{
      PMPI_Type_create_struct_c(count, blocklengths, displacements, types, newtype)};
  if (result == MPI_SUCCESS)
    datatypes.push_back({*newtype, false});
  return result;
}

extern "C" int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                                       MPI_Datatype *newtype) {
  if (fail(injection::operation::resize)) {
    *newtype = MPI_INT;
    return MPI_ERR_OTHER;
  }
  const int result{PMPI_Type_create_resized(oldtype, lb, extent, newtype)};
  if (result == MPI_SUCCESS)
    datatypes.push_back({*newtype, false});
  return result;
}

extern "C" int MPI_Type_commit(MPI_Datatype *datatype) {
  ++datatype_commits;
  if (fail(injection::operation::layout_commit) ||
      fail(datatype_commits == 1 ? injection::operation::struct_commit
                                 : injection::operation::resized_commit))
    return MPI_ERR_OTHER;
  return PMPI_Type_commit(datatype);
}

extern "C" int MPI_Type_contiguous_c(MPI_Count count, MPI_Datatype oldtype,
                                     MPI_Datatype *newtype) {
  if (fail(injection::operation::datatype_create))
    return MPI_ERR_TYPE;
  const int result{PMPI_Type_contiguous_c(count, oldtype, newtype)};
  if (result == MPI_SUCCESS)
    datatypes.push_back({*newtype, false});
  return result;
}

extern "C" int MPI_Cancel(MPI_Request *request) {
  ++cancellations;
  if (fail(injection::operation::cancel))
    return MPI_ERR_REQUEST;
  return PMPI_Cancel(request);
}

extern "C" int MPI_File_get_group(MPI_File file, MPI_Group *group) {
  if (fail(injection::operation::file_group))
    return MPI_ERR_FILE;
  return PMPI_File_get_group(file, group);
}

extern "C" int MPI_File_get_size(MPI_File file, MPI_Offset *size) {
  if (fail(injection::operation::file_arg))
    return MPI_ERR_ARG;
  if (fail(injection::operation::file_io))
    return MPI_ERR_FILE;
  return PMPI_File_get_size(file, size);
}

extern "C" int MPI_Info_set(MPI_Info info, const char *key, const char *value) {
  if (fail(injection::operation::info_set))
    return MPI_ERR_INFO;
  return PMPI_Info_set(info, key, value);
}

extern "C" int MPI_Group_size(MPI_Group group, int *size) {
  if (fail(injection::operation::group_size))
    return MPI_ERR_GROUP;
  return PMPI_Group_size(group, size);
}

extern "C" int MPI_Buffer_attach_c(void *buffer, MPI_Count size) {
  tracked_buffer = buffer;
  ++buffer_attaches;
  if (fail(injection::operation::buffer_attach))
    return MPI_ERR_OTHER;
  return PMPI_Buffer_attach_c(buffer, size);
}

extern "C" int MPI_Buffer_detach_c(void *buffer, MPI_Count *size) {
  ++buffer_detaches;
  return PMPI_Buffer_detach_c(buffer, size);
}

void *operator new(std::size_t size) {
  if (void *p = std::malloc(size == 0 ? 1 : size))
    return p;
  throw std::bad_alloc{};
}

void operator delete(void *p) noexcept {
  if (p != nullptr && p == tracked_buffer)
    ++buffer_deletes;
  std::free(p);
}

void operator delete(void *p, std::size_t) noexcept {
  ::operator delete(p);
}

extern "C" int MPI_Type_free(MPI_Datatype *datatype) {
  for (auto it{datatypes.rbegin()}; it != datatypes.rend(); ++it) {
    if (it->handle == *datatype) {
      if (it->released) {
        ++invalid_datatype_releases;
        return MPI_ERR_TYPE;
      }
      it->released = true;
      return PMPI_Type_free(datatype);
    }
  }
  if (*datatype == MPI_DATATYPE_NULL || *datatype == MPI_INT) {
    ++invalid_datatype_releases;
    return MPI_ERR_TYPE;
  }
  return PMPI_Type_free(datatype);
}