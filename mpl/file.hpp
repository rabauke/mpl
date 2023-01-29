#if !(defined MPL_FILE_HPP)

#define MPL_FILE_HPP

#include <mpl/utility.hpp>
#include <mpl/request.hpp>
#include <string>
#include <filesystem>
#include <type_traits>

namespace mpl {

  /// Class implementing parallel file i/o.
  class file {
  private:
    MPI_File file_{MPI_FILE_NULL};

  public:
    /// %file access mode
    enum class access_mode : int {
      read_only = MPI_MODE_RDONLY,   ///< read-only file access
      read_write = MPI_MODE_RDWR,    ///< read and write file access
      write_only = MPI_MODE_WRONLY,  ///< write-only file access
      create = MPI_MODE_CREATE,      ///< create file it it does not exist
      no_replace = MPI_MODE_EXCL,    ///< raises an error when file to create already exists
      delete_on_close = MPI_MODE_DELETE_ON_CLOSE,  ///< delete file when closed
      unique_open = MPI_MODE_UNIQUE_OPEN,          ///< file not opened concurrently
      sequential = MPI_MODE_SEQUENTIAL,            ///< file will be accessed sequentially
      append = MPI_MODE_APPEND                     ///< set initial file position to end of file
    };

    /// %file pointer positioning mode
    enum class whence_mode : int {
      set = MPI_SEEK_SET,      ///< pointer positioning relative to the file's beginning (absolute pointer positioning)
      current = MPI_SEEK_CUR,  ///< pointer positioning relative to current position
      end = MPI_SEEK_END       ///< pointer positioning relative to the file's end
    };

    /// default constructor
    file() = default;

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const char *name, access_mode mode) {
      open(comm, name, mode);
    }

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const std::string &name, access_mode mode) {
      open(comm, name, mode);
    }

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const std::filesystem::path &name,
                  access_mode mode) {
      open(comm, name, mode);
    }

    /// deleted copy constructor
    file(const file &) = delete;

    /// move constructor
    /// \param other file to move from
    file(file &&other) : file_{other.file_} {
      other.file_ = MPI_FILE_NULL;
    }

    /// destructor
    ~file() {
      try {
        close();
      } catch (io_failure &) {
      }
    }

    /// deleted copy-assignment operator
    file &operator=(const file &) = delete;

    /// move-assignment operator
    /// \param other file to move from
    file &operator=(file &&other) {
      close();
      file_ = other.file_;
      other.file_ = MPI_FILE_NULL;
      return *this;
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const char *name, access_mode mode) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{
          MPI_File_open(comm.comm_, name, static_cast<int_type>(mode), MPI_INFO_NULL, &file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const std::string &name, access_mode mode) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode),
                                  MPI_INFO_NULL, &file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const std::filesystem::path &name, access_mode mode) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode),
                                  MPI_INFO_NULL, &file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// close a file
    void close() {
      const int err{MPI_File_close(&file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// resize file (shrink or grow as required)
    /// \param size file size in bytes
    void resize(ssize_t size) {
      const int err{MPI_File_set_size(file_, size)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// resize file (grow as required)
    /// \param size file size in bytes
    void preallocate(ssize_t size) {
      const int err{MPI_File_preallocate(file_, size)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// get file size
    /// \return file size in bytes
    [[nodiscard]] ssize_t size() const {
      MPI_Offset size{0};
      const int err{MPI_File_get_size(file_, &size)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return size;
    }

    /// get file open-mode
    /// \return file open-mode
    [[nodiscard]] access_mode mode() const {
      int mode{0};
      const int err{MPI_File_get_amode(file_, &mode)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return static_cast<access_mode>(mode);
    }

    /// flush write buffers and write pending data to device
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void sync() const {
      const int err{MPI_File_sync(file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// set the process's file view
    /// \tparam T elementary read/write data type
    /// \param displacement beginning of the view in bytes from the beginning of the file
    /// \param l layout used in associated i/o operation
    /// \param representation data representation, e.g., "native", "internal" or "external32"
    /// \return status of performed i/o operation
    template<typename T>
    void set_view(ssize_t displacement, const layout<T> &l, const char *representation) {
      const int err{MPI_File_set_view(
          file_, displacement, detail::datatype_traits<T>::get_datatype(),
          detail::datatype_traits<layout<T>>::get_datatype(l), representation, MPI_INFO_NULL)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// update current individual %file pointer
    /// \param offset %file pointer offset
    /// \param whence %file pointer positioning mode
    void seek(ssize_t offset, whence_mode whence) {
      const int err{
        MPI_File_seek(file_, offset, static_cast<int>(whence))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// get current individual %file pointer
    /// \return current individual %file pointer
    ssize_t position() {
      MPI_Offset offset{0};
      const int err{
        MPI_File_get_position(file_, &offset)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return offset;
    }

    /// get absolute byte position in %file
    /// \param offset %file pointer offset
    /// \return absolute byte position in %file that corresponds to the given view-relative
    /// offset
    ssize_t byte_offset(ssize_t offset) {
      MPI_Offset displ{0};
      const int err{
        MPI_File_get_byte_offset(file_, offset, &displ)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return displ;
    }

    /// read data from file, blocking, non-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at(ssize_t offset, T &data) {
      status_t s;
      const int err{MPI_File_read_at(file_, offset, &data, 1,
                                     detail::datatype_traits<T>::get_datatype(),
                                     static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, non-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at(ssize_t offset, T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read_at(file_, offset, data, 1,
                                     detail::datatype_traits<layout<T>>::get_datatype(l),
                                     static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, non-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read(T &data) {
      status_t s;
      const int err{MPI_File_read(file_, &data, 1, detail::datatype_traits<T>::get_datatype(),
                                  static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, non-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read(T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read(file_, data, 1,
                                  detail::datatype_traits<layout<T>>::get_datatype(l),
                                  static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, non-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_shared(T &data) {
      status_t s;
      const int err{MPI_File_read_shared(file_, &data, 1,
                                         detail::datatype_traits<T>::get_datatype(),
                                         static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, non-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_shared(T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read_shared(file_, data, 1,
                                         detail::datatype_traits<layout<T>>::get_datatype(l),
                                         static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at(ssize_t offset, const T &data) {
      status_t s;
      const int err{MPI_File_write_at(file_, offset, &data, 1,
                                      detail::datatype_traits<T>::get_datatype(),
                                      static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at(ssize_t offset, const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write_at(file_, offset, data, 1,
                                      detail::datatype_traits<layout<T>>::get_datatype(l),
                                      static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write(const T &data) {
      status_t s;
      const int err{MPI_File_write(file_, &data, 1, detail::datatype_traits<T>::get_datatype(),
                                   static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write(const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write(file_, data, 1,
                                   detail::datatype_traits<layout<T>>::get_datatype(l),
                                   static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_shared(const T &data) {
      status_t s;
      const int err{MPI_File_write(file_, &data, 1, detail::datatype_traits<T>::get_datatype(),
                                   static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, shred file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_shared(const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write(file_, data, 1,
                                   detail::datatype_traits<layout<T>>::get_datatype(l),
                                   static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, non-blocking, non-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_at(ssize_t offset, T &data) {
      MPI_Request req;
      const int err{MPI_File_iread_at(file_, offset, &data, 1,
                                      detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, non-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_at(ssize_t offset, T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_read_at(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, non-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread(T &data) {
      MPI_Request req;
      const int err{
          MPI_File_read(file_, &data, 1, detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, non-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread(T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_read(file_, data, 1,
                                  detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, non-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_shared(T &data) {
      MPI_Request req;
      const int err{MPI_File_read_shared(file_, &data, 1,
                                         detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, non-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_shared(T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_read_shared(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data value to write
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_at(ssize_t offset, const T &data) {
      MPI_Request req;
      const int err{MPI_File_write_at(file_, offset, &data, 1,
                                      detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_at(ssize_t offset, const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_write_at(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite(const T &data) {
      MPI_Request req;
      const int err{
          MPI_File_write(file_, &data, 1, detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite(const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_write(file_, data, 1,
                                   detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_shared(const T &data) {
      MPI_Request req;
      const int err{
          MPI_File_write(file_, &data, 1, detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, shred file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_shared(const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_write(file_, data, 1,
                                   detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    friend class group;
  };


  /// bit-wise disjunction operator for %file access modes
  /// \param mode1 1st access mode
  /// \param mode2 2nd access mode
  /// \return combined %file access mode
  inline file::access_mode operator|(file::access_mode mode1, file::access_mode mode2) {
    using int_type = std::underlying_type_t<file::access_mode>;
    return static_cast<file::access_mode>(static_cast<int_type>(mode1) |
                                          static_cast<int_type>(mode2));
  }


  /// bit-wise disjunction assignment operator for %file access modes
  /// \param mode1 1st access mode
  /// \param mode2 2nd access mode
  /// \return combined %file access mode
  inline file::access_mode &operator|=(file::access_mode &mode1, file::access_mode mode2) {
    using int_type = std::underlying_type_t<file::access_mode>;
    mode1 = static_cast<file::access_mode>(static_cast<int_type>(mode1) |
                                           static_cast<int_type>(mode2));
    return mode1;
  }


  /// bit-wise conjunction operator for %file access modes
  /// \param mode1 1st access mode
  /// \param mode2 2nd access mode
  /// \return combined %file access mode
  inline file::access_mode operator&(file::access_mode mode1, file::access_mode mode2) {
    using int_type = std::underlying_type_t<file::access_mode>;
    return static_cast<file::access_mode>(static_cast<int_type>(mode1) &
                                          static_cast<int_type>(mode2));
  }


  /// bit-wise conjunction assignment operator for %file access modes
  /// \param mode1 1st access mode
  /// \param mode2 2nd access mode
  /// \return combined %file access mode
  inline file::access_mode &operator&=(file::access_mode &mode1, file::access_mode mode2) {
    using int_type = std::underlying_type_t<file::access_mode>;
    mode1 = static_cast<file::access_mode>(static_cast<int_type>(mode1) &
                                           static_cast<int_type>(mode2));
    return mode1;
  }


  inline group::group(const file &f) {
    MPI_File_get_group(f.file_, &gr_);
  }

}  // namespace mpl

#endif
