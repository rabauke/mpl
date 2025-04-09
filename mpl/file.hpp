#if !(defined MPL_FILE_HPP)

#define MPL_FILE_HPP

#include <mpl/utility.hpp>
#include <mpl/request.hpp>
#include <mpl/info.hpp>
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
      create = MPI_MODE_CREATE,      ///< create file if it does not exist
      no_replace = MPI_MODE_EXCL,    ///< raises an error when file to create already exists
      delete_on_close = MPI_MODE_DELETE_ON_CLOSE,  ///< delete file when closed
      unique_open = MPI_MODE_UNIQUE_OPEN,          ///< file not opened concurrently
      sequential = MPI_MODE_SEQUENTIAL,            ///< file will be accessed sequentially
      append = MPI_MODE_APPEND                     ///< set initial file position to end of file
    };

    /// %file pointer positioning mode
    enum class whence_mode : int {
      set =
          MPI_SEEK_SET,  ///< pointer positioning relative to the file's beginning (absolute pointer positioning)
      current = MPI_SEEK_CUR,  ///< pointer positioning relative to current position
      end = MPI_SEEK_END       ///< pointer positioning relative to the file's end
    };

    /// default constructor
    file() = default;

    /// constructs and opens a %file
    /// \param comm communicator
    /// \param name %file name
    /// \param mode %file open-mode
    /// \param i hints
    explicit file(const communicator &comm, const char *name, access_mode mode,
                  const info &i = info{}) {
      open(comm, name, mode, i);
    }

    /// constructs and opens a %file
    /// \param comm communicator
    /// \param name %file name
    /// \param mode %file open-mode
    /// \param i hints
    explicit file(const communicator &comm, const std::string &name, access_mode mode,
                  const info &i = info{}) {
      open(comm, name, mode, i);
    }

    /// constructs and opens a %file
    /// \param comm communicator
    /// \param name %file name
    /// \param mode %file open-mode
    /// \param i hints
    explicit file(const communicator &comm, const std::filesystem::path &name, access_mode mode,
                  const info &i = info{}) {
      open(comm, name, mode, i);
    }

    /// deleted copy constructor
    file(const file &) = delete;

    /// move constructor
    /// \param other file to move from
    file(file &&other) noexcept : file_{other.file_} {
      other.file_ = MPI_FILE_NULL;
    }

    /// destructor
    ~file() {
      try {
        close();
      } catch (io_failure &) {
        // must not throw
      }
    }

    /// deleted copy-assignment operator
    file &operator=(const file &) = delete;

    /// move-assignment operator
    /// \param other %file to move from
    file &operator=(file &&other) noexcept {
      try {
        close();
      } catch (io_failure &) {
        // must not throw
      }
      file_ = other.file_;
      other.file_ = MPI_FILE_NULL;
      return *this;
    }

    /// open a %file
    /// \param comm communicator
    /// \param name %file name
    /// \param mode %file open-mode
    /// \param i hints
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const char *name, access_mode mode,
              const info &i = info{}) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{
          MPI_File_open(comm.comm_, name, static_cast<int_type>(mode), i.info_, &file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    /// \param i hints
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const std::string &name, access_mode mode,
              const info &i = info{}) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode),
                                  i.info_, &file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    /// \param i hints
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void open(const communicator &comm, const std::filesystem::path &name, access_mode mode,
              const info &i = info{}) {
      using int_type = std::underlying_type_t<file::access_mode>;
      const int err{MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode),
                                  i.info_, &file_)};
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
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void resize(ssize_t size) {
      const int err{MPI_File_set_size(file_, size)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// resize file (grow as required)
    /// \param size file size in bytes
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
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
    void sync() {
      const int err{MPI_File_sync(file_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// set the process's file view
    /// \tparam T elementary read/write data type
    /// \param representation data representation, e.g., "native", "internal" or "external32"
    /// \param displacement beginning of the view in bytes from the beginning of the file
    /// \param i hints
    /// \return status of performed i/o operation
    template<typename T>
    void set_view(const char *representation, ssize_t displacement = 0,
                  const info &i = info{}) {
      const int err{MPI_File_set_view(
          file_, displacement, detail::datatype_traits<T>::get_datatype(),
          detail::datatype_traits<T>::get_datatype(), representation, i.info_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// set the process's file view
    /// \tparam T elementary read/write data type
    /// \param representation data representation, e.g., "native", "internal" or "external32"
    /// \param l layout defining the file view
    /// \param displacement beginning of the view in bytes from the beginning of the file
    /// \param i hints
    /// \return status of performed i/o operation
    template<typename T>
    void set_view(const char *representation, const layout<T> &l, ssize_t displacement = 0,
                  const info &i = info{}) {
      const int err{MPI_File_set_view(
          file_, displacement, detail::datatype_traits<T>::get_datatype(),
          detail::datatype_traits<layout<T>>::get_datatype(l), representation, i.info_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// update current individual %file pointer
    /// \param offset %file pointer offset
    /// \param whence %file pointer positioning mode
    void seek(ssize_t offset, whence_mode whence) {
      const int err{MPI_File_seek(file_, offset, static_cast<int>(whence))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// get current individual %file pointer
    /// \return current individual %file pointer
    [[nodiscard]] ssize_t position() const {
      MPI_Offset offset{0};
      const int err{MPI_File_get_position(file_, &offset)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return offset;
    }

    /// get absolute byte position in %file
    /// \param offset %file pointer offset
    /// \return absolute byte position in %file that corresponds to the given view-relative
    /// offset
    [[nodiscard]] ssize_t byte_offset(ssize_t offset) const {
      MPI_Offset displ{0};
      const int err{MPI_File_get_byte_offset(file_, offset, &displ)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return displ;
    }

    /// set %file hint
    /// \param i hint
    /// \note This is a collective operation and must be called by all processes in the
    /// communicator.
    void set_info(info &i) {
      const int err{MPI_File_set_info(file_, i.info_)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// get %file hint
    /// \return %file hint
    [[nodiscard]] info get_info() const {
      MPI_Info i;
      const int err{MPI_File_get_info(file_, &i)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return info(i);
    }

    /// Get the underlying MPI handle of the file.
    /// \return MPI handle of the file
    /// \note This function returns a non-owning handle to the underlying MPI file, which may
    /// be useful when refactoring legacy MPI applications to MPL.
    /// \warning This method will be removed in a future version.
    [[nodiscard]] MPI_File native_handle() const {
      return file_;
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
      const int err{MPI_File_iread_at(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
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
      const int err{MPI_File_iwrite_at(file_, offset, &data, 1,
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
      const int err{MPI_File_iwrite_at(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
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

    /// read data from file, non-blocking, non-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread(T &data) {
      MPI_Request req;
      const int err{
          MPI_File_iread(file_, &data, 1, detail::datatype_traits<T>::get_datatype(), &req)};
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
      const int err{MPI_File_iread(file_, data, 1,
                                   detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
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
          MPI_File_iwrite(file_, &data, 1, detail::datatype_traits<T>::get_datatype(), &req)};
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
      const int err{MPI_File_iwrite(file_, data, 1,
                                    detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
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

    /// write data to file, blocking, non-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_shared(const T &data) {
      status_t s;
      const int err{MPI_File_write_shared(file_, &data, 1,
                                          detail::datatype_traits<T>::get_datatype(),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, non-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_shared(const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write_shared(file_, data, 1,
                                          detail::datatype_traits<layout<T>>::get_datatype(l),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, non-blocking, non-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_shared(T &data) {
      MPI_Request req;
      const int err{MPI_File_iread_shared(file_, &data, 1,
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
      const int err{MPI_File_iread_shared(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
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
      const int err{MPI_File_iwrite_shared(file_, &data, 1,
                                           detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, non-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_shared(const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_iwrite_shared(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, blocking, collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at_all(ssize_t offset, T &data) {
      status_t s;
      const int err{MPI_File_read_at_all(file_, offset, &data, 1,
                                         detail::datatype_traits<T>::get_datatype(),
                                         static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at_all(ssize_t offset, T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read_at_all(file_, offset, data, 1,
                                         detail::datatype_traits<layout<T>>::get_datatype(l),
                                         static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at_all(ssize_t offset, const T &data) {
      status_t s;
      const int err{MPI_File_write_at_all(file_, offset, &data, 1,
                                          detail::datatype_traits<T>::get_datatype(),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at_all(ssize_t offset, const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write_at_all(file_, offset, data, 1,
                                          detail::datatype_traits<layout<T>>::get_datatype(l),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, non-blocking, collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_at_all(ssize_t offset, T &data) {
      MPI_Request req;
      const int err{MPI_File_iread_at_all(file_, offset, &data, 1,
                                          detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_at_all(ssize_t offset, T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_iread_at_all(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data value to write
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_at_all(ssize_t offset, const T &data) {
      MPI_Request req;
      const int err{MPI_File_iwrite_at_all(file_, offset, &data, 1,
                                           detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_at_all(ssize_t offset, const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_iwrite_at_all(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, blocking, collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_all(T &data) {
      status_t s;
      const int err{MPI_File_read_all(file_, &data, 1,
                                      detail::datatype_traits<T>::get_datatype(),
                                      static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_all(T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read_all(file_, data, 1,
                                      detail::datatype_traits<layout<T>>::get_datatype(l),
                                      static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_all(const T &data) {
      status_t s;
      const int err{MPI_File_write_all(file_, &data, 1,
                                       detail::datatype_traits<T>::get_datatype(),
                                       static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_all(const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write_all(file_, data, 1,
                                       detail::datatype_traits<layout<T>>::get_datatype(l),
                                       static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, non-blocking, collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_all(T &data) {
      MPI_Request req;
      const int err{MPI_File_iread_all(file_, &data, 1,
                                       detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, non-blocking, collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iread_all(T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_iread_all(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_all(const T &data) {
      MPI_Request req;
      const int err{MPI_File_iwrite_all(file_, &data, 1,
                                        detail::datatype_traits<T>::get_datatype(), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// write data to file, non-blocking, collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return request representing the ongoing i/o operation
    template<typename T>
    irequest iwrite_all(const T *data, const layout<T> &l) {
      MPI_Request req;
      const int err{MPI_File_iwrite_all(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &req)};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return impl::base_irequest{req};
    }

    /// read data from file, blocking, collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_ordered(T &data) {
      status_t s;
      const int err{MPI_File_read_ordered(file_, &data, 1,
                                          detail::datatype_traits<T>::get_datatype(),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_ordered(T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_read_ordered(file_, data, 1,
                                          detail::datatype_traits<layout<T>>::get_datatype(l),
                                          static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_ordered(const T &data) {
      status_t s;
      const int err{MPI_File_write_ordered(file_, &data, 1,
                                           detail::datatype_traits<T>::get_datatype(),
                                           static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_ordered(const T *data, const layout<T> &l) {
      status_t s;
      const int err{MPI_File_write_ordered(file_, data, 1,
                                           detail::datatype_traits<layout<T>>::get_datatype(l),
                                           static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, split-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    template<typename T>
    void read_at_all_begin(ssize_t offset, T &data) {
      const int err{MPI_File_read_at_all_begin(file_, offset, &data, 1,
                                               detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// read data from file, blocking, split-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    template<typename T>
    void read_at_all_begin(ssize_t offset, T *data, const layout<T> &l) {
      const int err{MPI_File_read_at_all_begin(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish reading data from file, blocking, split-collective, explicit offset
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at_all_end(T &data) {
      status_t s;
      const int err{MPI_File_read_at_all_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish reading data from file, blocking, split-collective, explicit offset
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at_all_end(T *data) {
      status_t s;
      const int err{MPI_File_read_at_all_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, split-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data value to write
    template<typename T>
    void write_at_all_begin(ssize_t offset, const T &data) {
      const int err{MPI_File_write_at_all_begin(file_, offset, &data, 1,
                                                detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// write data to file, blocking, split-collective, explicit offset
    /// \tparam T write data type
    /// \param offset file offset in bytes
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    template<typename T>
    void write_at_all_begin(ssize_t offset, const T *data, const layout<T> &l) {
      const int err{MPI_File_write_at_all_begin(
          file_, offset, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish writing data to file, blocking, split-collective, explicit offset
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at_all_end(const T &data) {
      status_t s;
      const int err{MPI_File_write_at_all_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish writing data to file, blocking, split-collective, explicit offset
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_at_all_end(const T *data) {
      status_t s;
      const int err{MPI_File_write_at_all_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, split-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    template<typename T>
    void read_all_begin(T &data) {
      const int err{
          MPI_File_read_all_begin(file_, &data, 1, detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// read data from file, blocking, split-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    template<typename T>
    void read_all_begin(T *data, const layout<T> &l) {
      const int err{MPI_File_read_all_begin(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish reading data from file, blocking, split-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_all_end(T &data) {
      status_t s;
      const int err{MPI_File_read_all_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish reading data from file, blocking, split-collective, individual file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_all_end(T *data) {
      status_t s;
      const int err{MPI_File_read_all_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, split-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    template<typename T>
    void write_all_begin(const T &data) {
      const int err{MPI_File_write_all_begin(file_, &data, 1,
                                             detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// write data to file, blocking, split-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    template<typename T>
    void write_all_begin(const T *data, const layout<T> &l) {
      const int err{MPI_File_write_all_begin(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish writing data to file, blocking, split-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_all_end(const T &data) {
      status_t s;
      const int err{MPI_File_write_all_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish writing data to file, blocking, split-collective, individual file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_all_end(const T *data) {
      status_t s;
      const int err{MPI_File_write_all_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// read data from file, blocking, split-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    template<typename T>
    void read_ordered_begin(T &data) {
      const int err{MPI_File_read_ordered_begin(file_, &data, 1,
                                                detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// read data from file, blocking, split-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    template<typename T>
    void read_ordered_begin(T *data, const layout<T> &l) {
      const int err{MPI_File_read_ordered_begin(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish reading data from file, blocking, split-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_ordered_end(T &data) {
      status_t s;
      const int err{MPI_File_read_ordered_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish reading data from file, blocking, split-collective, shared file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_ordered_end(T *data) {
      status_t s;
      const int err{MPI_File_read_ordered_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// write data to file, blocking, split-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    template<typename T>
    void write_ordered_begin(const T &data) {
      const int err{MPI_File_write_ordered_begin(file_, &data, 1,
                                                 detail::datatype_traits<T>::get_datatype())};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// write data to file, blocking, split-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    template<typename T>
    void write_ordered_begin(const T *data, const layout<T> &l) {
      const int err{MPI_File_write_ordered_begin(
          file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
    }

    /// finish writing data to file, blocking, split-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_ordered_end(const T &data) {
      status_t s;
      const int err{MPI_File_write_ordered_end(file_, &data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
    }

    /// finish writing data to file, blocking, split-collective, shared file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write_ordered_end(const T *data) {
      status_t s;
      const int err{MPI_File_write_ordered_end(file_, data, static_cast<MPI_Status *>(&s))};
      if (err != MPI_SUCCESS)
        throw io_failure(err);
      return s;
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
