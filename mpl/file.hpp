#if !(defined MPL_FILE_HPP)

#define MPL_FILE_HPP

#include <mpl/utility.hpp>
#include <string>
#include <filesystem>
#include <type_traits>

namespace mpl {

  /// ToDo: add error handling

  /// Class implementing parallel file i/o.
  class file {
  private:
    MPI_File file_{MPI_FILE_NULL};

  public:
    enum class openmode : int {
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

    /// default constructor
    file() = default;

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const char *name, openmode mode) {
      open(comm, name, mode);
    }

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const std::string &name, openmode mode) {
      open(comm, name, mode);
    }

    /// constructs and opens a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    explicit file(const communicator &comm, const std::filesystem::path &name, openmode mode) {
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
      close();
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
    void open(const communicator &comm, const char *name, openmode mode) {
      using int_type = std::underlying_type_t<file::openmode>;
      MPI_File_open(comm.comm_, name, static_cast<int_type>(mode), MPI_INFO_NULL, &file_);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    void open(const communicator &comm, const std::string &name, openmode mode) {
      using int_type = std::underlying_type_t<file::openmode>;
      MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode), MPI_INFO_NULL,
                    &file_);
    }

    /// open a file
    /// \param comm communicator
    /// \param name file name
    /// \param mode file open-mode
    void open(const communicator &comm, const std::filesystem::path &name, openmode mode) {
      using int_type = std::underlying_type_t<file::openmode>;
      MPI_File_open(comm.comm_, name.c_str(), static_cast<int_type>(mode), MPI_INFO_NULL,
                    &file_);
    }

    /// close a file
    void close() {
      MPI_File_close(&file_);
    }

    /// resize file (shrink or grow as required)
    /// \param size file size in bytes
    void set_size(ssize_t size) {
      MPI_File_set_size(file_, size);
    }

    /// resize file (grow as required)
    /// \param size file size in bytes
    void preallocate(ssize_t size) {
      MPI_File_preallocate(file_, size);
    }

    /// get file size
    /// \return file size in bytes
    [[nodiscard]] ssize_t size() const {
      MPI_Offset size{0};
      MPI_File_get_size(file_, &size);
      return size;
    }

    /// get file open-mode
    /// \return file open-mode
    [[nodiscard]] openmode mode() const {
      int mode{0};
      MPI_File_get_amode(file_, &mode);
      return static_cast<openmode>(mode);
    }

    template<typename T>
    /// \tparam T elementary read/write data type
    void set_view(size_t displacement, const layout<T> &l, const char *representation) {
      MPI_File_set_view(file_, displacement, detail::datatype_traits<T>::get_datatype(),
                        detail::datatype_traits<layout<T>>::get_datatype(l), representation,
                        MPI_INFO_NULL);
    }

    /// read data from file, blocking, non-collective, explicit offset
    /// \tparam T read data type
    /// \param offset file offset in bytes
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read_at(ssize_t offset, T &data) {
      status_t s;
      MPI_File_read_at(file_, offset, &data, 1, detail::datatype_traits<T>::get_datatype(),
                       static_cast<MPI_Status *>(&s));
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
      MPI_File_read_at(file_, offset, data, 1,
                       detail::datatype_traits<layout<T>>::get_datatype(l),
                       static_cast<MPI_Status *>(&s));
      return s;
    }

    /// read data from file, blocking, non-collective, file-pointer based
    /// \tparam T read data type
    /// \param data value to read
    /// \return status of performed i/o operation
    template<typename T>
    status_t read(T &data) {
      status_t s;
      MPI_File_read(file_, &data, 1, detail::datatype_traits<T>::get_datatype(),
                    static_cast<MPI_Status *>(&s));
      return s;
    }

    /// read data from file, blocking, non-collective, file-pointer based
    /// \tparam T read data type
    /// \param data pointer to the data to read
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t read(T *data, const layout<T> &l) {
      status_t s;
      MPI_File_read(file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                    static_cast<MPI_Status *>(&s));
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
      MPI_File_write_at(file_, offset, &data, 1, detail::datatype_traits<T>::get_datatype(),
                        static_cast<MPI_Status *>(&s));
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
      MPI_File_write_at(file_, offset, data, 1,
                        detail::datatype_traits<layout<T>>::get_datatype(l),
                        static_cast<MPI_Status *>(&s));
      return s;
    }

    /// write data to file, blocking, non-collective, file-pointer based
    /// \tparam T write data type
    /// \param data value to write
    /// \return status of performed i/o operation
    template<typename T>
    status_t write(const T &data) {
      status_t s;
      MPI_File_write(file_, &data, 1, detail::datatype_traits<T>::get_datatype(),
                     static_cast<MPI_Status *>(&s));
      return s;
    }

    /// write data to file, blocking, non-collective, file-pointer based
    /// \tparam T write data type
    /// \param data pointer to the data to write
    /// \param l layout used in associated i/o operation
    /// \return status of performed i/o operation
    template<typename T>
    status_t write(const T *data, const layout<T> &l) {
      status_t s;
      MPI_File_write(file_, data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                     static_cast<MPI_Status *>(&s));
      return s;
    }
  };


  inline file::openmode operator|(file::openmode mode1, file::openmode mode2) {
    using int_type = std::underlying_type_t<file::openmode>;
    return static_cast<file::openmode>(static_cast<int_type>(mode1) |
                                       static_cast<int_type>(mode2));
  }


  inline file::openmode &operator|=(file::openmode &mode1, file::openmode mode2) {
    using int_type = std::underlying_type_t<file::openmode>;
    mode1 = static_cast<file::openmode>(static_cast<int_type>(mode1) |
                                        static_cast<int_type>(mode2));
    return mode1;
  }

}  // namespace mpl

#endif
