#if !(defined MPL_INFO_HPP)

#define MPL_INFO_HPP

#include <string>
#include <string_view>
#include <optional>


namespace mpl {

  namespace impl {
    class base_communicator;
  }

  /// Stores key-value pairs to affect specific as well as implementation defined MPI
  /// functionalities.
  class info {
    MPI_Info info_{MPI_INFO_NULL};

    explicit info(MPI_Info info) : info_{info} {}

  public:
    /// Creates a new info object with no key-value pairs attached.
    info() { MPI_Info_create(&info_); }

    /// Copy-constructs a new info object.
    /// \param other the other info object to copy from
    info(const info &other) { MPI_Info_dup(other.info_, &info_); }

    /// Move-constructs a new info object.
    /// \param other the other info object to move from
    info(info &&other) noexcept : info_{other.info_} { other.info_ = MPI_INFO_NULL; }

    /// Copies an info object.
    /// \param other the other info object to copy from
    info &operator()(const info &other) {
      if (this != &other) {
        if (info_ != MPI_INFO_NULL)
          MPI_Info_free(&info_);
        MPI_Info_dup(other.info_, &info_);
      }
      return *this;
    }

    /// Moves an info object.
    /// \param other the other info object to move from
    info &operator()(info &&other) noexcept {
      if (this != &other) {
        if (info_ != MPI_INFO_NULL)
          MPI_Info_free(&info_);
        info_ = other.info_;
        other.info_ = MPI_INFO_NULL;
      }
      return *this;
    }

    /// Destructor.
    ~info() {
      if (info_ != MPI_INFO_NULL)
        MPI_Info_free(&info_);
    }

    /// Stores a key-value pair.
    /// \param key the key
    /// \param value the value
    void set(std::string_view key, std::string_view value) {
      MPI_Info_set(info_, key.data(), value.data());
    }

    /// Removes a key-value pair with the the given key.
    /// \param key the key
    void remove(std::string_view key) { MPI_Info_delete(info_, key.data()); }

    /// Retrieves the value for a given key.
    /// \param key the key
    /// \return the value if the info object contains a key-value pair with the given key.
    [[nodiscard]] std::optional<std::string> value(std::string_view key) const {
      int flag{0};
#if MPI_VERSION < 4
      std::vector<char> str(MPI_MAX_INFO_VAL + 1);
      MPI_Info_get(info_, key.data(), MPI_MAX_INFO_VAL, str.data(), &flag);
#else
      int bufflen{0};
      MPI_Info_get_string(info_, key.data(), &bufflen, nullptr, &flag);
      std::vector<char> str(bufflen);
      MPI_Info_get_string(info_, key.data(), &bufflen, str.data(), &flag);
#endif
      if (flag)
        return std::string{str.data()};
      return {};
    }

    /// Gets the number of key-value pairs.
    /// \return number of key-value pairs in the info object
    [[nodiscard]] int size() const {
      int number_of_keys{0};
      MPI_Info_get_nkeys(info_, &number_of_keys);
      return number_of_keys;
    }

    /// Gets the nth key.
    /// \param n index, must be non-negative but less than size()
    /// \return the nth key in the info object
    [[nodiscard]] std::string key(int n) const {
      if (0 <= n and n < size()) {
        std::vector<char> str(MPI_MAX_INFO_VAL + 1);
        MPI_Info_get_nthkey(info_, n, str.data());
        return std::string{str.data()};
      }
      return {};
    }

    friend class impl::base_communicator;
  };
}  // namespace mpl

#endif  // MPL_INFO_HPP
