#if !(defined MPL_COMMANDLINE_HPP)

#define MPL_COMMANDLINE_HPP

#include <cstddef>
#include <vector>
#include <utility>
#include <string>

namespace mpl {

  /// Represents a collection of command-line arguments.
  /// \see class \c communicator::spawn
  class command_line : private std::vector<std::string> {
    using base = std::vector<std::string>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// Constructs an empty collection of command-line arguments.
    explicit command_line() : base() {}

    /// Constructs collection of command-line arguments from a braces expression of strings.
    /// \param init list of initial values
    command_line(std::initializer_list<std::string> init) : base(init) {}

    /// Constructs collection of command-line arguments from another collection.
    /// \param other the other collection to copy from
    command_line(const command_line &other) = default;

    /// Move-constructs collection of command-line arguments from another collection.
    /// \param other the other collection to move from
    command_line(command_line &&other) noexcept : base(std::move(other)) {}

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
  };

}  // namespace mpl

#endif
