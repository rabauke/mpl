#if !(defined MPL_COMMANDLINE_HPP)

#define MPL_COMMANDLINE_HPP

#include <cstddef>
#include <vector>
#include <utility>
#include <string>


namespace mpl {

  /// Represents a set of command-line arguments.
  /// \see class \c communicator::spawn
  class command_line : private std::vector<std::string> {
    using base = std::vector<std::string>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// Constructs an empty set of command-line arguments.
    explicit command_line() = default;

    /// Constructs set of command-line arguments from a braces expression of strings.
    /// \param init list of initial values
    command_line(std::initializer_list<std::string> init) : base(init) {
    }

    /// Constructs set of command-line arguments from another set.
    /// \param other the other set to copy from
    command_line(const command_line &other) = default;

    /// Move-constructs set of command-line arguments from another set.
    /// \param other the other set to move from
    command_line(command_line &&other) noexcept : base(std::move(other)) {
    }

    using base::operator=;
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    using base::size;
    using base::push_back;
  };

  /// Represents a list of command-line argument sets.
  /// \see class \c communicator::spawn_multiple
  class command_lines : private std::vector<command_line> {
    using base = std::vector<command_line>;

  public:
    using base::size_type;
    using base::value_type;
    using base::iterator;
    using base::const_iterator;

    /// Constructs an empty list of command-line argument sets.
    command_lines() = default;

    /// Constructs list of command-line argument sets from a braces expression of strings.
    /// \param init list of initial values
    command_lines(std::initializer_list<command_line> init) : base(init) {
    }

    /// Constructs list of command-line argument sets from another list.
    /// \param other the other list to copy from
    command_lines(const command_lines &other) = default;

    /// Move-constructs list of command-line argument sets from another list.
    /// \param other the other list to move from
    command_lines(command_lines &&other) noexcept : base(std::move(other)) {
    }

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
