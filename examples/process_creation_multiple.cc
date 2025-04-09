#include <cstdlib>
// include MPL header file
#include <mpl/mpl.hpp>


int main() {
  using namespace std::string_literals;
  // get a reference to communicator "world"
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // spawn 2 new processes
  mpl::info info;
  info.set("host", "localhost");
  auto inter_comm{
      comm_world.spawn_multiple(0,
                                {{"./process_creation_client"s, "arg1"s},
                                 {"./process_creation_client"s, "arg1"s, "arg2"s},
                                 {"./process_creation_client"s, "arg1"s, "arg2"s, "arg3"s}},
                                {info, info, info})};
  // broadcast a message to the created processes
  double message;
  if (comm_world.rank() == 0) {
    // root rank
    message = 1.23;
    inter_comm.bcast(mpl::root, message);
  } else
    // non-root ranks
    inter_comm.bcast(mpl::proc_null, message);

  return EXIT_SUCCESS;
}
