#include <cstdlib>
#include <iostream>
#include <sstream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // split communicator comm_world into two groups consisting of processes with odd and even
  // rank in comm_world
  const int world_rank{comm_world.rank()};
  const int my_group{world_rank % 2};
  mpl::communicator local_communicator{mpl::communicator::split, comm_world, my_group};
  const int local_leader{0};
  const int remote_leader{my_group == 0 ? 1 : 0};
  // comm_world is used as the communicator that can communicate with processes in the local
  // group as well as in the remote group
  mpl::inter_communicator icom{local_communicator, local_leader, comm_world, remote_leader};
  // gather data from all processes in the remote group
  const int send_data{world_rank};  // as an example, send rank in comm_world
  std::vector<int> recv_data(icom.remote_size());
  // will receive a set of odd or even numbers
  icom.allgather(send_data, recv_data.data());
  // output communicator characteristics and received data
  std::stringstream stream;
  stream << "inter communicator size: " << icom.size() << ";\t"
         << "inter communicator rank: " << icom.rank() << ";\t"
         << "inter communicator remote size: " << icom.remote_size() << ";\t"
         << "gathered data: ";
  for (auto &val : recv_data)
    stream << val << ' ';
  stream << '\n';
  std::cout << stream.str();
  return EXIT_SUCCESS;
}
