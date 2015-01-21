#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  std::cout << "Hello world! I am running on \"" 
   	    << mpl::environment::processor_name() 
   	    << "\". My rank is "
  	    << comm_world.rank()
  	    << " out of " 
   	    << comm_world.size() << " processes.\n";
  return EXIT_SUCCESS;
}
