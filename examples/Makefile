MPL_HEADERS = $(shell ls ../mpl/*hpp)
BINS = hello_world blocking blocking_vector probe nonblocking nonblocking_mult standard_types arrays struct gather collective reduce_lcm reduce_min_loc layouts subarray communicator vibrating_string_mpi vibrating_string_mpl matrix_gather distributed_grid distributed_grid_scatter_gather heat_equation_successive_over-relaxation heat_equation_Jacobi_method
CC = mpicc
CXX = mpicxx
#CXXFLAGS = -std=c++98 -pedantic -Wall -Wno-long-long
CCFLAGS = -std=c99 -Wall -O3
CXXFLAGS = -std=c++11 -Wall -O3 -DMPL_DEBUG

all: $(BINS)

hello_world: hello_world.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o hello_world -I .. hello_world.cc

blocking: blocking.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o blocking -I .. blocking.cc

blocking_vector: blocking_vector.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o blocking_vector -I .. blocking_vector.cc

probe: probe.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o probe -I .. probe.cc

nonblocking: nonblocking.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o nonblocking -I .. nonblocking.cc

nonblocking_mult: nonblocking_mult.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o nonblocking_mult -I .. nonblocking_mult.cc

standard_types: standard_types.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o standard_types -I .. standard_types.cc

arrays: arrays.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o arrays -I .. arrays.cc

struct: struct.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o struct -I .. struct.cc

gather: gather.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o gather -I .. gather.cc

collective: collective.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o collective -I .. collective.cc

reduce_lcm: reduce_lcm.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o reduce_lcm -I .. reduce_lcm.cc

reduce_min_loc: reduce_min_loc.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o reduce_min_loc -I .. reduce_min_loc.cc

layouts: layouts.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o layouts -I .. layouts.cc

subarray: subarray.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o subarray -I .. subarray.cc

communicator: communicator.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o communicator -I .. communicator.cc

vibrating_string_mpi: vibrating_string_mpi.c
	$(CC) $(CCFLAGS) -o vibrating_string_mpi -I .. vibrating_string_mpi.c -lm

vibrating_string_mpl: vibrating_string_mpl.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o vibrating_string_mpl -I .. vibrating_string_mpl.cc

matrix_gather: matrix_gather.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o matrix_gather -I .. matrix_gather.cc

distributed_grid: distributed_grid.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o distributed_grid -I .. distributed_grid.cc

distributed_grid_scatter_gather: distributed_grid_scatter_gather.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o distributed_grid_scatter_gather -I .. distributed_grid_scatter_gather.cc

heat_equation_successive_over-relaxation: heat_equation_successive_over-relaxation.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o heat_equation_successive_over-relaxation -I .. heat_equation_successive_over-relaxation.cc

heat_equation_Jacobi_method: heat_equation_Jacobi_method.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o heat_equation_Jacobi_method -I .. heat_equation_Jacobi_method.cc

clean:
	rm -f $(BINS) *.o
