#include <cstring>
#include <string>
#include <iostream>
#include <mpi.h>
#include "message.hpp"


namespace willow { namespace mpi {


void init (int argc, char *argv[])
{
  
  MPI::Init (argc, argv);
  
}



void finalize ()
{
  MPI::Finalize ();
}



int rank () 
{
  
  int world_rank = MPI::COMM_WORLD.Get_rank (); 

  return world_rank;
  
}


int size ()
{

  int world_size = MPI::COMM_WORLD.Get_size ();

  return world_size;

}



void barrier () 
{
  MPI::COMM_WORLD.Barrier ();
}


void broadcast_dbl (arma::vec& data) 
{

  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI::DOUBLE, 0);
  
};

void broadcast_int (arma::ivec& data)
{
  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI_INT, 0);
  
};


void broadcast_string (std::string& data)
{
  arma::ivec str_len(1);
  str_len(0) = data.length() + 1;
  
  mpi::broadcast_int (str_len);
  mpi::barrier ();

  const size_t slen = str_len(0);
  char* cstr = new char [slen]; 
  std::strcpy (cstr, data.c_str());
  
  MPI::COMM_WORLD.Bcast (cstr, slen, MPI::CHAR, 0);

  data = std::string (cstr);

  delete [] cstr;
  
//  return ret_data;
  
}


void reduce_dbl_sum (arma::vec& data)
{

  const size_t count = data.n_elem;
  arma::vec recv_data(count);
  recv_data.zeros();
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(),
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
};


void reduce_int_sum (arma::ivec& data)
{
  const size_t count = data.n_elem;
  arma::ivec recv_data(count);
  recv_data.zeros();
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(), count,
			  MPI::INT, MPI::SUM, 0);

  data = recv_data;
  
};



void reduce_dbl_sum_1 (double& data)
{

  double recv_data = 0.0;
  const size_t count = 1;
  
  MPI::COMM_WORLD.Reduce (&data, &recv_data,
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
  
};




void reduce_dbl_mat_sum (arma::mat& data)
{

  const size_t n_rows = data.n_rows;
  const size_t n_cols = data.n_cols;
  const size_t count = data.n_elem;
  arma::mat recv_data(n_rows, n_cols);
  
  recv_data.zeros();
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(),
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
};



}  } // namespace willow::mpi
/*

using namespace willow;

int main (int argc, char* argv[])
{
  
  mpi::init (argc, argv);

  auto sys_nproc = mpi::size ();
  auto sys_me = mpi::rank ();

  std::string my_data;

  if (sys_me == 0) my_data = "Hello MPI ";

  int str_len = my_data.length() + 1;

  mpi::broadcast_int (&str_len, 1);
  
  mpi::broadcast_string (my_data); 

  std::cout << " sys_me " << sys_me << "  "  << str_len << std::endl;
  
  mpi::finalize ();
  
}

*/
