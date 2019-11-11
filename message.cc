#include <cstring>
#include <string>
#include <iostream>
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


void abort (int errorcode)
{
  MPI::COMM_WORLD.Abort(errorcode);
}


void broadcast_dbl   (double& data) 
{

  const size_t count = 1;
  
  MPI::COMM_WORLD.Bcast (&data, count, MPI::DOUBLE, 0);
  
}


void broadcast_dbl_vec (arma::vec& data) 
{

  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI::DOUBLE, 0);
  
}



void broadcast_dbl_mat (arma::mat& data) 
{

  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI::DOUBLE, 0);
  
}

 

void broadcast_dbl_cube(arma::cube& data) 
{

  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI::DOUBLE, 0);
  
}


void broadcast_int  (int& data)
{
  const size_t count = 1;
  
  MPI::COMM_WORLD.Bcast (&data, count, MPI_INT, 0);

}



void broadcast_int_vec (arma::ivec& data)
{
  const size_t count = data.n_elem;
  
  MPI::COMM_WORLD.Bcast (data.memptr(), count, MPI_INT, 0);
  
}



void broadcast_string (std::string& data)
{
  int str_len = data.length() + 1;
  
  mpi::broadcast_int (str_len);
  mpi::barrier ();

  const size_t slen = str_len;
  char* cstr = new char [slen]; 
  std::strcpy (cstr, data.c_str());
  
  MPI::COMM_WORLD.Bcast (cstr, slen, MPI::CHAR, 0);

  data = std::string (cstr);

  delete [] cstr;
  
}





void reduce_dbl_sum (double& data)
{

  double recv_data = 0.0;
  const size_t count = 1;
  
  MPI::COMM_WORLD.Reduce (&data, &recv_data,
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
  
}



void reduce_dbl_vec_sum (arma::vec& data)
{

  const size_t count = data.n_elem;
  arma::vec recv_data(count);
  recv_data.zeros();
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(),
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
}




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
}




void reduce_dbl_cube_sum(arma::cube& data)
{

  const size_t n_rows = data.n_rows;
  const size_t n_cols = data.n_cols;
  const size_t n_slices = data.n_slices;
  
  const size_t count = data.n_elem;
  arma::cube recv_data(n_rows, n_cols, n_slices, arma::fill::zeros);
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(),
			  count, MPI::DOUBLE, MPI::SUM, 0);

  data = recv_data;
}


void reduce_int_sum (int& data)
{

  int      recv_data = 0.0;
  const size_t count = 1;
  
  MPI::COMM_WORLD.Reduce (&data, &recv_data,
			  count, MPI::INT, MPI::SUM, 0);

  data = recv_data;
  
}




void reduce_int_vec_sum (arma::ivec& data)
{
  const size_t count = data.n_elem;
  arma::ivec recv_data(count);
  recv_data.zeros();
  
  MPI::COMM_WORLD.Reduce (data.memptr(), recv_data.memptr(), count,
			  MPI::INT, MPI::SUM, 0);

  data = recv_data;
  
}



void allreduce_int_sum (int& data)
{
  
  int recv_data = 0.0;
  const size_t count = 1;
  
  MPI::COMM_WORLD.Allreduce (&data, &recv_data,
			     count, MPI::INT, MPI::SUM);
  
  data = recv_data;
  
}



void allreduce_dbl_sum (double& data)
{
  
  double recv_data = 0.0;
  const size_t count = 1;
  
  MPI::COMM_WORLD.Allreduce (&data, &recv_data,
			     count, MPI::DOUBLE, MPI::SUM);
  
  data = recv_data;
  
}



void allreduce_dbl_vec_sum (arma::vec& data)
{

  const size_t count = data.n_elem;
  arma::vec recv_data(count);
  recv_data.zeros();
  
  MPI::COMM_WORLD.Allreduce (data.memptr(), recv_data.memptr(),
			     count, MPI::DOUBLE, MPI::SUM);

  data = recv_data;
}




void allreduce_dbl_mat_sum (arma::mat& data)
{

  const size_t n_rows = data.n_rows;
  const size_t n_cols = data.n_cols;
  const size_t count = data.n_elem;
  arma::mat recv_data(n_rows, n_cols);
  
  recv_data.zeros();
  
  MPI::COMM_WORLD.Allreduce (data.memptr(), recv_data.memptr(),
			     count, MPI::DOUBLE, MPI::SUM);

  data = recv_data;
}



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
