#ifndef W_MESSAGE_H
#define W_MESSAGE_H

#include <string>
#include <armadillo>
#include <mpi.h>

namespace willow { namespace mpi {

extern void init (int argc, char *argv[]);

extern void finalize (); 

extern int rank ();

extern int size ();

extern void barrier ();

extern void abort (int errorcode);

extern void broadcast_dbl     (double& data);

extern void broadcast_dbl_vec (arma::vec& data);

extern void broadcast_dbl_mat (arma::mat& data);

extern void broadcast_dbl_cube(arma::cube& data);

extern void broadcast_int     (int& data);

extern void broadcast_int_vec (arma::ivec& data);

extern void broadcast_string (std::string& data);


extern void reduce_dbl_sum   (double& data);

extern void reduce_dbl_vec_sum (arma::vec& data); 

extern void reduce_dbl_mat_sum (arma::mat& data);

extern void reduce_dbl_cube_sum(arma::cube& data);

extern void reduce_int_sum     (int& data);

extern void reduce_int_vec_sum (arma::ivec& data);


extern void allreduce_int_sum   (int& data);

extern void allreduce_dbl_sum   (double& data);

extern void allreduce_dbl_vec_sum (arma::vec& data);

extern void allreduce_dbl_mat_sum (arma::mat& data);

}  } // namespace willow::mpi


#endif
