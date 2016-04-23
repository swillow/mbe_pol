#ifndef W_MESSAGE_H
#define W_MESSAGE_H

#include <string>
#include <armadillo>

namespace willow { namespace mpi {

extern void init (int argc, char *argv[]);

extern void finalize (); 

extern int rank ();

extern int size ();

extern void barrier ();

extern void broadcast_dbl (arma::vec& data);

extern void broadcast_int (arma::ivec& data);

extern void broadcast_string (std::string& data);

extern void reduce_dbl_sum (arma::vec& data); 

extern void reduce_int_sum (arma::ivec& data);

extern void reduce_dbl_sum_1 (double& data); 

extern void reduce_dbl_mat_sum (arma::mat& data); 


}  } // namespace willow::mpi


#endif
