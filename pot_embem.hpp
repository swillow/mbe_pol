#ifndef W_POT_EMBEM_H
#define W_POT_EMBEM_H

#include <armadillo>
#include <vector>

namespace willow { namespace embem {


extern void   pot_init ();
extern void   pot_end  ();
extern double pot (const arma::mat& pos,
		   const std::vector<unsigned short>& Zs,
		   const std::vector<double>& Qs);


} } // namespace willow::embem


#endif
