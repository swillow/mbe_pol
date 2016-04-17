#ifndef W_POT_EMBEM_H
#define W_POT_EMBEM_H

#include <armadillo>


namespace willow { namespace embem {

extern double pot_embem (const arma::mat& pos,
			 const vector<unsigned short>& Zs,
			 const vector<double>& Qs);


} } // namespace willow::embem


#endif
