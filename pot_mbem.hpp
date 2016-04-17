#ifndef W_POT_MBEM_H
#define W_POT_MBEM_H

#include <armadillo>


namespace willow { namespace embem {

extern double pot_mbem (const arma::mat& pos,
			const vector<unsigned short>& Zs);


} } // namespace willow::embem


#endif
