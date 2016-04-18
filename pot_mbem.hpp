#ifndef W_POT_MBEM_H
#define W_POT_MBEM_H

#include <armadillo>
#include <vector>

namespace willow { namespace mbem {

extern double pot (const arma::mat& pos,
		   const std::vector<unsigned short>& Zs);


} } // namespace willow::embem


#endif
