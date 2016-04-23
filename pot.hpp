#ifndef W_POT_EMBEM_H
#define W_POT_EMBEM_H

#include <armadillo>
#include <vector>

namespace willow { namespace pot {


extern void   libint2_init ();
extern void   libint2_end  ();

extern double embem (const arma::mat& pos,
		     const arma::ivec& Zs,
		     const arma::vec& Qs);

extern double  mbem (const arma::mat& pos,
		     const arma::ivec& Zs);

extern void esp_update (const arma::mat& pos,
			const arma::ivec& Zs,
			arma::vec& Qs);

} } // namespace willow::pot


#endif
