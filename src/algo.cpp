#include "algo.hpp"
#include <omp.h>
#include <boost/math/distributions/beta.hpp>

int add(int a, int b) {
    return a + b;
}

double beta(int dofs, double value, int times)
{
    double output;
    
    #pragma omp parallel for
    for (int i=0; i<times; ++i)
    {
        boost::math::beta_distribution<> beta_dist(dofs, dofs);
        output = 2.0*cdf(beta_dist, value);
    }
    
    return output;
}
