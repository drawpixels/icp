//
// nicp_lm_functor.h
//

#ifndef NICP_LM_FUNCTOR_H
#define NICP_LM_FUNCTOR_H

#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include "mesh.h"
#include "Deformable.h"

struct nicp_lm_functor : SparseFunctor<double,int>
{
	Deformable mSource;
	Mesh mTarget;
	double c_fit, c_rigid, c_smooth;
	
	nicp_lm_functor (const Deformable& src, const Mesh& tgt, double f=1.0, double r=1.0, double s=1.0);
	void SetCoeff (double f, double r, double s);
    int operator() (const VectorXd &x, VectorXd &fvec) const;
    int df (const VectorXd &x, JacobianType &fjac) const;
};

#endif
