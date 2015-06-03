//
// nicp_lm_functor.h
//

#ifndef NICP_COSTFUNCTION_H
#define NICP_COSTFUNCTION_H

#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <ceres/ceres.h>
#include "mesh.h"
#include "Deformable.h"

class EFit_CostFunction : public ceres::CostFunction
{
public:
	Deformable mSource;
	Mesh mTarget;
	double c_fit;
	int NumParams;
	
	EFit_CostFunction (const Deformable& src, const Mesh& tgt, double c=1.0);
	void SetCoeff (double c) { c_fit=c; };
	virtual bool Evaluate (double const* const* parameters, double* residuals, double ** jacobians) const;
};

class ERigid_CostFunction : public ceres::CostFunction
{
public:
	Deformable mSource;
	Mesh mTarget;
	double c_rigid;
	int NumParams;
	
	ERigid_CostFunction (const Deformable& src, const Mesh& tgt, double c=1.0);
	void SetCoeff (double c) { c_rigid=c; };
	virtual bool Evaluate (double const* const* parameters, double* residuals, double ** jacobians) const;
};

class ESmooth_CostFunction : public ceres::CostFunction
{
public:
	Deformable mSource;
	Mesh mTarget;
	double c_smooth;
	int NumParams;
	
	ESmooth_CostFunction (const Deformable& src, const Mesh& tgt, double c=1.0);
	void SetCoeff (double c) { c_smooth=c; };
	virtual bool Evaluate (double const* const* parameters, double* residuals, double ** jacobians) const;
};

#endif
