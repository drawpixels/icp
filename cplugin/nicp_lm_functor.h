//
// nicp_lm_functor.h
//

#ifndef NICP_LM_FUNCTOR_H
#define NICP_LM_FUNCTOR_H

#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>
#include "mesh.h"
#include "Deformable.h"


// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	//Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	//Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct nicp_lm_functor : Functor<double>
{
	Deformable mSource;
	Mesh mTarget;
	double c_fit, c_rigid, c_smooth;
	
	nicp_lm_functor (const Deformable& src, const Mesh& tgt, double f=1.0, double r=1.0, double s=1.0);
	void SetCoeff (double f, double r, double s);
    int operator() (const VectorXd &x, VectorXd &fvec) const;
    int df (const VectorXd &x, MatrixXd &fjac) const;
};

#endif
