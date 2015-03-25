//
// nicp_lm_functor.cpp
//

#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>
#include "mesh.h"
#include "Deformable.h"
#include "nicp_lm_functor.h"

nicp_lm_functor::nicp_lm_functor (const Mesh& src, const Mesh& tgt, double f, double r, double s) 
 : mSource(src), mTarget(tgt), c_fit(f), c_rigid(r), c_smooth(s) {
	m_inputs = mSource.NumVertices()*12 + 6;
	m_values = mSource.NumVertices()*9 + mSource.NumEdges()*6;
}

void nicp_lm_functor::SetCoeff (double f, double r, double s) {
	c_fit = f;
	c_rigid = r;
	c_smooth = s;
}

// x = (in) parameters, fvec = (out) output values using the parameters
int nicp_lm_functor::operator() (const VectorXd &params, VectorXd &fvec) const
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();

	int idx=0;	//-- Running index to fill up the output values
	//-- E-fit
	Mesh DD = mSource.Deform(params);
	for (int i=0; i<nVertices; i++) {
		RowVector3d delta = mTarget.Vertex(i) - DD.Vertex(i);
		fvec(idx++) = delta(0) * c_fit;
		fvec(idx++) = delta(1) * c_fit;
		fvec(idx++) = delta(2) * c_fit;
	}
	//-- E-rigid
	Matrix3d A;
	RowVector3d b;
	RowVector3d a0, a1, a2;
	for (int i; i<nVertices; i++) {
		Deformable::PtoAB(params,i,A,b);
		a0 = A.row(0);
		a1 = A.row(1);
		a2 = A.row(2);
		fvec(idx++) = a0.dot(a1) * c_rigid;
		fvec(idx++) = a1.dot(a2) * c_rigid;
		fvec(idx++) = a2.dot(a0) * c_rigid;
		fvec(idx++) = (1.0 - a0.dot(a0)) * c_rigid;
		fvec(idx++) = (1.0 - a1.dot(a1)) * c_rigid;
		fvec(idx++) = (1.0 - a2.dot(a2)) * c_rigid;
	}
	//-- E-smooth
	int i0, i1;
	RowVector3d v0, v1;
	Matrix3d A0, A1;
	RowVector3d b0, b1, s0, s1;
	for (int i=0; i<nEdges; i++) {
		i0 = mSource.Edge(i)(0);
		i1 = mSource.Edge(i)(1);
		v0 = mSource.Vertex(i0);
		v1 = mSource.Vertex(i1);
		Deformable::PtoAB(params,i0,A0,b0);
		Deformable::PtoAB(params,i1,A1,b1);
		s0 = (v1-v0)*A0 + (v0+b0) - (v1+b1);
		s1 = (v0-v1)*A1 + (v1+b1) - (v0+b0);
		fvec(idx++) = s0(0) * c_smooth;
		fvec(idx++) = s0(1) * c_smooth;
		fvec(idx++) = s0(2) * c_smooth;
		fvec(idx++) = s1(0) * c_smooth;
		fvec(idx++) = s1(1) * c_smooth;
		fvec(idx++) = s1(2) * c_smooth;
	}
    return 0;
}

// x = (in) parameters, fjac = (out) jacobian (15x3 matrix)
int nicp_lm_functor::df (const VectorXd &params, MatrixXd &fjac) const
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();
	int nParams = params.rows();

	int idx=0;	//-- Running index to fill up the output values
	//-- E-fit
	Mesh DD = mSource.Deform(params);
	MatrixXd W = mSource.Weights();
	Matrix3d R = Deformable::RotMatrix(params(nParams-6),params(nParams-5),params(nParams-4));
	RowVector3d R0 = R.row(0);
	RowVector3d R1 = R.row(1);
	RowVector3d R2 = R.row(2);

	Matrix3d dRx, dRy, dRz;
	Deformable::D_RotMatrix(params(nParams-6),params(nParams-5),params(nParams-4),dRx,dRy,dRz);
	RowVector3d disp;
	for (int i=0; i<nVertices; i++) {
		for (int j=0; j<nVertices; j++) {
			if (W(j,i)!=0) {
				disp = -W(j,i) * (mSource.Vertex(i) - mSource.Vertex(j));
				fjac.block<3,1>(idx,j*12)    = c_fit * disp(0) * R0;
				fjac.block<3,1>(idx,j*12+1)  = c_fit * disp(0) * R1;
				fjac.block<3,1>(idx,j*12+2)  = c_fit * disp(0) * R2;
				fjac.block<3,1>(idx,j*12+3)  = c_fit * disp(1) * R0;
				fjac.block<3,1>(idx,j*12+4)  = c_fit * disp(1) * R1;
				fjac.block<3,1>(idx,j*12+5)  = c_fit * disp(1) * R1;
				fjac.block<3,1>(idx,j*12+6)  = c_fit * disp(2) * R0;
				fjac.block<3,1>(idx,j*12+7)  = c_fit * disp(2) * R1;
				fjac.block<3,1>(idx,j*12+8)  = c_fit * disp(2) * R2;
				fjac.block<3,1>(idx,j*12+9)  = c_fit * -W(j,i) * R0;
				fjac.block<3,1>(idx,j*12+10) = c_fit * -W(j,i) * R1;
				fjac.block<3,1>(idx,j*12+11) = c_fit * -W(j,i) * R2;
			}
			fjac.block<3,1>(idx,nParams-6) = c_fit * -DD.Vertex(i) * dRx;
			fjac.block<3,1>(idx,nParams-5) = c_fit * -DD.Vertex(i) * dRy;
			fjac.block<3,1>(idx,nParams-4) = c_fit * -DD.Vertex(i) * dRz;
			fjac.block<3,1>(idx,nParams-3) = Vector3d(-c_fit, 0, 0);
			fjac.block<3,1>(idx,nParams-2) = Vector3d(0, -c_fit, 0);
			fjac.block<3,1>(idx,nParams-1) = Vector3d(0, 0, -c_fit);
		}
		idx += 3;
	}
	// E-rigid 
	Matrix3d A;
	RowVector3d b;
	for (int i=0; i<nVertices; i++) {
		Deformable::PtoAB(params,i,A,b);
		fjac.block<6,9>(idx,i*12) = c_rigid * ( Matrix<double,6,9>() << 
			A(1,0),A(1,1),A(1,2),  A(0,0),A(0,1),A(0,2),  0,0,0,
			0,0,0,  A(2,0),A(2,1),A(2,2),  A(1,0),A(1,1),A(1,2),
			A(2,0),A(2,1),A(2,2),  0,0,0,  A(0,0),A(0,1),A(0,2),
			-2*A(0,0),-2*A(0,1),-2*A(0,2),  0,0,0,  0,0,0,
			0,0,0,  -2*A(1,0),-2*A(1,1),-2*A(1,2),  0,0,0,
			0,0,0,  0,0,0,  -2*A(2,0),-2*A(2,1),-2*A(2,2) ).finished();
		idx += 6;
	}
	// E-smooth
	int i0, i1;
	for (int i=0; i<nVertices; i++) {
		i0 = mSource.Edge(i)(0);
		i1 = mSource.Edge(i)(1);
		disp = mSource.Vertex(i0) - mSource.Vertex(i1);

		fjac.block<3,12>(idx,i0*12) = c_smooth * ( Matrix<double,3,12>() <<
			disp(0),0,0,  disp(1),0,0,  disp(2),0,0,  1,0,0,
			0,disp(0),0,  0,disp(1),0,  0,disp(2),0,  0,1,0,
			0,0,disp(0),  0,0,disp(1),  0,0,disp(2),  0,0,1 ).finished();
		fjac.block<3,3>(idx,i1*12+9) <<
			-c_smooth, 0, 0,
			0, -c_smooth, 0,
			0, 0, -c_smooth;
		idx += 3;

		fjac.block<3,12>(idx,i1*12) = c_smooth * ( Matrix<double,3,12>() <<
			-disp(0),0,0,  -disp(1),0,0,  -disp(2),0,0,  1,0,0,
			0,-disp(0),0,  0,-disp(1),0,  0,-disp(2),0,  0,1,0,
			0,0,-disp(0),  0,0,-disp(1),  0,0,-disp(2),  0,0,1 ).finished();
		fjac.block<3,3>(idx,i0*12+9) <<
			-c_smooth, 0, 0,
			0, -c_smooth, 0,
			0, 0, -c_smooth;
		idx += 3;
	}
    return 0;
}

