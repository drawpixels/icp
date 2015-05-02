//
// nicp_lm_functor.cpp
//

#include <sstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <maya/MGlobal.h>
#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include "mesh.h"
#include "Deformable.h"
#include "nicp_lm_functor.h"

nicp_lm_functor::nicp_lm_functor (const Deformable& src, const Mesh& tgt, double f, double r, double s) 
 :  SparseFunctor<double,int> (src.NumVertices()*12 + 6, src.NumVertices()*9 + src.NumEdges()*6),
	mSource(src), mTarget(tgt), c_fit(f), c_rigid(r), c_smooth(s) {
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
	RowVector3d delta;
	for (int i=0; i<nVertices; i++) {
		delta = mTarget.Vertex(i) - DD.Vertex(i);
		fvec(idx++) = delta(0) * c_fit;
		fvec(idx++) = delta(1) * c_fit;
		fvec(idx++) = delta(2) * c_fit;
	}
	//-- E-rigid
	Matrix3d A;
	RowVector3d b;
	RowVector3d a0, a1, a2;
	for (int i=0; i<nVertices; i++) {
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
	/* DEBUG PRINT *
	char sInfo[500];
	sprintf(sInfo,"fvec=%ld count=%d (%d,%d)",fvec.rows(),idx,nVertices,nEdges);
	MGlobal::displayInfo(sInfo);
	for (int i=0,j=0; i<(idx/10) && i<5; i++,j+=10) {
		sprintf(sInfo,"%4d: %f %f %f %f %f %f %f %f %f %f",
			j,fvec(j),fvec(j+1),fvec(j+2),fvec(j+3),fvec(j+4),
			fvec(j+5),fvec(j+6),fvec(j+7),fvec(j+8),fvec(j+9));
		MGlobal::displayInfo(sInfo);
	}
	* DEBUG PRINT */
	return 0;
}

typedef Eigen::Triplet<double> T;

void Push_Triplet (std::vector<T> &tlist, const int row, const int col, const MatrixXd &m)
{
	for (int i=0; i<m.rows(); i++) 
		for (int j=0; j<m.cols(); j++) 
			tlist.push_back(T(row+i,col+j,m(i,j)));
	/* DEBUG PRINT */
	for (int i=0; i<m.rows(); i++) {
		std::cout << (row+i) << ": ";
		for (int j=0; j<m.cols(); j++) {
			std::cout << (col+j) << "=" << m(i,j) << " ";
		}
		std::cout << std::endl;
	}
	/* DEBUG PRINT */
}

// x = (in) parameters, fjac = (out) jacobian (15x3 matrix)
int nicp_lm_functor::df (const VectorXd &params, JacobianType &fjac) const
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();
	int nParams = params.rows();

	std::vector<T> tripList;
	tripList.reserve(3 * nVertices * (mSource.K()*12+6) + 6 * nVertices * 9 + 6 * nEdges * 15);

	int idx=0;	//-- Running index to fill up the output values

	//-- E-fit
	Mesh DL = mSource.Deform(params,true);
	double w;
	RowVector3d delta;
	Matrix3d rot, rot_t, drot_dx, drot_dy, drot_dz;
	rot = Deformable::RotMatrix(params(nParams-6),params(nParams-5),params(nParams-4));
	rot_t = rot.transpose();
	Deformable::D_RotMatrix(params(nParams-6),params(nParams-5),params(nParams-4),drot_dx,drot_dy,drot_dz);
	for (int i=0; i<nVertices; i++) {
		for (int j=0; j<nVertices; j++) {
			w = mSource.Weights()(j,i);
			if (w!=0) {
				delta = mSource.Vertex(i) - mSource.Vertex(j);
				Push_Triplet (tripList, idx, j*12,   -c_fit * w * delta(0) * rot_t);
				Push_Triplet (tripList, idx, j*12+3, -c_fit * w * delta(1) * rot_t);
				Push_Triplet (tripList, idx, j*12+6, -c_fit * w * delta(2) * rot_t);
				Push_Triplet (tripList, idx, j*12+9, -c_fit * w * rot_t);
			}
		}
		Push_Triplet (tripList, idx, nParams-6, -c_fit * (DL.Vertex(i) * drot_dx).transpose());
		Push_Triplet (tripList, idx, nParams-5, -c_fit * (DL.Vertex(i) * drot_dy).transpose());
		Push_Triplet (tripList, idx, nParams-4, -c_fit * (DL.Vertex(i) * drot_dz).transpose());
		Push_Triplet (tripList, idx, nParams-3, (Matrix3d() << -c_fit,0,0, 0,-c_fit,0, 0,0,-c_fit).finished());
		idx += 3;
		std::cout << "E-fit " << i << std::endl;
	}
	// E-rigid 
	Matrix3d A;
	RowVector3d b;
	for (int i=0; i<nVertices; i++) {
		Deformable::PtoAB(params,i,A,b);
		Push_Triplet (tripList, idx, i*12, c_rigid * (MatrixXd(6,9) <<
			A(1,0),A(1,1),A(1,2),  A(0,0),A(0,1),A(0,2),  0,0,0,
			0,0,0,  A(2,0),A(2,1),A(2,2),  A(1,0),A(1,1),A(1,2),
			A(2,0),A(2,1),A(2,2),  0,0,0,  A(0,0),A(0,1),A(0,2),
			-2*A(0,0),-2*A(0,1),-2*A(0,2),  0,0,0,  0,0,0,
			0,0,0,  -2*A(1,0),-2*A(1,1),-2*A(1,2),  0,0,0,
			0,0,0,  0,0,0,  -2*A(2,0),-2*A(2,1),-2*A(2,2)).finished());
		idx += 6;
		std::cout << "E-rigid " << i << std::endl;
	}
	// E-smooth
	int i0, i1;
	for (int i=0; i<nEdges; i++) {
		i0 = mSource.Edge(i)(0);
		i1 = mSource.Edge(i)(1);
		delta = mSource.Vertex(i1) - mSource.Vertex(i0);

		Push_Triplet (tripList, idx, i0*12,	c_smooth * (MatrixXd(3,12) << 
			delta(0),0,0,  delta(1),0,0,  delta(2),0,0,  1,0,0,
			0,delta(0),0,  0,delta(1),0,  0,delta(2),0,  0,1,0,
			0,0,delta(0),  0,0,delta(1),  0,0,delta(2),  0,0,1).finished());
		Push_Triplet (tripList, idx, i1*12+9, (Matrix3d() << 
			-c_smooth,0,0,  0,-c_smooth,0,  0,0,-c_smooth).finished());
		idx += 3;

		Push_Triplet (tripList, idx, i1*12, c_smooth * (MatrixXd(3,12) << 
			-delta(0),0,0,  -delta(1),0,0,  -delta(2),0,0,  1,0,0,
			0,-delta(0),0,  0,-delta(1),0,  0,-delta(2),0,  0,1,0,
			0,0,-delta(0),  0,0,-delta(1),  0,0,-delta(2),  0,0,1).finished());
		Push_Triplet (tripList, idx, i0*12+9, (Matrix3d() << 
			-c_smooth,0,0,  0,-c_smooth,0,  0,0,-c_smooth).finished());
		idx += 3;
		std::cout << "E-smooth " << i << std::endl;
	}
	fjac.resize(m_values,m_inputs);
	fjac.setFromTriplets(tripList.begin(),tripList.end());
    return 0;
}

