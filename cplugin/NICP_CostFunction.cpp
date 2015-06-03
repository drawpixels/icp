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
#include <ceres/ceres.h>
#include "mesh.h"
#include "Deformable.h"
#include "NICP_CostFunction.h"

EFit_CostFunction::EFit_CostFunction (const Deformable& src, const Mesh& tgt, double f) 
  : mSource(src), mTarget(tgt), c_fit(f), NumParams(src.NumVertices()*12+6) 
{
	for (int i=0; i<src.NumVertices(); i++)
		mutable_parameter_block_sizes()->push_back(12);
	mutable_parameter_block_sizes()->push_back(6);
	set_num_residuals(src.NumVertices()*3);
}

ERigid_CostFunction::ERigid_CostFunction (const Deformable& src, const Mesh& tgt, double r) 
  : mSource(src), mTarget(tgt), c_rigid(r), NumParams(src.NumVertices()*12+6) 
{
	for (int i=0; i<src.NumVertices(); i++)
		mutable_parameter_block_sizes()->push_back(12);
	mutable_parameter_block_sizes()->push_back(6);
	set_num_residuals(src.NumVertices()*6);
}

ESmooth_CostFunction::ESmooth_CostFunction (const Deformable& src, const Mesh& tgt, double s) 
  : mSource(src), mTarget(tgt), c_smooth(s), NumParams(src.NumVertices()*12+6) 
{
	for (int i=0; i<src.NumVertices(); i++)
		mutable_parameter_block_sizes()->push_back(12);
	mutable_parameter_block_sizes()->push_back(6);
	set_num_residuals(src.NumEdges()*6);
}

void SetJac (double* jac, const int row, const int col, const MatrixXd &m, const int size)
{
	for (int i=0; i<m.rows(); i++) 
		for (int j=0; j<m.cols(); j++) 
			jac[(row+i) * size + (col+j)] = m(i,j);
}

void SetParameterVector (VectorXd &vecParams, double const* const* arrParams, const int nVerts) 
{
	int idx = 0;
	//-- 12 local parameters per vertices
	for (int i=0; i<nVerts; i++) 
		for (int j=0; j<12; j++) 
			vecParams(idx++) = arrParams[i][j];
	//-- 6 global parameters
	for (int j=0; j<6; j++) 
		vecParams(idx++) = arrParams[nVerts][j];
}

bool EFit_CostFunction::Evaluate (double const* const* parameters, double* residuals, double** jacobians) const 
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();
	// Convert arguments into Eigen vectors
	VectorXd vParams(NumParams);
	SetParameterVector(vParams,parameters,nVertices);
	//-- E-fit
	//std::cout << "E-fit\n";
	Mesh DD = mSource.Deform(vParams);
	RowVector3d delta;
	for (int i=0, idx=0; i<nVertices; i++) {
		delta = mTarget.Vertex(i) - DD.Vertex(i);
		residuals[idx++] = delta(0) * c_fit;
		residuals[idx++] = delta(1) * c_fit;
		residuals[idx++] = delta(2) * c_fit;
	}
	//-- Calculate the Jacobian
	if (jacobians != NULL) {
		int idx=0;	//-- Running index to fill up the output values
		Mesh DL = mSource.Deform(vParams,true);
		double w;
		RowVector3d delta;
		Matrix3d rot, rot_t, drot_dx, drot_dy, drot_dz;
		rot = Deformable::RotMatrix(vParams(NumParams-6),vParams(NumParams-5),vParams(NumParams-4));
		rot_t = rot.transpose();
		Deformable::D_RotMatrix(vParams(NumParams-6),vParams(NumParams-5),vParams(NumParams-4),drot_dx,drot_dy,drot_dz);
		for (int i=0; i<nVertices; i++) {
			for (int j=0; j<nVertices; j++) {
				if (jacobians[j] != NULL) {
					w = mSource.Weights()(j,i);
					if (w!=0) {
						delta = mSource.Vertex(i) - mSource.Vertex(j);
						SetJac(jacobians[j], i*3, 0, -c_fit * w * delta(0) * rot_t.col(0), 12);
						SetJac(jacobians[j], i*3, 1, -c_fit * w * delta(0) * rot_t.col(1), 12);
						SetJac(jacobians[j], i*3, 2, -c_fit * w * delta(0) * rot_t.col(2), 12);
						SetJac(jacobians[j], i*3, 3, -c_fit * w * delta(1) * rot_t.col(0), 12);
						SetJac(jacobians[j], i*3, 4, -c_fit * w * delta(1) * rot_t.col(1), 12);
						SetJac(jacobians[j], i*3, 5, -c_fit * w * delta(1) * rot_t.col(2), 12);
						SetJac(jacobians[j], i*3, 6, -c_fit * w * delta(2) * rot_t.col(0), 12);
						SetJac(jacobians[j], i*3, 7, -c_fit * w * delta(2) * rot_t.col(1), 12);
						SetJac(jacobians[j], i*3, 8, -c_fit * w * delta(2) * rot_t.col(2), 12);
						SetJac(jacobians[j], i*3, 9, -c_fit * w * rot_t.col(0), 12);
						SetJac(jacobians[j], i*3,10, -c_fit * w * rot_t.col(1), 12);
						SetJac(jacobians[j], i*3,11, -c_fit * w * rot_t.col(2), 12);
					} else {
						SetJac(jacobians[j], i*3, 0, MatrixXd::Zero(3,12), 12);
					}
				}
			}
			if (jacobians[nVertices] != NULL) {
				SetJac(jacobians[nVertices], i*3, 0, -c_fit * (DL.Vertex(i) * drot_dx).transpose(), 6);
				SetJac(jacobians[nVertices], i*3, 1, -c_fit * (DL.Vertex(i) * drot_dy).transpose(), 6);
				SetJac(jacobians[nVertices], i*3, 2, -c_fit * (DL.Vertex(i) * drot_dz).transpose(), 6);
				SetJac(jacobians[nVertices], i*3, 3, (Matrix3d() << -c_fit,0,0,  0,-c_fit,0,  0,0,-c_fit).finished(), 6);
			}
		}
	}
	return true;
}

bool ERigid_CostFunction::Evaluate (double const* const* parameters, double* residuals, double** jacobians) const 
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();
	// Convert arguments into Eigen vectors
	VectorXd vParams(NumParams);
	SetParameterVector(vParams,parameters,nVertices);
	//-- E-rigid
	//std::cout << "E-Rigid\n";
	Matrix3d A;
	RowVector3d b;
	RowVector3d a0, a1, a2;
	for (int i=0, idx=0; i<nVertices; i++) {
		Deformable::PtoAB(vParams,i,A,b);
		a0 = A.row(0);
		a1 = A.row(1);
		a2 = A.row(2);
		residuals[idx++] = a0.dot(a1) * c_rigid;
		residuals[idx++] = a1.dot(a2) * c_rigid;
		residuals[idx++] = a2.dot(a0) * c_rigid;
		residuals[idx++] = (1.0 - a0.dot(a0)) * c_rigid;
		residuals[idx++] = (1.0 - a1.dot(a1)) * c_rigid;
		residuals[idx++] = (1.0 - a2.dot(a2)) * c_rigid;
	}
	//-- Calculate the Jacobian
	if (jacobians != NULL) {
		//std::cout << "Jacobian E-Rigid\n";
		//-- Calculate Jacobian for a set of local parameters for a vertex
		for (int i=0; i<nVertices; i++) {
			if (jacobians[i] != NULL) {
				//-- Initialise jacobian to zero
				SetJac (jacobians[i], 0, 0, MatrixXd::Zero(nVertices*6,12), 12);
				//-- Set jacobian for vertex i
				Deformable::PtoAB(vParams,i,A,b);
				SetJac (jacobians[i], i*6, 0, c_rigid * (MatrixXd(6,9) <<
					A(1,0),A(1,1),A(1,2),  A(0,0),A(0,1),A(0,2),  0,0,0,
					0,0,0,  A(2,0),A(2,1),A(2,2),  A(1,0),A(1,1),A(1,2),
					A(2,0),A(2,1),A(2,2),  0,0,0,  A(0,0),A(0,1),A(0,2),
					-2*A(0,0),-2*A(0,1),-2*A(0,2),  0,0,0,  0,0,0,
					0,0,0,  -2*A(1,0),-2*A(1,1),-2*A(1,2),  0,0,0,
					0,0,0,  0,0,0,  -2*A(2,0),-2*A(2,1),-2*A(2,2)).finished(), 12);
			}
		}
		//-- Calculate Jacobian for the global parameters
		if (jacobians[nVertices] != NULL) {
			SetJac (jacobians[nVertices], 0, 0, MatrixXd::Zero(nVertices*6,6), 6);
		}
	}
	return true;
}

bool ESmooth_CostFunction::Evaluate (double const* const* parameters, double* residuals, double** jacobians) const 
{
	int nVertices = mSource.NumVertices();
	int nEdges = mSource.NumEdges();
	// Convert arguments into Eigen vectors
	VectorXd vParams(NumParams);
	SetParameterVector(vParams,parameters,nVertices);
	//-- E-smooth
	//std::cout << "E-smooth\n";
	int i0, i1;
	RowVector3d v0, v1;
	Matrix3d A0, A1;
	RowVector3d b0, b1, s0, s1;
	for (int i=0, idx=0; i<nEdges; i++) {
		i0 = mSource.Edge(i)(0);
		i1 = mSource.Edge(i)(1);
		v0 = mSource.Vertex(i0);
		v1 = mSource.Vertex(i1);
		Deformable::PtoAB(vParams,i0,A0,b0);
		Deformable::PtoAB(vParams,i1,A1,b1);
		s0 = (v1-v0)*A0 + (v0+b0) - (v1+b1);
		s1 = (v0-v1)*A1 + (v1+b1) - (v0+b0);
		residuals[idx++] = s0(0) * c_smooth;
		residuals[idx++] = s0(1) * c_smooth;
		residuals[idx++] = s0(2) * c_smooth;
		residuals[idx++] = s1(0) * c_smooth;
		residuals[idx++] = s1(1) * c_smooth;
		residuals[idx++] = s1(2) * c_smooth;
	}
	//-- Calculate the Jacobian
	if (jacobians != NULL) {
		//std::cout << "Jacobian E-smooth\n";
		for (int i=0; i<nVertices; i++) {
			if (jacobians[i] != NULL) {
				//-- Initialise jacobian to zero
				SetJac (jacobians[i], 0, 0, MatrixXd::Zero(nEdges*6,12), 12);
			}
		}
		for (int i=0; i<nEdges; i++) {
			int i0 = mSource.Edge(i)(0);
			int i1 = mSource.Edge(i)(1);
			RowVector3d delta = mSource.Vertex(i1) - mSource.Vertex(i0);
			if (jacobians[i0]!=NULL) {
				SetJac (jacobians[i0], i*6, 0, c_smooth * (MatrixXd(3,12) << 
					delta(0),0,0,  delta(1),0,0,  delta(2),0,0,  1,0,0,
					0,delta(0),0,  0,delta(1),0,  0,delta(2),0,  0,1,0,
					0,0,delta(0),  0,0,delta(1),  0,0,delta(2),  0,0,1).finished(), 12);
				SetJac (jacobians[i0], i*6+3, 9, (Matrix3d() << 
					-c_smooth,0,0,  0,-c_smooth,0,  0,0,-c_smooth).finished(), 12);
			}
			if (jacobians[i1]!=NULL) {
				SetJac (jacobians[i1], i*6+3, 0, c_smooth * (MatrixXd(3,12) << 
					-delta(0),0,0,  -delta(1),0,0,  -delta(2),0,0,  1,0,0,
					0,-delta(0),0,  0,-delta(1),0,  0,-delta(2),0,  0,1,0,
					0,0,-delta(0),  0,0,-delta(1),  0,0,-delta(2),  0,0,1).finished(), 12);
				SetJac (jacobians[i1], i*6, 9, (Matrix3d() << 
				-c_smooth,0,0,  0,-c_smooth,0,  0,0,-c_smooth).finished(), 12);
			}
		}
		//-- Calculate Jacobian for the global parameters
		if (jacobians[nVertices] != NULL) {
			SetJac (jacobians[nVertices], 0, 0, MatrixXd::Zero(nEdges*6,6), 6);
		}
	}
	return true;
}




