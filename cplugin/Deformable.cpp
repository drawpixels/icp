//
// Deformable.c
//


#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Core>
#include <maya/MGlobal.h>
#include "Mesh.h"
#include "Deformable.h"


Deformable::Deformable (const MatrixX3d& v, const MatrixX2i& e, const int k) 
: Mesh(v,e), _Knn(k) {
	char sInfo[500];
	// Initialise 
	const int nLen = NumVertices();
	//-- Calculate weights
	int idxKnn[_Knn+2];
	double distKnn[_Knn+2];
	double distSum, distMax, denom;
	_Weights = MatrixXd::Zero(nLen,nLen);
	for (int j=0; j<nLen; j++) {
		KNN(Vertex(j),_Knn+2,idxKnn,distKnn);
		/* DEBUG PRINT *
		cout << j << " " << idxKnn[0] << "," << idxKnn[1] << "," << idxKnn[2] << "," << idxKnn[3] << "," << idxKnn[4] << "," << idxKnn[5] << " "
			<< distKnn[0] << "," << distKnn[1] << "," << distKnn[2] << "," << distKnn[3] << "," << distKnn[4] << "," << distKnn[5]
			<< endl;
		sprintf(sInfo,"%2d: %2d %2d %2d %2d %2d %2d %f %f %f %f %f %f",j,
			idxKnn[0],idxKnn[1],idxKnn[2],idxKnn[3],idxKnn[4],idxKnn[5],
			distKnn[0],distKnn[1],distKnn[2],distKnn[3],distKnn[4],distKnn[5]);
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
		distSum = 0;
		distMax = distKnn[_Knn+1];
		for (int i=1; i<_Knn+1; i++)
			distSum += distKnn[i];
		denom = (double)_Knn - distSum / distMax;
		for (int i=1; i<_Knn+1; i++)
			_Weights(idxKnn[i],j) = (1 - distKnn[i]/distMax) / denom;
	}
	/* DEBUG PRINT *
	for (int i=0; i<10; i++) {
		sprintf(sInfo,"%2d %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f", i,
		weights(i,0),weights(i,1),weights(i,2),weights(i,3),weights(i,4),weights(i,5),weights(i,6),weights(i,7),weights(i,8),weights(i,9),
		weights(i,10),weights(i,11),weights(i,12),weights(i,13),weights(i,14),weights(i,15),weights(i,16),weights(i,17),weights(i,18),weights(i,19));
		MGlobal::displayInfo(sInfo);
	}
	* DEBUG PRINT */
}

//
// Deform the mesh using parameters - Local deformation OR Local and Global rotation and translation
//
Mesh Deformable::Deform (const VectorXd& params, const bool local) const {
	char sInfo[500];

	int nLen = NumVertices();
	MatrixX3d deformset = MatrixXd::Zero(nLen,3);

	Matrix3d A;
	RowVector3d b;
	RowVector3d tmp;

	for (int i=0; i<nLen; i++) {
		//deformset[i][0] = deformset[i][1] = deformset[i][2] = 0.0;
		for (int j=0; j<nLen; j++) {
			if (_Weights(j,i)!=0) {
				Deformable::PtoAB(params,j,A,b);
				/* DEBUG PRINT *
				sprintf(sInfo,"(%f,%f,%f %f,%f,%f %f,%f,%f) (%f,%f,%f)",
					A(0,0),A(0,1),A(0,2),
					A(1,0),A(1,1),A(1,2),
					A(2,0),A(2,1),A(2,2),
					b(0),b(1),b(2));
				MGlobal::displayInfo(sInfo);
				* DEBUG PRINT */
				/* DEBUG PRINT *
				if (i<5) {
					sprintf(sInfo,"%d(%f,%f,%f) %d(%f,%f,%f)",i,Vertex(i)(0),Vertex(i)(1),Vertex(i)(2),j,Vertex(j)(0),Vertex(j)(1),Vertex(j)(2));
					MGlobal::displayInfo(sInfo);
					sprintf(sInfo,"A=(%f,%f,%f %f,%f,%f %f,%f,%f)",A(0,0),A(0,1),A(0,2),A(1,0),A(1,1),A(1,2),A(2,0),A(2,1),A(2,2));
					MGlobal::displayInfo(sInfo);
					tmp = Vertex(i) - Vertex(j);
					sprintf(sInfo,"i-j=(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp * A;
					sprintf(sInfo,"t*A=(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp + Vertex(j);
					sprintf(sInfo,"t+j=(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp + b;
					sprintf(sInfo,"t+b=(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp * weights(j,i);
					sprintf(sInfo,"t*w=(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					deformset(i,0) += tmp(0);
					deformset(i,1) += tmp(1);
					deformset(i,2) += tmp(2);
				} else {
				* DEBUG PRINT */
					tmp = Vertex(i) - Vertex(j);
					tmp = tmp * A;
					tmp = tmp + Vertex(j);
					tmp = tmp + b;
					tmp = tmp * _Weights(j,i);
					deformset(i,0) += tmp(0);
					deformset(i,1) += tmp(1);
					deformset(i,2) += tmp(2);
				//}
			}
		}
		/* DEBUG PRINT *
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,Vertex(i)(0),Vertex(i)(1),Vertex(i)(2),deformset(i,0),deformset(i,1),deformset(i,2));
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}
	if (local)
		return Mesh(deformset,Edges());

	//-- Global transformation
	int nParam = params.rows();
	Matrix3d Rot = Deformable::RotMatrix(params[nParam-6],params[nParam-5],params[nParam-4]);
	MatrixXd Txn(nLen,3);
	Txn.col(0) = VectorXd::Constant(nLen,params[nParam-3]);
	Txn.col(1) = VectorXd::Constant(nLen,params[nParam-2]);
	Txn.col(2) = VectorXd::Constant(nLen,params[nParam-1]);
	deformset = deformset * Rot;
	deformset = deformset + Txn;
	/* DEBUG PRINT *
	for (int i=0; i<nLen; i++) {
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,Vertex(i)(0),Vertex(i)(1),Vertex(i)(2),deformset(i,0),deformset(i,1),deformset(i,2));
		MGlobal::displayInfo(sInfo);
	}
	* DEBUG PRINT */
	return Mesh(deformset,Edges());
}

//
// Extract A & b from P for a vertex
//
void Deformable::PtoAB (const VectorXd& params, int i, Matrix3d& A, RowVector3d& b) {
	A.row(0) = params.segment<3>(i*12);
	A.row(1) = params.segment<3>(i*12+3);
	A.row(2) = params.segment<3>(i*12+6);
	b = params.segment<3>(i*12+9);
}

//
// Rotational matrix - rotate sequence = X, Y, Z
//
Matrix3d Deformable::RotMatrix (const double x, const double y, const double z) {
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	Matrix3d rot;
	rot << cy*cz, cy*sz, -sy, 
		sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, sx*cy, 
		cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy; 
	return rot;
}

//
// Partial differentiation of the Rotational matrix - rotate sequence = X, Y, Z
//
void Deformable::D_RotMatrix (const double x, const double y, const double z, Matrix3d& drot_dx, Matrix3d& drot_dy, Matrix3d& drot_dz) {
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	//-- Partial differentiation wrt theta-X
	drot_dx << 0, 0, 0,
		cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy,
		-sx*sy*cz+cx*sz, -sx*sy*sz-cx*cz, -sx*cy;
	//-- Partial differentiation wrt theta-Y
	drot_dy << -sy*cz, -sy*sz, -cy,
		sx*cy*cz, sx*cy*sz, -sx*sy,
		cx*cy*cz, cx*cy*sz, -cx*sy;
	//-- Partial differentiation wrt theta-Z
	drot_dz << -cy*sz, cy*cz, 0,
		-sx*sy*sz-cx*cz, sx*sy*cz-cx*sz, 0,
		-cx*sy*sz+sx*cz, cx*sy*cz+sx*sz, 0;
}

