//
// Deformable.h
//

#ifndef DEFORMABLE_H
#define DEFORMABLE_H

#include <Eigen/Core>
#include "Mesh.h"

#define DEF_KNN 4

using namespace std;
using namespace Eigen;

class Deformable : public Mesh {

public:
	// Constructors
	Deformable (const MatrixX3d& v, const MatrixX2i& e, const int k=DEF_KNN);
	Deformable (const Mesh& m) : Deformable (m.Vertices(), m.Edges()) {};
	// Main functions
	Mesh Deform (const VectorXd& params) const;
	// Supporting functions 
	const MatrixXd& Weights() const { return _Weights; };
	static void PtoAB (const VectorXd& params, int i, Matrix3d& A, RowVector3d& b);
	static Matrix3d RotMatrix (const double x, const double y, const double z);
	static void D_RotMatrix (const double x, const double y, const double z, Matrix3d& drot_dx, Matrix3d& drot_dy, Matrix3d& drot_dz);
	

private:
	const int _Knn;
	MatrixXd _Weights;
};

#endif
