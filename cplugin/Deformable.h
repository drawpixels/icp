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

using namespace Eigen;

class Deformable : public Mesh {
public:
	Deformable (const MatrixX3d& v, const MatrixX2i& e, const int k=DEF_KNN);
	Deformable (const Mesh& m) : Deformable (m.Vertices(), m.Edges()) {};
	void Deform (VectorXd& params, MatrixX3d& deformset);
	void PtoAB (VectorXd& params, int i, Matrix3d& A, RowVector3d& b);
	void RotMatrix (double x, double y, double z, Matrix3d& rot);
	void D_RotMatrix (double x, double y, double z, Matrix3d& drot_dx, Matrix3d& drot_dy, Matrix3d& drot_dz);
private:
	int knn;
	MatrixXd weights;
};

#endif
