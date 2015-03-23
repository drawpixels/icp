//
// Mesh.h
//

#ifndef MESH_H
#define MESH_H

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class Mesh {
public:
	Mesh () {};
	Mesh (const MatrixX3d& v, const MatrixX2i& e) : _Vertices(v), _Edges(e) {};
	Mesh (const Mesh& m) { _Vertices = m._Vertices; _Edges = m._Edges; };
	const int NumVertices () const { return _Vertices.rows(); };
	const MatrixX3d& Vertices () const { return _Vertices; };
	const MatrixX2i& Edges () const { return _Edges; };
	const RowVector3d Vertex (int i) const { return _Vertices.row(i); };
	RowVector3d Vertex (int i) { return _Vertices.row(i); };
	int KNN (const RowVector3d& pt, int k, int* idx, double* dist) const;
	int Closest (const RowVector3d& pt, double* minDist=NULL) const;
	const MatrixX3d Match (const Mesh& target) const;
//	void Match (const Mesh& target, MatrixX3d& matchset) const;
private:
	MatrixX3d _Vertices;
	MatrixX2i _Edges;
};

double Distance (const RowVector3d& p1, const RowVector3d& p2);

#endif
