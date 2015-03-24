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
	// Constructor
	Mesh () {};
	Mesh (const MatrixX3d& v, const MatrixX2i& e) : _Vertices(v), _Edges(e) {};
	Mesh (const Mesh& m) : _Vertices(m._Vertices), _Edges(m._Edges) {};
	// Main funtions
	Mesh Match (const Mesh& target) const;
	int KNN (const RowVector3d& pt, int k, int* idx, double* dist) const;
	int Closest (const RowVector3d& pt, double* minDist=NULL) const;
	// Supporting functions
	const int NumVertices () const { return _Vertices.rows(); };
	const MatrixX3d& Vertices () const { return _Vertices; };
	const RowVector3d Vertex (int i) const { return _Vertices.row(i); };
	RowVector3d Vertex (int i) { return _Vertices.row(i); };
	const int NumEdges () const { return _Edges.rows(); };
	const MatrixX2i& Edges () const { return _Edges; };
	const RowVector2i Edge (int i) const { return _Edges.row(i); };
	RowVector2i Edge (int i) { return _Edges.row(i); };

private:
	MatrixX3d _Vertices;
	MatrixX2i _Edges;
};

// Global functions
double Distance (const RowVector3d& p1, const RowVector3d& p2);

#endif
