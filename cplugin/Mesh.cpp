//
// Mesh.cpp
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <nabo/nabo.h>
#include <maya/MGlobal.h>
#include "Mesh.h"

using namespace std;
using namespace Eigen;

//
// Return the Euclidean distance between 2 points
//
double Distance (const RowVector3d& p1, const RowVector3d& p2) {
	double disp, sumDispSq = 0.0;
	for (int i=0; i<3; i++) {
		disp = p1(i) - p2(i);
		sumDispSq += disp * disp;
	}
	return sqrt(sumDispSq);	
}

//
// Return the indices and distances of the k nearest neighbors in the array
// If pt is in dset, it is the closest point & sort(dist)[0] = 0.0
//
void Mesh::KNN (const RowVector3d& pt, int k, int* idx, double* dist) const {
	if (_NNS==NULL) {
		_VT = _Vertices.transpose();
		_NNS = Nabo::NNSearchD::createKDTreeLinearHeap(_VT);
	}
	Vector3d q = pt;
	VectorXi vidx(k);
	VectorXd vdist(k);
	_NNS->knn(q,vidx,vdist,k,0,Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);
	for (int i=0; i<k; i++) {
		idx[i] = vidx(i);
		dist[i] = sqrt(vdist(i));
	}
}

//
// Return the index of the closest point in the array
//
int Mesh::Closest (const RowVector3d& pt, double* minDist) const {
	int idx; 
	double dist;
	this->KNN(pt,1,&idx,&dist);
	if (minDist)
		*minDist = dist;
	return idx;
}

//
// Return the indices and distances of the k nearest neighbors in the array
// If pt is in dset, it is the closest point & sort(dist)[0] = 0.0
//
// Exhaustive approach - comparing every point with every point
// Not used. Keep for reference.
//
void Mesh::_KNN (const RowVector3d& pt, int k, int* idx, double* dist) const {
	typedef std::pair <double, int> PAIR;
	class compare {
	public:
		bool operator() (const PAIR p1, const PAIR p2) {
			return p1.first < p2.first;
		}
	};
	std::priority_queue<PAIR, std::vector <PAIR>, compare> rank;
	
	//-- Calculate the distance and put into the rank
	double d;
	for (int i=0; i<NumVertices(); i++) {
		d = Distance(Vertex(i),pt);
		if (rank.size()<k) {
			PAIR d_idx(d,i);
			rank.push(d_idx);
		} else {
			PAIR top = rank.top();
			if (top.first>d) {
				rank.pop();
				PAIR d_idx(d,i);
				rank.push(d_idx);
			}
		}
	}
	int size = rank.size();
	//-- Extract sorted list from rank
	for (int i=k; i>0; i--) {
		PAIR t = rank.top();
		idx[i-1] = t.second;
		dist[i-1] = t.first;
		rank.pop();
	}
}

//
// Return the index of the closest point in the array
// Exhaustive approach. Not used. Keep for reference.
//
int Mesh::_Closest (const RowVector3d& pt, double* minDist) const {
	//-- Initialise with first point in dataset 
	double min = Distance(Vertex(0),pt);
	int idx = 0;
	//-- Compare with the other points
	double dist;
	int nLen = NumVertices();
	for (int i=1; i<nLen; i++) {
		//p = TNT::Array1D<double> (dataset.dim2(),dataset[i]);
		//dist = Distance(p,pt);
		dist = Distance(Vertex(i),pt);
		if (dist<min) {
			min = dist;
			idx = i;
		}
	}
	//-- Return minimum distance if required
	if (minDist)
		*minDist = min*min;
	return idx;
}

//
// For each point in source, find the closest point in the target. 
// Return the closest points in array
//
Mesh Mesh::Match (const Mesh& target) const {
	int nLen = NumVertices();
	int idx;
	double d;
	char sInfo[100];
	MatrixX3d m = MatrixX3d (nLen,3);
	for (int i=0; i<nLen; i++) {
		idx = target.Closest(Vertex(i),&d);
		m.row(i) = target.Vertex(idx);
		/* DEBUG PRINT *
		sprintf(sInfo,"%4d %4d (%8.4f) %f %f %f",i,idx,d,m(i,0),m(i,1),m(i,2));
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}
	return Mesh(m,_Edges);
}
