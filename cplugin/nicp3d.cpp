//
// Non-rigid Iterative Closest Point 
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <maya/MGlobal.h>
#include <maya/MTypes.h>
#include <maya/MFnPlugin.h> 
#include <maya/MPxCommand.h>
#include <maya/MSelectionList.h>
#include <maya/MDagPath.h>
#include <maya/MFn.h>
#include <maya/MFnMesh.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>

#define NUM_NN 4
#define INIT_C_FIT 0.1
#define INIT_C_RIGID 1000.0
#define INIT_C_SMOOTH 100.0

using namespace std;
using namespace Eigen;

class nicp3d : public MPxCommand {
public:
	nicp3d() {};
	virtual MStatus doIt(const MArgList& argList);
	static void* creator();
private:
	VectorXd params;
	MatrixXd weights;
	int GetModels (MDagPath& dag1, MDagPath& dag2);
	void GetVerticesEdges (MDagPath dag, MatrixX3d& vert, MatrixX2i& edg);
	void Initialise (const MatrixX3d vert);
	void Match (const MatrixX3d& source, const MatrixX3d& target, MatrixX3d& matchset);
	int KNN (const MatrixX3d& dataset, const RowVector3d& pt, int k, int* idx, double* dist);
	int Closest (const MatrixX3d& dataset, const RowVector3d& pt, double* minDist=NULL);
	void Deform (const MatrixX3d& dataset, MatrixX3d& deformset);
	void PtoAB (int i, Matrix3d& A, RowVector3d& b);
	void RotMatrix (double x, double y, double z, Matrix3d& rot);
	void ModifyVertices (MDagPath path, const MatrixX3d& dataset);
	void D_RotMatrix (double x, double y, double z, Matrix3d& drotx, Matrix3d& droty, Matrix3d& drotz);
};

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

void* nicp3d::creator() { return new nicp3d; }
 
MStatus nicp3d::doIt(const MArgList& argList) {
	MDagPath dagSrc, dagTgt;
	char sInfo[500];
	int n = GetModels(dagSrc,dagTgt);
	if (n!=2) {
		MGlobal::displayInfo("2 models must be selected.");
		return MS::kFailure;
	}
	if (!dagSrc.hasFn(MFn::kMesh) || !dagTgt.hasFn(MFn::kMesh)) {
		MGlobal::displayInfo("Selected models must be mesh");
		return MS::kFailure;
	}
	MatrixX3d ptsSrc, ptsTgt;
	MatrixX2i edgSrc, edgTgt;
	GetVerticesEdges(dagSrc,ptsSrc,edgSrc);
	GetVerticesEdges(dagTgt,ptsTgt,edgTgt);
	sprintf(sInfo,"Source mesh has %ld vertics, %ld edges",ptsSrc.rows(),edgSrc.rows());
	MGlobal::displayInfo(sInfo);
	sprintf(sInfo,"Target mesh has %ld vertics, %ld edges",ptsTgt.rows(),edgTgt.rows());
	MGlobal::displayInfo(sInfo);
	Initialise(ptsSrc);
	MatrixX3d ptsNN;
	Match(ptsSrc,ptsTgt,ptsNN);
	MatrixX3d ptsDD;
	Deform(ptsSrc,ptsDD);
	ModifyVertices(dagSrc,ptsDD);
	return MS::kSuccess;
}

//
// Return the selected models - must select TWO models SRC & TGT
//
int nicp3d::GetModels (MDagPath& dag0, MDagPath& dag1) {
	MSelectionList list;
	MGlobal::getActiveSelectionList(list);
	if (list.length()>=2) {
		list.getDagPath(0,dag0);
		list.getDagPath(1,dag1);
	}
	return list.length();
}

//
// Return the vertices of the mesh in Numpy.Array
//
void nicp3d::GetVerticesEdges (MDagPath dag, MatrixX3d& vert, MatrixX2i & edg) {
	MFnMesh mesh(dag);
	//-- Get vertices
	MPointArray pts;
	mesh.getPoints(pts,MSpace::kWorld);
	int nVert = pts.length();
	MatrixX3d v(nVert,3);
	for (int i=0; i<nVert; i++) {
		v(i,0) = pts[i].x;
		v(i,1) = pts[i].y;
		v(i,2) = pts[i].z;
	}
	vert = v;
	//-- Get edges
	int nEdg = mesh.numEdges();
	MatrixX2i e(nEdg,2);
	int2 v2;
	for (int i=0; i<nEdg; i++) {
		mesh.getEdgeVertices(i,v2);
		e(i,0) = v2[0];
		e(i,1) = v2[1];
	}
	edg = e;
}

//
// Intialise parameters and weights
//
void nicp3d::Initialise (const MatrixX3d vert) {
	char sInfo[512];
	int nLen = vert.rows();
	//--- Initialise parameters: A = Identity, b = zeros, Rxyz = 0, Txyz = 0
	params = VectorXd::Zero(nLen*12+6);
	for (int i=0; i<nLen; i++) {
		int ii = i*12;
		params(ii) = 1.0;
		params(ii+4) = 1.0;
		params(ii+8) = 1.0;
	}
	int nParams = params.rows();
	params(nParams-5) = 0.5;
	params(nParams-1) = 2.0;
	//-- Calculate weights
	int idxKnn[NUM_NN+2];
	double distKnn[NUM_NN+2];
	double distSum, distMax, denom;
	weights = MatrixXd::Zero(nLen,nLen);
	for (int j=0; j<nLen; j++) {
		KNN(vert,vert.row(j),NUM_NN+2,idxKnn,distKnn);
		/* DEBUG PRINT *
		sprintf(sInfo,"%2d: %2d %2d %2d %2d %2d %2d %f %f %f %f %f %f\n",j,
			idxKnn[0],idxKnn[1],idxKnn[2],idxKnn[3],idxKnn[4],idxKnn[5],
			distKnn[0],distKnn[1],distKnn[2],distKnn[3],distKnn[4],distKnn[5]);
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
		distSum = 0;
		distMax = distKnn[NUM_NN+1];
		for (int i=1; i<NUM_NN+1; i++)
			distSum += distKnn[i];
		denom = (double)NUM_NN - distSum / distMax;
		for (int i=1; i<NUM_NN+1; i++)
			weights(idxKnn[i],j) = (1 - distKnn[i]/distMax) / denom;
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
// Return the indices and distances of the k nearest neighbors in the array
// If pt is in dset, it is the closest point & sort(dist)[0] = 0.0
//
int nicp3d::KNN (const MatrixX3d& dataset, const RowVector3d& pt, int k, int* idx, double* dist) {
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
	for (int i=0; i<dataset.rows(); i++) {
		//TNT::Array1D<double> p(dataset.dim2(),dataset[i]);
		//d = Distance(p,pt);
		d = Distance(dataset.row(i),pt);
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
	return size;
}

//
// Return the index of the closest point in the array
//
int nicp3d::Closest (const MatrixX3d& dataset, const RowVector3d& pt, double* minDist) {
	//-- Initialise with first point in dataset 
	double min = Distance(dataset.row(0),pt);
	int idx = 0;
	//-- Compare with the other points
	double dist;
	int nLen = dataset.rows();
	for (int i=1; i<nLen; i++) {
		//p = TNT::Array1D<double> (dataset.dim2(),dataset[i]);
		//dist = Distance(p,pt);
		dist = Distance(dataset.row(i),pt);
		if (dist<min) {
			min = dist;
			idx = i;
		}
	}
	//-- Return minimum distance if required
	if (minDist)
		*minDist = min;
	return idx;
}

//
// For each point in source, find the closest point in the target. 
// Return the closest points in array
//
void nicp3d::Match (const MatrixX3d& source, const MatrixX3d& target, MatrixX3d& matchset) {
	int nLen = source.rows();
	int idx;
	char sInfo[100];
	matchset = MatrixX3d (nLen,3);
	for (int i=0; i<nLen; i++) {
		idx = Closest(target,source.row(i));
		matchset.row(i) = target.row(idx);
		/* DEBUG PRINT *
		sprintf(sInfo,"%f %f %f",matchset(i,0),matchset(i,1),matchset(i,2));
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}
}

//
// Deform the mesh using parameters
//
void nicp3d::Deform (const MatrixX3d& dataset, MatrixX3d& deformset) {
	char sInfo[5000];

	int nLen = dataset.rows();
	deformset = MatrixXd::Zero(nLen,3);

	Matrix3d A;
	RowVector3d b;
	RowVector3d tmp;

	for (int i=0; i<nLen; i++) {
		//deformset[i][0] = deformset[i][1] = deformset[i][2] = 0.0;
		for (int j=0; j<nLen; j++) {
			if (weights(j,i)!=0) {
				PtoAB(j,A,b);
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
					sprintf(sInfo,"%d(%f,%f,%f) %d(%f,%f,%f)",i,dataset(i,0),dataset(i,1),dataset(i,2),j,dataset(j,0),dataset(j,1),dataset(j,2));
					MGlobal::displayInfo(sInfo);
					tmp = dataset.row(i) - dataset.row(j);
					sprintf(sInfo,"(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					sprintf(sInfo,"A=(%f,%f,%f %f,%f,%f %f,%f,%f)",
						A(0,0),A(0,1),A(0,2),A(1,0),A(1,1),A(1,2),A(2,0),A(2,1),A(2,2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp * A;
					sprintf(sInfo,"(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp + dataset.row(j);
					sprintf(sInfo,"(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp + b;
					sprintf(sInfo,"(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					tmp = tmp * weights(j,i);
					sprintf(sInfo,"(%f,%f,%f)",tmp(0),tmp(1),tmp(2));
					MGlobal::displayInfo(sInfo);
					deformset(i,0) += tmp(0);
					deformset(i,1) += tmp(1);
					deformset(i,2) += tmp(2);
				} else {
				* DEBUG PRINT */
					tmp = dataset.row(i) - dataset.row(j);
					tmp = tmp * A;
					tmp = tmp + dataset.row(j);
					tmp = tmp + b;
					tmp = tmp * weights(j,i);
					deformset(i,0) += tmp(0);
					deformset(i,1) += tmp(1);
					deformset(i,2) += tmp(2);
				//}
			}
		}
		/* DEBUG PRINT *
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,dataset(i,0),dataset(i,1),dataset(i,2),deformset(i,0),deformset(i,1),deformset(i,2));
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}	
	//-- Global transformation
	int nParam = params.rows();
	Matrix3d Rot;
	RotMatrix(params[nParam-6],params[nParam-5],params[nParam-4],Rot);
	RowVector3d Txn = params.segment<3>(nParam-3);
	MatrixX3d txnset(nLen,3);
	txnset.col(0) = VectorXd::Constant(nLen,Txn(0));
	txnset.col(1) = VectorXd::Constant(nLen,Txn(1));
	txnset.col(2) = VectorXd::Constant(nLen,Txn(2));

	//--
	//-- Printing content of Rot will reset it to Zero. ????????
	//--
	/* DEBUG PRRINT *
	sprintf(sInfo,"R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot(0,0),Rot(0,1),Rot(0,2), Rot(1,0),Rot(1,1),Rot(1,2), Rot(2,0),Rot(2,1),Rot(2,2),
		Txn(0),Txn(1),Txn(2));
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */
	
	deformset = deformset * Rot;
	deformset = deformset + txnset;
	/* DEBUG PRINT *
	for (int i=0; i<nLen; i++) {
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,dataset(i,0),dataset(i,1),dataset(i,2),deformset(i,0),deformset(i,1),deformset(i,2));
		MGlobal::displayInfo(sInfo);
	}
	* DEBUG PRINT */
}

//
// Extract A & b from P for a vertex
//
void nicp3d::PtoAB (int i, Matrix3d& A, RowVector3d& b) {
	A.row(0) = params.segment<3>(i*12);
	A.row(1) = params.segment<3>(i*12+3);
	A.row(2) = params.segment<3>(i*12+6);
	b = params.segment<3>(i*12+9);
}

//
// Rotational matrix - rotate sequence = X, Y, Z
//
void nicp3d::RotMatrix (double x, double y, double z, Matrix3d& rot) {
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	rot << cy*cz, cy*sz, -sy, 
		sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, 
		sx*cy, cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy; 
}

//
// Update vertices of the mesh using the dataset 
//
void nicp3d::ModifyVertices (MDagPath path, const MatrixX3d& dataset) {
	int nLen = dataset.rows();
	MPointArray pts(nLen);
	for (int i=0; i<nLen; i++)
		pts.set(i,dataset(i,0),dataset(i,1),dataset(i,2));
	MFnMesh mesh(path);
	mesh.setPoints(pts,MSpace::kWorld);
	return;
}

//
// Partial differentiation of the Rotational matrix - rotate sequence = X, Y, Z
//
void nicp3d::D_RotMatrix (double x, double y, double z, Matrix3d& drotx, Matrix3d& droty, Matrix3d& drotz) {	
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	//-- Partial differentiation wrt theta-X
	drotx << 0, 0, 0,
		cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy,
		-sx*sy*cz+cx*sz, -sx*sy*sz-cx*cz, -sx*cy;
	//-- Partial differentiation wrt theta-Y
	droty << -sy*cz, -sy*sz, -cy,
		sx*cy*cz, sx*cy*sz, -sx*sy,
		cx*cy*cz, cx*cy*sz, -cx*sy;
	//-- Partial differentiation wrt theta-Z
	drotz << -cy*sz, cy*cz, 0,
		-sx*sy*sz-cx*cz, sx*sy*cz-cx*sz, 0
		-cx*sy*sz+sx*cz, cx*sy*cz+sx*sz, 0;
}


MStatus initializePlugin(MObject obj) {
	MFnPlugin plugin(obj, "Registration", "3.0", "Any");
	MStatus status = plugin.registerCommand("nicp3d", nicp3d::creator);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	return status;
}
 
MStatus uninitializePlugin(MObject obj) {
	MFnPlugin plugin(obj);
	MStatus status = plugin.deregisterCommand("nicp3d");
	CHECK_MSTATUS_AND_RETURN_IT(status);
	return status;
} 

