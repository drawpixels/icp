//
// Non-rigid Iterative Closest Point 
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <queue>
#include <vector>
#include <tnt_array1d.h>
#include <tnt_array1d_utils.h>
#include <tnt_array2d.h>
#include <tnt_array2d_utils.h>
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

class nicp3d : public MPxCommand {
public:
	nicp3d() {};
	virtual MStatus doIt(const MArgList& argList);
	static void* creator();
private:
	TNT::Array1D<double> params;
	TNT::Array2D<double> weights;
	int KNN (TNT::Array2D<double> dataset, double* pt, int k, int* idx, double* dist);
	int Closest (TNT::Array2D<double> dataset, double* pt, double* minDist=NULL);
	int GetModels (MDagPath& dag1, MDagPath& dag2);
	void GetVerticesEdges (MDagPath dag, TNT::Array2D<double>& vert, TNT::Array2D<int>& edg);
	void Initialise (TNT::Array2D<double> vert);
	void Match (TNT::Array2D<double> source, TNT::Array2D<double> target, TNT::Array2D<double>& matchset);
	void Deform (TNT::Array2D<double> dataset, TNT::Array2D<double> deformset);
	void PtoAB (int i, TNT::Array2D<double>& A, TNT::Array1D<double>& b);
	void RotMatrix (double x, double y, double z, TNT::Array2D<double>& rot);
	void D_RotMatrix (double x, double y, double z, TNT::Array2D<double>& drotx, TNT::Array2D<double>& droty, TNT::Array2D<double>& drotz);
	void ModifyVertices (MDagPath path, TNT::Array2D<double> dataset);
};

//
// Return the Euclidean distance between 2 points
//
double Distance (double* p1, double* p2) {
	double disp, sumDispSq = 0.0;
	for (int i=0; i<3; i++) {
		disp = p1[i] - p2[i];
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
	TNT::Array2D<double> ptsSrc, ptsTgt;
	TNT::Array2D<int> edgSrc, edgTgt;
	GetVerticesEdges(dagSrc,ptsSrc,edgSrc);
	GetVerticesEdges(dagTgt,ptsTgt,edgTgt);
	sprintf(sInfo,"Source mesh has %d vertics, %d edges",ptsSrc.dim1(),edgSrc.dim1());
	MGlobal::displayInfo(sInfo);
	sprintf(sInfo,"Target mesh has %d vertics, %d edges",ptsTgt.dim1(),edgTgt.dim1());
	MGlobal::displayInfo(sInfo);
	Initialise(ptsSrc);
	TNT::Array2D<double> ptsNN;
	Match(ptsSrc,ptsTgt,ptsNN);
	TNT::Array2D<double> ptsDD (ptsSrc.dim1(),3);
	Deform(ptsSrc,ptsDD);
	return MS::kSuccess;
}

//
// Return the indices and distances of the k nearest neighbors in the array
// If pt is in dset, it is the closest point & sort(dist)[0] = 0.0
//
int nicp3d::KNN (TNT::Array2D <double> dataset, double* pt, int k, int* idx, double* dist) {
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
	for (int i=0; i<dataset.dim1(); i++) {
		//TNT::Array1D<double> p(dataset.dim2(),dataset[i]);
		//d = Distance(p,pt);
		d = Distance(dataset[i],pt);
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
int nicp3d::Closest (TNT::Array2D<double> dataset, double* pt, double* minDist) {
	double dist;
	//-- Initialise with first point in dataset 
	//TNT::Array1D<double> p(dataset.dim2(),dataset[0]);
	//double min = Distance(p,pt);
	double min = Distance(dataset[0],pt);
	int idx = 0;
	//-- Compare with the other points
	int nLen = dataset.dim1();
	for (int i=1; i<nLen; i++) {
		//p = TNT::Array1D<double> (dataset.dim2(),dataset[i]);
		//dist = Distance(p,pt);
		dist = Distance(dataset[i],pt);
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
// Intialise parameters and weights
//
void nicp3d::Initialise (TNT::Array2D<double> vert) {
	char sInfo[512];
	int nLen = vert.dim1();
	//--- Initialise parameters: A = Identity, b = zeros, Rxyz = 0, Txyz = 0
	params = TNT::Array1D<double>(nLen*12+6,0.0);
	for (int i=0; i<nLen; i++) {
		int ii = i*12;
		params[ii] = 1.0;
		params[ii+4] = 1.0;
		params[ii+8] = 1.0;
	}
	//int nParams = params.dim1();
	//params[nParams-1] = 1.0;
	//params[nParams-2] = 2.0;
	//-- Calculate weights
	int idxKnn[NUM_NN+2];
	double distKnn[NUM_NN+2];
	double distSum, distMax, denom;
	weights = TNT::Array2D<double>(nLen,nLen,0.0);
	for (int j=0; j<nLen; j++) {
		KNN(vert,vert[j],NUM_NN+2,idxKnn,distKnn);
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
			weights[idxKnn[i]][j] = (1 - distKnn[i]/distMax) / denom;
	}
	/* DEBUG PRINT *
	for (int i=0; i<10; i++) {
		sprintf(sInfo,"%2d %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f", i,
		weights[i][0],weights[i][1],weights[i][2],weights[i][3],weights[i][4],weights[i][5],weights[i][6],weights[i][7],weights[i][8],weights[i][9],
		weights[i][10],weights[i][11],weights[i][12],weights[i][13],weights[i][14],weights[i][15],weights[i][16],weights[i][17],weights[i][18],weights[i][19]);
		MGlobal::displayInfo(sInfo);
	}
	* DEBUG PRINT */
}


//
// Return the vertices of the mesh in Numpy.Array
//
void nicp3d::GetVerticesEdges (MDagPath dag, TNT::Array2D<double>& vert, TNT::Array2D<int>& edg) {
	MFnMesh mesh(dag);
	//-- Get vertices
	MPointArray pts;
	mesh.getPoints(pts,MSpace::kWorld);
	int nVert = pts.length();
	TNT::Array2D<double> v(nVert,3);
	for (int i=0; i<nVert; i++) {
		v[i][0] = pts[i].x;
		v[i][1] = pts[i].y;
		v[i][2] = pts[i].z;
	}
	vert = v;
	//-- Get edges
	int nEdg = mesh.numEdges();
	TNT::Array2D<int> e(nEdg,2);
	int2 v2;
	for (int i=0; i<nEdg; i++) {
		mesh.getEdgeVertices(i,v2);
		e[i][0] = v2[0];
		e[i][1] = v2[1];
	}
	edg = e;
}

//
// For each point in source, find the closest point in the target. 
// Return the closest points in array
//
void nicp3d::Match (TNT::Array2D<double> source, TNT::Array2D<double> target, TNT::Array2D<double>& matchset) {
	int nLen = source.dim1();
	int idx;
	char sInfo[100];
	matchset = TNT::Array2D<double> (nLen,3);
	for (int i=0; i<nLen; i++) {
		idx = Closest(target,source[i]);
		memcpy(matchset[i],target[idx],sizeof(double)*3);
		/* DEBUG PRINT *
		sprintf(sInfo,"%f %f %f",matchset[i][0],matchset[i][1],matchset[i][2]);
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}
}

//
// Deform the mesh using parameters
//
void nicp3d::Deform (TNT::Array2D<double> dataset, TNT::Array2D<double> deformset) {
	char sInfo[5000];

	//-- Get global transformation matrices
	int nParam = params.dim1();
	TNT::Array2D<double> Rot;
	RotMatrix(params[nParam-6],params[nParam-5],params[nParam-4],Rot);
	TNT::Array1D<double> Txn(3,params.data()+nParam-3);
	/* DEBUG PRRINT *
	sprintf(sInfo,"a:R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot[0][0],Rot[0][1],Rot[0][2],
		Rot[1][0],Rot[1][1],Rot[1][2],
		Rot[2][0],Rot[2][1],Rot[2][2],
		Txn[0],Txn[1],Txn[2]);
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */

	/* DEBUG PRRINT *
	sprintf(sInfo,"b:R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot[0][0],Rot[0][1],Rot[0][2],
		Rot[1][0],Rot[1][1],Rot[1][2],
		Rot[2][0],Rot[2][1],Rot[2][2],
		Txn[0],Txn[1],Txn[2]);
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */

	int nLen = dataset.dim1();

	/* DEBUG PRRINT *
	sprintf(sInfo,"R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot[0][0],Rot[0][1],Rot[0][2],
		Rot[1][0],Rot[1][1],Rot[1][2],
		Rot[2][0],Rot[2][1],Rot[2][2],
		Txn[0],Txn[1],Txn[2]);
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */

	TNT::Array2D<double> A;
	TNT::Array1D<double> b;
	TNT::Array1D<double> tmp;
	/* DEBUG PRRINT *
	sprintf(sInfo,"R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot[0][0],Rot[0][1],Rot[0][2],
		Rot[1][0],Rot[1][1],Rot[1][2],
		Rot[2][0],Rot[2][1],Rot[2][2],
		Txn[0],Txn[1],Txn[2]);
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */
	for (int i=0; i<nLen; i++) {
		/* DEBUG PRRINT */
		sprintf(sInfo,"%2d R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",i,
			Rot[0][0],Rot[0][1],Rot[0][2],
			Rot[1][0],Rot[1][1],Rot[1][2],
			Rot[2][0],Rot[2][1],Rot[2][2],
			Txn[0],Txn[1],Txn[2]);
		MGlobal::displayInfo(sInfo);
		/* DEBUG PRINT */
		deformset[i][0] = deformset[i][1] = deformset[i][2] = 0.0;
		for (int j=0; j<nLen; j++) {
			if (weights[j][i]!=0) {
				PtoAB(j,A,b);
				/* DEBUG PRINT *
				sprintf(sInfo,"(%f,%f,%f %f,%f,%f %f,%f,%f) (%f,%f,%f)",
					A[0][0],A[0][1],A[0][2],
					A[1][0],A[1][1],A[1][2],
					A[2][0],A[2][1],A[2][2],
					b[0],b[1],b[2]);
				MGlobal::displayInfo(sInfo);
				* DEBUG PRINT */
				/* DEBUG PRINT *
				if ((A[0][0]!=1.0)||(A[0][1]!=0.0)||(A[0][2]!=0.0)||
					(A[1][0]!=0.0)||(A[1][1]!=1.0)||(A[1][2]!=0.0)||
					(A[2][0]!=0.0)||(A[2][1]!=0.0)||(A[2][2]!=1.0)||
					(b[0]!=0.0)||(b[1]!=0.0)||(b[2]!=0.0)) {
						sprintf(sInfo,"Wrong A,B at %d",j);
						MGlobal::displayInfo(sInfo);
				}
				* DEBUG PRINT */
				/* DEBUG PRINT *
				if (i<5) {
					sprintf(sInfo,"%d(%f,%f,%f) %d(%f,%f,%f)",i,dataset[i][0],dataset[i][1],dataset[i][2],j,dataset[j][0],dataset[j][1],dataset[j][2]);
					MGlobal::displayInfo(sInfo);
					tmp = TNT::Array1D<double>(3,dataset[i]) - TNT::Array1D<double>(3,dataset[j]);
					sprintf(sInfo,"(%f,%f,%f)",tmp[0],tmp[1],tmp[2]);
					MGlobal::displayInfo(sInfo);
					sprintf(sInfo,"A=(%f,%f,%f)",A[0][0],A[0][1],A[0][2]);
					MGlobal::displayInfo(sInfo);
					sprintf(sInfo,"  (%f,%f,%f)",A[1][0],A[1][1],A[1][2]);
					MGlobal::displayInfo(sInfo);
					sprintf(sInfo,"  (%f,%f,%f)",A[2][0],A[2][1],A[2][2]);
					MGlobal::displayInfo(sInfo);
					tmp = TNT::matmult(tmp,A);
					sprintf(sInfo,"(%f,%f,%f)",tmp[0],tmp[1],tmp[2]);
					MGlobal::displayInfo(sInfo);
					tmp += TNT::Array1D<double>(3,dataset[j]);
					sprintf(sInfo,"(%f,%f,%f)",tmp[0],tmp[1],tmp[2]);
					MGlobal::displayInfo(sInfo);
					tmp += b;
					sprintf(sInfo,"(%f,%f,%f)",tmp[0],tmp[1],tmp[2]);
					MGlobal::displayInfo(sInfo);
					tmp *= weights[j][i];
					sprintf(sInfo,"(%f,%f,%f)",tmp[0],tmp[1],tmp[2]);
					MGlobal::displayInfo(sInfo);
					deformset[i][0] += tmp[0];
					deformset[i][1] += tmp[1];
					deformset[i][2] += tmp[2];
				} else {
				* DEBUG PRINT */
					tmp = TNT::Array1D<double>(3,dataset[i]) - TNT::Array1D<double>(3,dataset[j]);
					tmp = TNT::matmult(tmp,A);
					tmp += TNT::Array1D<double>(3,dataset[j]);
					tmp += b;
					tmp *= weights[j][i];
					deformset[i][0] += tmp[0];
					deformset[i][1] += tmp[1];
					deformset[i][2] += tmp[2];
					//}
			}
		}
		/* DEBUG PRINT *
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,dataset[i][0],dataset[i][1],dataset[i][2],deformset[i][0],deformset[i][1],deformset[i][2]);
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
	}	
	/* DEBUG PRRINT *
	sprintf(sInfo,"R=(%f,%f,%f %f,%f,%f %f,%f,%f) T=(%f,%f,%f)",
		Rot[0][0],Rot[0][1],Rot[0][2],
		Rot[1][0],Rot[1][1],Rot[1][2],
		Rot[2][0],Rot[2][1],Rot[2][2],
		Txn[0],Txn[1],Txn[2]);
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */
	
	//-- Global transformation
	deformset = TNT::matmult(deformset,Rot,sInfo);
	MGlobal::displayInfo(sInfo);
	//deformset = TNT::addrow(deformset,Txn);
	//deformset = TNT::addrow(TNT::matmult(deformset,Rot),Txn);
	/* DEBUG PRINT */
	for (int i=0; i<nLen; i++) {
		sprintf(sInfo,"%d (%f,%f,%f) (%f,%f,%f)",i,dataset[i][0],dataset[i][1],dataset[i][2],deformset[i][0],deformset[i][1],deformset[i][2]);
		MGlobal::displayInfo(sInfo);
	}
	/* DEBUG PRINT */
}

//
// Extract A & b from P for a vertex
//
void nicp3d::PtoAB (int i, TNT::Array2D<double>& A, TNT::Array1D<double>& b) {
	//A = TNT::Array2D<double> (3,3,params[i*12]);
	//b = TNT::Array1D<double> (3,params[i*12+9]);
	A = TNT::Array2D<double> (3,3,params.data()+i*12);
	b = TNT::Array1D<double> (3,params.data()+i*12+9);
}

//
// Rotational matrix - rotate sequence = X, Y, Z
//
void nicp3d::RotMatrix (double x, double y, double z, TNT::Array2D<double>& rot) {
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	double rot_elem[] = {
		cy*cz, cy*sz, -sy, 
		sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, sx*cy,
		cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy 
	};
	rot = TNT::Array2D<double>(3,3,rot_elem);
}

//
// Rotational matrix - rotate sequence = X, Y, Z
//
void nicp3d::D_RotMatrix (double x, double y, double z, TNT::Array2D<double>& drotx, TNT::Array2D<double>& droty, TNT::Array2D<double>& drotz) {	
	double sx = sin(x);
	double cx = cos(x);
	double sy = sin(y);
	double cy = cos(y);
	double sz = sin(z);
	double cz = cos(z);
	double drotx_elem[] = {
		0, 0, 0, 
		cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy, 
		-sx*sy*cz+cx*sz, -sx*sy*sz-cx*cz, -sx*cy
	};
	drotx = TNT::Array2D<double>(3,3,drotx_elem);
	double droty_elem[] = {
		-sy*cz, -sy*sz, -cy, 
		sx*cy*cz, sx*cy*sz, -sx*sy, 
		cx*cy*cz, cx*cy*sz, -cx*sy
	};
	droty = TNT::Array2D<double>(3,3,droty_elem);
	double drotz_elem[] = {
		-cy*sz, cy*cz, 0, 
		-sx*sy*sz-cx*cz, sx*sy*cz-cx*sz, 0, 
		-cx*sy*sz+sx*cz, cx*sy*cz+sx*sz, 0
	};
	drotz = TNT::Array2D<double>(3,3,drotz_elem);
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

