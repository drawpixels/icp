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
#include <unsupported/Eigen/NonLinearOptimization>
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
#include "Mesh.h"
#include "Deformable.h"
#include "nicp_lm_functor.h"

#define INIT_C_FIT 0.1
#define INIT_C_RIGID 100.0
#define INIT_C_SMOOTH 10.0

using namespace std;
using namespace Eigen;

class nicp3d : public MPxCommand {
public:
	nicp3d() {};
	virtual MStatus doIt (const MArgList& argList);
	static void* creator ();
private:
	VectorXd params;
	int GetModels (MDagPath& dag1, MDagPath& dag2);
	Mesh GetMesh (const MDagPath dag);
	VectorXd Initialise (const Mesh& m);
	void NICP (const Deformable& src, const Mesh& tgt, VectorXd params, double tol=0.001, int runs=100);
	void ModifyVertices (MDagPath path, const Mesh& m);
};

void* nicp3d::creator () { return new nicp3d; }
 
MStatus nicp3d::doIt (const MArgList& argList) {
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
	Deformable mSrc = GetMesh(dagSrc);
	Mesh mTgt = GetMesh(dagTgt);
	sprintf(sInfo,"Source mesh has %d vertics, %d edges",mSrc.NumVertices(),mSrc.NumEdges());
	MGlobal::displayInfo(sInfo);
	sprintf(sInfo,"Target mesh has %d vertics, %d edges",mTgt.NumVertices(),mTgt.NumEdges());
	MGlobal::displayInfo(sInfo);

	VectorXd params = Initialise (mSrc);
	NICP(mSrc,mTgt,params,0.01,10);

	Mesh DD = mSrc.Deform(params);
	ModifyVertices(dagSrc,DD);
	
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
Mesh nicp3d::GetMesh (const MDagPath dag) {
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
	//-- Get edges
	int nEdg = mesh.numEdges();
	MatrixX2i e(nEdg,2);
	int2 v2;
	for (int i=0; i<nEdg; i++) {
		mesh.getEdgeVertices(i,v2);
		e(i,0) = v2[0];
		e(i,1) = v2[1];
	}
	//-- Return mesh
	return Mesh(v,e);
}

//
// Intialise parameters and weights
//
VectorXd nicp3d::Initialise (const Mesh& m) {
	int nLen = m.NumVertices();
	//--- Initialise parameters: A = Identity, b = zeros, Rxyz = 0, Txyz = 0
	VectorXd p = VectorXd::Zero(nLen*12+6);
	for (int i=0; i<nLen; i++) {
		int ii = i*12;
		p(ii) = 1.0;
		p(ii+4) = 1.0;
		p(ii+8) = 1.0;
	}
	//int n = p.rows();
	//p(n-1) = 5.0;
	return p;
}

void nicp3d::NICP (const Deformable& src, const Mesh& tgt, VectorXd params, double tol, int runs) {
	char sInfo[500];
	double e, err = 1000000.0;
	for (int i=0; i<runs; i++) {
		e = err;
		Mesh DD = src.Deform(params);
		Mesh NN = DD.Match(tgt);
		nicp_lm_functor functor(src,NN,INIT_C_FIT,INIT_C_RIGID,INIT_C_SMOOTH);
		LevenbergMarquardt<nicp_lm_functor> lm(functor);
		int info = lm.lmder1(params);
		// End if fail to optimise 
		if (info!=1)
			break;
		err = lm.fvec.norm();
		sprintf(sInfo,"%3d,%f",i,err);
		MGlobal::displayInfo(sInfo);
		/* DEBUG PRINT */
		sprintf(sInfo,"info=%d,iter=%ld,nfev=%ld,njev=%ld fvec=%f %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
			info,lm.iter,lm.nfev,lm.njev,lm.fvec.norm(),params(0),params(1),params(2),params(3),params(4),params(5),
			params(6),params(7),params(8),params(9),params(10),params(11));
		MGlobal::displayInfo(sInfo);
		/* DEBUG PRINT */
	}
}


//
// Update vertices of the mesh using the dataset 
//
void nicp3d::ModifyVertices (MDagPath path, const Mesh& m) {
	int nLen = m.NumVertices();
	MPointArray pts(nLen);
	for (int i=0; i<nLen; i++)
		pts.set(i,m.Vertex(i)(0),m.Vertex(i)(1),m.Vertex(i)(2));
	MFnMesh mesh(path);
	mesh.setPoints(pts,MSpace::kWorld);
	return;
}

MStatus initializePlugin (MObject obj) {
	MFnPlugin plugin(obj, "Registration", "3.0", "Any");
	MStatus status = plugin.registerCommand("nicp3d", nicp3d::creator);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	return status;
}
 
MStatus uninitializePlugin (MObject obj) {
	MFnPlugin plugin(obj);
	MStatus status = plugin.deregisterCommand("nicp3d");
	CHECK_MSTATUS_AND_RETURN_IT(status);
	return status;
} 

