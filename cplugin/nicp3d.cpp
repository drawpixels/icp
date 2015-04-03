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

#define INIT_C_FIT 5.0
#define INIT_C_RIGID 50.0
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
	int GetModels (MDagPath& dag1, MDagPath& dag2, MDagPath& dag3);
	Mesh GetMesh (const MDagPath dag);
	VectorXd Initialise (const Mesh& m);
	Mesh NICP (const Deformable& src, const Mesh& tgt, VectorXd& params, double tol=0.001, int runs=100);
	void ModifyVertices (MDagPath path, const Mesh& m);
};

void* nicp3d::creator () { return new nicp3d; }
 
MStatus nicp3d::doIt (const MArgList& argList) {
	MDagPath dagSrc, dagS2, dagTgt;
	char sInfo[500];
	int n = GetModels(dagSrc,dagS2,dagTgt);
	if (n!=3) {
		MGlobal::displayInfo("3 models must be selected.");
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
	Mesh NN = NICP(mSrc,mTgt,params,0.01,20);
	/* DEBUG PRINT */
	int i,j;
	for (i=0,j=0; i<params.rows()/12; i++,j+=12) {
		sprintf(sInfo, "%4d [%f,%f,%f %f,%f,%f %f,%f,%f] [%f,%f,%f]", 
			i,params(j),params(j+1),params(j+2),params(j+3),params(j+4),params(j+5),
			params(j+6),params(j+7),params(j+8),params(j+9),params(j+10),params(j+11));
		MGlobal::displayInfo(sInfo);
	}
	sprintf(sInfo,"G (%f %f %f) (%f %f %f)",params(j),params(j+1),params(j+2),params(j+3),params(j+4),params(j+5));
	MGlobal::displayInfo(sInfo);
	/* DEBUG PRINT */
	Mesh DD = mSrc.Deform(params);
	ModifyVertices(dagSrc,DD);
	ModifyVertices(dagS2,NN);
	
	return MS::kSuccess;
}

//
// Return the selected models - must select TWO models SRC & TGT
//
int nicp3d::GetModels (MDagPath& dag0, MDagPath& dag1, MDagPath& dag2) {
	MSelectionList list;
	MGlobal::getActiveSelectionList(list);
	if (list.length()>=3) {
		list.getDagPath(0,dag0);
		list.getDagPath(1,dag1);
		list.getDagPath(2,dag2);
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

Mesh nicp3d::NICP (const Deformable& src, const Mesh& tgt, VectorXd& params, double tol, int runs) {
	char sInfo[500];
	int n = params.rows();
	double e, err = 1000000.0;
	double c_fit = INIT_C_FIT;
	double c_rigid = INIT_C_RIGID;
	double c_smooth = INIT_C_SMOOTH;
	Mesh DD, NN;
	int info;
	for (int r=0; r<runs; r++) {
		e = err;
		DD = src.Deform(params);
		NN = DD.Match(tgt);
		nicp_lm_functor functor(src,NN,c_fit,c_rigid,c_smooth);
		LevenbergMarquardt<nicp_lm_functor> lm(functor);
		info = lm.lmder1(params);
		err = lm.fvec.norm();
		//DenseIndex nfev;
		//VectorXd fvec(functor.values());
		//info = LevenbergMarquardt<nicp_lm_functor>::lmdif1(functor,params,&nfev);
		//functor(params,fvec);
		//err = fvec.norm();
		sprintf(sInfo,"%3d: %f [%f,%f,%f]",r,err,c_fit,c_rigid,c_smooth);
		MGlobal::displayInfo(sInfo);
		//sprintf(sInfo,"[%f,%f,%f] [%f,%f,%f]", params(n-6),params(n-5),params(n-4),params(n-3),params(n-2),params(n-1));
		//MGlobal::displayInfo(sInfo);
		/* DEBUG PRINT *
		sprintf(sInfo,"info=%d,iter=%ld,nfev=%ld,njev=%ld fvec=%f",
			info,lm.iter,lm.nfev,lm.njev,lm.fvec.norm());
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
		if ((e-err)<tol) {
			c_rigid /= 2.0;
			c_smooth /= 2.0;
			if ((c_rigid<0.1)||(c_smooth<0.1))
				break;
		}
	}
	/* DEBUG PRINT *
	int i,j;
	for (i=0,j=0; i<params.rows()/10; i++,j+=10) {
		sprintf(sInfo, "%4d %f,%f,%f,%f,%f,%f,%f,%f,%f,%f", 
			i,params(j),params(j+1),params(j+2),params(j+3),params(j+4),
			params(j+5),params(j+6),params(j+7),params(j+8),params(j+9));
		MGlobal::displayInfo(sInfo);
	}
	sprintf(sInfo,"%4d ",i);
	for ( ; j<params.rows(); j++) 
		sprintf(sInfo,"%s %f",sInfo,params(j));
	MGlobal::displayInfo(sInfo);
	* DEBUG PRINT */
	return NN;
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

