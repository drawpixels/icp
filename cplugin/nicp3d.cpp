//
// Non-rigid Iterative Closest Point 
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>
#include <ctime>
#include <exception>
#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <ceres/ceres.h>
#include <maya/MGlobal.h>
#include <maya/MTypes.h>
#include <maya/MFnPlugin.h>
#include <maya/MArgList.h> 
#include <maya/MPxCommand.h>
#include <maya/MSelectionList.h>
#include <maya/MDagPath.h>
#include <maya/MFn.h>
#include <maya/MFnMesh.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include "Mesh.h"
#include "Deformable.h"
#include "NICP_costFunction.h"

#define DEFAULT_TOL 0.001
#define DEFAULT_RUN 20
#define DEFAULT_C_FIT 5.0
#define DEFAULT_C_RIGID 50.0
#define DEFAULT_C_SMOOTH 10.0

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
	Mesh NICP (Deformable& src, Mesh& tgt, VectorXd& params, double tol, int runs, double c_fit, double c_rigid, double c_smooth);
	void ModifyVertices (MDagPath path, const Mesh& m);
};

void* nicp3d::creator () { return new nicp3d; }
 
MStatus nicp3d::doIt (const MArgList& argList) {
	MStatus status;
	MDagPath dagSrc, dagS2, dagTgt;
	char sInfo[500];
	std::clock_t c_start,c_end;
	int run; 
	double c_fit, c_rigid, c_smooth;
	
	// Parse command arguments
	if (argList.length()>=1) {
		run = argList.asInt(0,&status);
		if (!status)
			run = DEFAULT_RUN;
	} else 
		run = DEFAULT_RUN;
	
	if (argList.length()>=2) {
		c_fit = argList.asDouble(1,&status);
		if (!status)
			c_fit = DEFAULT_C_FIT;
	} else 
		c_fit = DEFAULT_C_FIT;
	
	if (argList.length()>=3) {
		c_rigid = argList.asDouble(2,&status);
		if (!status)
			c_rigid = DEFAULT_C_RIGID;
	} else 
		c_rigid = DEFAULT_C_RIGID;
	
	if (argList.length()>=4) {
		c_smooth = argList.asDouble(3,&status);
		if (!status)
			c_smooth = DEFAULT_C_SMOOTH;
	} else 
		c_smooth = DEFAULT_C_SMOOTH;
		
	std::cout << "RUN = " << run << std::endl;
	std::cout << "C_FIT = " << c_fit << std::endl;
	std::cout << "C_RIGID = " << c_rigid << std::endl;
	std::cout << "C_SMOOTH = " << c_smooth << std::endl;
	
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

	try {
		c_start = std::clock();
		Mesh NN = NICP(mSrc,mTgt,params,0.01,run,c_fit,c_rigid,c_smooth);
		c_end = std::clock();
		std::cout << "Elapse time = " << (float)(c_end - c_start)/(CLOCKS_PER_SEC) << " sec." << std::endl; 
		/* DEBUG PRINT *
		int i,j;
		for (i=0,j=0; i<params.rows()/12; i++,j+=12) {
			sprintf(sInfo, "%4d [%8.4f,%8.4f,%8.4f   %8.4f,%8.4f,%8.4f   %8.4f,%8.4f,%8.4f]  (%8.4f,%8.4f,%8.4f)", 
				i,params(j),params(j+1),params(j+2),params(j+3),params(j+4),params(j+5),
				params(j+6),params(j+7),params(j+8),params(j+9),params(j+10),params(j+11));
			MGlobal::displayInfo(sInfo);
		}
		sprintf(sInfo,"G (%8.4f %8.4f %8.4f) (%8.4f %8.4f %8.4f)",params(j),params(j+1),params(j+2),params(j+3),params(j+4),params(j+5));
		MGlobal::displayInfo(sInfo);
		* DEBUG PRINT */
		Mesh DD = mSrc.Deform(params);
		ModifyVertices(dagSrc,DD);
		ModifyVertices(dagS2,NN);
	}
	catch (exception & e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}
	
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

Mesh nicp3d::NICP (Deformable& src, Mesh& tgt, VectorXd& params, double tol, int runs, double c_fit, double c_rigid, double c_smooth) {
	char sInfo[50000];
	int nPars = params.rows();
	int nVerts = src.NumVertices();
	double e=0, err;
	Mesh DD, NN;
	MatrixXd fjac;
	int info;
	int nfev=0;
	int njev=0;
	int iter=0;
	int r;
	MatrixXd fjac2;
	VectorXd pbackup=params;
	for (r=0; r<runs; r++) {
		DD = src.Deform(params);
		NN = DD.Match(tgt);
		int nVerts = src.NumVertices();
		vector<double *> param_blocks(nVerts+1);
		for (int i=0; i<nVerts+1; i++)
			param_blocks[i] = &params(i*12);
		ceres::Problem problem;
		EFit_CostFunction* efit_costFunction = new EFit_CostFunction(src,NN,c_fit);
		problem.AddResidualBlock(efit_costFunction, NULL, param_blocks);
		ERigid_CostFunction* erigid_costFunction = new ERigid_CostFunction(src,NN,c_rigid);
		problem.AddResidualBlock(erigid_costFunction, NULL, param_blocks);
		ESmooth_CostFunction* esmooth_costFunction = new ESmooth_CostFunction(src,NN,c_smooth);
		problem.AddResidualBlock(esmooth_costFunction, NULL, param_blocks);
		ceres::Solver::Options options;
		options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
		//options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
		//options.trust_region_strategy_type = ceres::DOGLEG;
		//options.dogleg_type = ceres::SUBSPACE_DOGLEG;
	    options.minimizer_progress_to_stdout = true;
	    ceres::Solver::Summary summary;
	    ceres::Solve(options, &problem, &summary);
		//std::cout << summary.BriefReport() << std::endl;
		std::cout << summary.FullReport() << std::endl;
		err = summary.final_cost;
		std::cout << "run=" << r << " err=" << err << " c=[" << c_fit << "," << c_rigid << "," << c_smooth << "] iter=" << iter << " nfev=" << nfev << " njev=" << njev << std::endl;
		if (e>0) {
			if (((e-err)/e)<tol) {
				c_rigid /= 2.0;
				c_smooth /= 2.0;
				if ((c_rigid<0.1)||(c_smooth<0.1))
					break;
			}
		}
		e = err;
	}
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

