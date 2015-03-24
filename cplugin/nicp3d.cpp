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
	int GetModels (MDagPath& dag1, MDagPath& dag2);
	Mesh GetMesh (MDagPath dag);
	VectorXd Initialise (const Mesh m);
	void ModifyVertices (MDagPath path, const MatrixX3d& dataset);
};

// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct nicp_lm_functor : Functor<double>
{
	MatrixX3d& ptsSrc;
	MatrixX3d& ptsTgt;
	MatrixX2i& edgSrc;
	MatrixXd& weights;
	int srcLen;
	int edgeLen;
	
	int setup(MatrixX3d& s, MatrixX3d& t, MatrixX2i& e, MatrixXd& w) {
		int slen = s.rows();
		int tlen = t.rows();
		int elen = e.rows();
		int wrows = w.rows();
		int wcols = w.cols();
		//-- Sizes of source, target and weights must match.
		if ((slen!=tlen)||(slen!=wrows)||(slen!=wcols))
			return 0;
		//-- Size of source, parameters and outputs must match.
		if ((m_inputs!=(slen*12+6))||(m_values!=(slen*12+elen*6)))
			return 0;
		ptsSrc = s;
		ptsTgt = t;
		edgSrc = e;
		weights = w;
		srcLen = slen;
		edgeLen = elen;
		return 1;
	}
	
	// x = (in) parameters, fvec = (out) output values using the parameters
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
		//-- E-fit
		MatrixX3d deform;
		for (int i=0; i<srcLen; i++) {
			
		}
        return 0;
    }

	// x = (in) parameters, fjac = (out) jacobian (15x3 matrix)
    int df(const VectorXd &x, MatrixXd &fjac) const
    {

        return 0;
    }
};

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
	Deformable mSrc = GetMesh(dagSrc);
	Mesh mTgt = GetMesh(dagTgt);
	sprintf(sInfo,"Source mesh has %ld vertics, %ld edges",mSrc.Vertices().rows(),mSrc.Edges().rows());
	MGlobal::displayInfo(sInfo);
	sprintf(sInfo,"Target mesh has %ld vertics, %ld edges",mTgt.Vertices().rows(),mTgt.Edges().rows());
	MGlobal::displayInfo(sInfo);
	VectorXd params = Initialise (mSrc);
//	MatrixX3d ptsNN = mSrc.Match(mTgt);
//	MatrixX3d ptsDD;
//	mSrc.Deform(params,ptsDD);
	Mesh NN = mSrc.Match(mTgt);
	Mesh DD = mSrc.Deform(params);
	
//	Initialise(ptsSrc);
//	MatrixX3d ptsNN;
//	Match(ptsSrc,ptsTgt,ptsNN);
//	MatrixX3d ptsDD;
//	Deform(ptsSrc,ptsDD);
//	ModifyVertices(dagSrc,ptsDD);
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
Mesh nicp3d::GetMesh (MDagPath dag) {
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
VectorXd nicp3d::Initialise (const Mesh m) {
	int nLen = m.NumVertices();
	//--- Initialise parameters: A = Identity, b = zeros, Rxyz = 0, Txyz = 0
	VectorXd p = VectorXd::Zero(nLen*12+6);
	for (int i=0; i<nLen; i++) {
		int ii = i*12;
		p(ii) = 1.0;
		p(ii+4) = 1.0;
		p(ii+8) = 1.0;
	}
	return p;
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

