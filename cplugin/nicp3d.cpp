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
	Mesh GetMesh (const MDagPath dag);
	VectorXd Initialise (const Mesh& m);
	void ModifyVertices (MDagPath path, const Mesh& m);
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

	//Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	//Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct nicp_lm_functor : Functor<double>
{
	Deformable mSource;
	Mesh mTarget;
	double c_fit, c_rigid, c_smooth;
	
	nicp_lm_functor(Mesh& s, Mesh& t) : mSource(s), mTarget(t) {
		m_inputs = mSource.NumVertices()*12 + 6;
		m_values = mSource.NumVertices()*9 + mSource.NumEdges()*6;
		c_fit = INIT_C_FIT;
		c_rigid = INIT_C_RIGID;
		c_smooth = INIT_C_SMOOTH;
	}
	
	void SetCoeff(double f, double r, double s) {
		c_fit = f;
		c_rigid = r;
		c_smooth = s;
	}
	
	// x = (in) parameters, fvec = (out) output values using the parameters
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
		int nVertices = mSource.NumVertices();
		int nEdges = mSource.NumEdges();
		int idx=0;	//-- Running index to fill up the output values
		//-- E-fit
		Mesh DD = mSource.Deform(x);
		for (int i=0; i<nVertices; i++) {
			Vector3d delta = mTarget.Vertex(i) - DD.Vertex(i);
			fvec(idx++) = delta(0) * c_fit;
			fvec(idx++) = delta(1) * c_fit;
			fvec(idx++) = delta(2) * c_fit;
		}
		//-- E-rigid
		Matrix3d A;
		RowVector3d b,a0,a1,a2;
		for (int i; i<nVertices; i++) {
			Deformable::PtoAB(x,i,A,b);
			a0 = A.row(0);
			a1 = A.row(1);
			a2 = A.row(2);
			fvec(idx++) = a0.dot(a1) * c_rigid;
			fvec(idx++) = a1.dot(a2) * c_rigid;
			fvec(idx++) = a2.dot(a0) * c_rigid;
			fvec(idx++) = 1.0 - a0.dot(a0) * c_rigid;
			fvec(idx++) = 1.0 - a1.dot(a1) * c_rigid;
			fvec(idx++) = 1.0 - a2.dot(a2) * c_rigid;
		}
		//-- E-smooth
		RowVector2i e;
		int i0, i1;
		RowVector3d v0, v1;
		Matrix3d A0, A1;
		RowVector3d b0, b1, s0, s1;
		for (int i=0; i<nEdges; i++) {
			e = mSource.Edge(i);
			i0 = e(0);
			i1 = e(1);
			v0 = mSource.Vertex(i0);
			v1 = mSource.Vertex(i1);
			Deformable::PtoAB(x,i0,A0,b0);
			Deformable::PtoAB(x,i1,A1,b1);
			s0 = (v1-v0)*A0 + (v0+b0) - (v1+b1);
			s1 = (v0-v1)*A1 + (v1+b1) - (v0+b0);
			fvec(idx++) = s0(0) * c_smooth;
			fvec(idx++) = s0(1) * c_smooth;
			fvec(idx++) = s0(2) * c_smooth;
			fvec(idx++) = s1(0) * c_smooth;
			fvec(idx++) = s1(1) * c_smooth;
			fvec(idx++) = s1(2) * c_smooth;
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
	Mesh DD = mSrc.Deform(params);
	Mesh NN = DD.Match(mTgt);
	
	int info;
	DenseIndex nfev=0;
	nicp_lm_functor functor(mSrc,NN);
	info = LevenbergMarquardt<nicp_lm_functor>::lmdif1(functor,params,&nfev);
	//NumericalDiff<nicp_lm_functor> numDiff(functor);
	//LevenbergMarquardt<NumericalDiff<nicp_lm_functor> > lm(numDiff);
	//info = lm.minimize(params);
	sprintf(sInfo,"info=%d,DIdx=%ld %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
		info,nfev,params(0),params(1),params(2),params(3),params(4),params(5),
		params(6),params(7),params(8),params(9),params(10),params(11));
	MGlobal::displayInfo(sInfo);
	
	DD = mSrc.Deform(params);
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

