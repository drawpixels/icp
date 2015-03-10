'''
nicp_3d.py

Non-rigid Iterative Closest Point algorithm in 3D.
Row-major implementation for Maya.
'''

import sys
import maya as m
import maya.api.OpenMaya as om
import numpy as np
from scipy import optimize

'''
Global Parameters
'''
num_nn = 4
c_fit = 0.1
c_rigid = 1000.0
c_smooth = 100.0

'''
Return the selected models - must select TWO models SRC & TGT
'''
def GetModel ():
	list = om.MGlobal.getActiveSelectionList()
	if (list.length()!=2):
		return None
	else:
		return list.getDagPath(0), list.getDagPath(1)

'''
Return the vertices of the mesh in Numpy.Array
'''
def GetVerticesEdges (path):
	mesh = om.MFnMesh(path)
	pts = mesh.getPoints(om.MSpace.kWorld)
	v = np.zeros([len(pts),3])
	for i in range(len(pts)):
		v[i,0] = pts[i].x
		v[i,1] = pts[i].y
		v[i,2] = pts[i].z
	n = mesh.numEdges
	e = np.zeros([n,2])
	for i in range(n):
		e[i] = mesh.getEdgeVertices(i)
	return (v,e)

'''
Return the indices and distances of the k nearest neighbors in the array
If pt is in dset, it is the closest point & sort(dist)[0] = 0.0
'''
def KNN (dset, pt, k):
	disp = dset - pt
	dist = np.sqrt((disp*disp).sum(1))
	sidx = np.argsort(dist)
	return sidx[0:k],np.sort(dist)[0:k]

'''
Return the index of the closest point in the array
'''
def Closest (dset, pt):
	disp = dset - pt
	dist = (disp*disp).sum(1)
	return np.argmin(dist)

'''
For each point in source, find the closest point in the target. 
Return the closest points in array
'''
def Match (source, target):
	nn = np.zeros(source.shape)
	for i in range(source.shape[0]):
		c = Closest(target,source[i])
		nn[i] = target[c]
		print nn[i]
	return nn

'''
Return the sum of distance between points in 2 datasets
Assumption: points in 2 datasets are corresponding
'''
def Distance (s1, s2):
	disp = s1 - s2
	dist = np.sqrt((disp*disp).sum(1))
	return dist.sum(0)

'''
Transform source to match target so that the sum sq error is minimum.
Return Transformation matrix
'''
def FindTransform (source, target):
	s_mean = source.mean(0)
	s1 = source - s_mean
	t_mean = target.mean(0)
	t1 = target - t_mean
	#- print "s_mean = ", s_mean
	#- print "t_mean = ", t_mean
	dim = source.shape[1]
	W = np.zeros([dim,dim])
	for i in range(source.shape[0]):
		W += np.dot(np.array([t1[i]]).T , np.array([s1[i]]))
	#- print "W = \n", W
	U, S, VT = np.linalg.svd(W)
	#- print "U = \n", U
	#- print "S = \n", S
	#- print "VT = \n", VT
	R = np.dot(U,VT).T    #- Row-major matrix
	t = t_mean - np.dot(s_mean, R)
	#print "R = \n", R
	#print "t = \n", t

	# Return Rotation and Translation separately
	return (R,t)
	# Retun Rotation and Translation in combined matrix
	#- T = np.identity(dim+1)
	#- T[0:dim,0:dim] = R
	#- T[dim,0:dim] = t
	#print "T = \n", T
	#- return T

'''
Return the transformation matrix which align the source to target
Iteration stops if change in error is less than 'tol' or more than specified number of runs
'''
def ICP (source,target,tol,run):
	dim = source.shape[1]
	ss = source.copy()
	err = 100000000    # arbitrary large number to start
	for i in range(run):
		e = err
		nn = Match(ss,target)
		#R,t = FindTransform(source,nn)
		R,t = FindTransform(source,nn)
		#print "R = \n", R
		#print "t = \n", t
		ss = np.dot(source, R) + t
		err = Distance(nn,ss)
		print "err = %f (%2d)" % (err,i)
		if ((e-err)<tol):
			break
	T = np.identity(dim+1)
	T[0:dim,0:dim] = R
	T[dim,0:dim] = t
	return T

'''
Initialise 
'''
def Initialise (source, target):
	src_len = source.shape[0]
	tgt_len = target.shape[0]
	#-- Initialize parameters: A = identity, b = zero, Rxyz = 0, Txyz = 0
	A = np.identity(3)
	b = np.zeros(3)
	P = np.append(np.tile(np.append(A.flatten(),b.flatten()),src_len),np.zeros(6))
	#-- Calculate weights over points
	w = np.zeros([src_len,src_len])
	for j in range(src_len):
		idx,dist = KNN(source,source[j],num_nn+2)
		#print idx,np.array_str(dist,100,4)
		denom = num_nn - dist[1:-1].sum()/dist[-1]
		for i in range(1,num_nn+1):
			w[idx[i],j] = (1 - dist[i]/dist[-1]) / denom
	#print w[0:10,0:10]
	return (P,w)

'''
Extract A & b from P for a vertex
'''
def PtoAB (P, v):
	#A = P[(v*12):(v*12+9)].reshape(3,3)
	A = P[(v*12):(v*12+9)].reshape(3,3)
	b = P[(v*12+9):(v*12+12)]
	return (A,b)

'''
Rotational matrix - rotate sequence = X, Y, Z
'''
def RotMatrix (x, y, z):
	sx = np.sin(x)
	cx = np.cos(x)
	sy = np.sin(y)
	cy = np.cos(y)
	sz = np.sin(z)
	cz = np.cos(z)
	R = np.array( \
		[[cy*cz, cy*sz, -sy], \
		 [sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, sx*cy], \
		 [cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy]])
	return R

'''
Differentiation of Rotational matrix - rotate sequence = X, Y, Z
'''
def D_RotMatrix (x, y, z):
	sx = np.sin(x)
	cx = np.cos(x)
	sy = np.sin(y)
	cy = np.cos(y)
	sz = np.sin(z)
	cz = np.cos(z)
	d_Rx = np.array( \
		[[0, 0, 0], \
		 [cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy], \
		 [-sx*sy*cz+cx*sz, -sx*sy*sz-cx*cz, -sx*cy]])
	d_Ry = np.array( \
		[[-sy*cz, -sy*sz, -cy], \
		 [sx*cy*cz, sx*cy*sz, -sx*sy], \
		 [cx*cy*cz, cx*cy*sz, -cx*sy]])
	d_Rz = np.array( \
		[[-cy*sz, cy*cz, 0], \
		 [-sx*sy*sz-cx*cz, sx*sy*cz-cx*sz, 0], \
		 [-cx*sy*sz+sx*cz, cx*sy*cz+sx*sz, 0]])
	return (d_Rx, d_Ry, d_Rz)

'''
Deform the source
'''
def Deform (P, w, source):
	deform = np.zeros_like(source)
	src_len = source.shape[0]
	#-- Local transformation
	for i in range(src_len):
		for j in range(src_len):
			A, b = PtoAB(P,j)
			if (w[j,i]!=0):
				deform[i] += w[j,i] * (np.dot((source[i]-source[j]),A) + source[j] + b)
	#return deform
	#-- Global transformation
	R = RotMatrix(P[-6],P[-5],P[-4])
	T = np.array([P[-3],P[-2],P[-1]])
	return np.dot(deform,R) + T

'''
Function to minimize 
Combination of E-fit, E-rigid, E-smooth
'''
def MinFunc (P, w, srcdata, edges, tgtdata):
	src_len = srcdata.shape[0]
	edge_len = edges.shape[0]
	#-- E-fit
	e_fit = tgtdata - Deform(P,w,srcdata)
	#-- E-rigid
	e_rigid = np.zeros(src_len*6)
	for i in range(src_len):
		A, b = PtoAB(P,i)
		e_rigid[i*3]   = A[0,0]*A[1,0] + A[0,1]*A[1,1] + A[0,2]*A[1,2]
		e_rigid[i*3+1] = A[1,0]*A[2,0] + A[1,1]*A[2,1] + A[1,2]*A[2,2]
		e_rigid[i*3+2] = A[2,0]*A[0,0] + A[2,1]*A[0,1] + A[2,2]*A[0,2]
		e_rigid[i*3+3] = 1 - A[0,0]*A[0,0] - A[0,1]*A[0,1] - A[0,2]*A[0,2]
		e_rigid[i*3+4] = 1 - A[1,0]*A[1,0] - A[1,1]*A[1,1] - A[1,2]*A[1,2]
		e_rigid[i*3+5] = 1 - A[2,0]*A[2,0] - A[2,1]*A[2,1] - A[2,2]*A[2,2]
	#-- E-smooth
	e_smooth = np.zeros((edge_len*2,3))
	for i in range(edge_len):
		p1 = edges[i,0]
		p2 = edges[i,1]
		A1,b1 = PtoAB(P,p1)
		A2,b2 = PtoAB(P,p2)
		e_smooth[i*2]   = np.dot((srcdata[p2]-srcdata[p1]),A1) + srcdata[p1] + b1 - (srcdata[p2] + b2)
		e_smooth[i*2+1] = np.dot((srcdata[p1]-srcdata[p2]),A2) + srcdata[p2] + b2 - (srcdata[p1] + b1)
	global c_fit,c_rigid,c_smooth
	e_fit *= c_fit
	e_rigid *= c_rigid
	e_smooth *= c_smooth
	return np.append(np.append(e_fit.flatten(),e_rigid),e_smooth.flatten())

'''
Differentiate of minimization function 
Sequence of rows must follow the definition of minimization function
Sequence of columns must follow the parameter list 
'''
def D_MinFunc (P, w, srcdata, edges, tgtdata):
	src_len = srcdata.shape[0]
	edges_len = edges.shape[0]
	nParam = P.shape[0]
	d_e_fit = np.zeros([src_len*3,nParam])
	d_e_rigid = np.zeros([src_len*6,nParam])
	d_e_smooth = np.zeros([edges_len*2*3,nParam])
	R = RotMatrix(P[-6],P[-5],P[-4])
	d_Rx, d_Ry, d_Rz = D_RotMatrix(P[-6],P[-5],P[-4])
	dd = Deform(P,w,srcdata)
	for i in range(src_len):
		#-- E-fit
		for j in range(src_len):
			if (w[j,i]!=0):
				disp = srcdata[i] - srcdata[j]
				d_e_fit[i*3:i*3+3,j*12]    = -w[j,i] * np.dot(disp,np.array([R[0],[0,0,0],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+1]  = -w[j,i] * np.dot(disp,np.array([R[1],[0,0,0],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+2]  = -w[j,i] * np.dot(disp,np.array([R[2],[0,0,0],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+3]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],R[0],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+4]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],R[1],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+5]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],R[2],[0,0,0]]))
				d_e_fit[i*3:i*3+3,j*12+6]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],[0,0,0],R[0]]))
				d_e_fit[i*3:i*3+3,j*12+7]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],[0,0,0],R[1]]))
				d_e_fit[i*3:i*3+3,j*12+8]  = -w[j,i] * np.dot(disp,np.array([[0,0,0],[0,0,0],R[2]]))
				d_e_fit[i*3:i*3+3,j*12+9]  = -w[j,i] * R[0]
				d_e_fit[i*3:i*3+3,j*12+10] = -w[j,i] * R[1]
				d_e_fit[i*3:i*3+3,j*12+11] = -w[j,i] * R[2]
		d_e_fit[i*3:i*3+3,-6] = np.dot(-dd[i],d_Rx)
		d_e_fit[i*3:i*3+3,-5] = np.dot(-dd[i],d_Ry)
		d_e_fit[i*3:i*3+3,-4] = np.dot(-dd[i],d_Rz)
		d_e_fit[i*3:i*3+3,-3] = np.array([-1,0,0])
		d_e_fit[i*3:i*3+3,-2] = np.array([0,-1,0])
		d_e_fit[i*3:i*3+3,-1] = np.array([0,0,-1])
		#-- E-rigid
		A,b = PtoAB(P,i)
		d_e_rigid[i*6,  i*12:i*12+9] = np.array([A[1,0],A[1,1],A[1,2], A[0,0],A[0,1],A[0,2], 0,0,0])
		d_e_rigid[i*6+1,i*12:i*12+9] = np.array([0,0,0, A[2,0],A[2,1],A[2,2], A[1,0],A[1,1],A[1,2]])
		d_e_rigid[i*6+2,i*12:i*12+9] = np.array([A[2,0],A[2,1],A[2,2], 0,0,0, A[0,0],A[0,1],A[0,2]])
		d_e_rigid[i*6+3,i*12:i*12+9] = np.array([-2*A[0,0],-2*A[0,1],-2*A[0,2], 0,0,0, 0,0,0])
		d_e_rigid[i*6+4,i*12:i*12+9] = np.array([0,0,0, -2*A[1,0],-2*A[1,1],-2*A[1,2], 0,0,0])
		d_e_rigid[i*6+5,i*12:i*12+9] = np.array([0,0,0, 0,0,0, -2*A[2,0],-2*A[2,1],-2*A[2,2]])
	#-- E-smooth
	for i in range(edges_len):
		p1 = edges[i,0]
		p2 = edges[i,1]
		disp = srcdata[p2] - srcdata[p1]
		
		d_e_smooth[i*6,  p1*12:p1*12+12] = np.array([disp[0],0,0, disp[1],0,0, disp[2],0,0, 1,0,0])
		d_e_smooth[i*6+1,p1*12:p1*12+12] = np.array([0,disp[0],0, 0,disp[1],0, 0,disp[2],0, 0,1,0])
		d_e_smooth[i*6+2,p1*12:p1*12+12] = np.array([0,0,disp[0], 0,0,disp[1], 0,0,disp[2], 0,0,1])
		d_e_smooth[i*6,  p2*12+9:p2*12+12] = np.array([-1,0,0])
		d_e_smooth[i*6+1,p2*12+9:p2*12+12] = np.array([0,-1,0])
		d_e_smooth[i*6+2,p2*12+9:p2*12+12] = np.array([0,0,-1])

		d_e_smooth[i*6+3,p2*12:p2*12+12] = np.array([-disp[0],0,0, -disp[1],0,0, -disp[2],0,0, 1,0,0])
		d_e_smooth[i*6+4,p2*12:p2*12+12] = np.array([0,-disp[0],0, 0,-disp[1],0, 0,-disp[2],0, 0,1,0])
		d_e_smooth[i*6+5,p2*12:p2*12+12] = np.array([0,0,-disp[0], 0,0,-disp[1], 0,0,-disp[2], 0,0,1])
		d_e_smooth[i*6+3,p1*12+9:p1*12+12] = np.array([-1,0,0])
		d_e_smooth[i*6+4,p1*12+9:p1*12+12] = np.array([0,-1,0])
		d_e_smooth[i*6+5,p1*12+9:p1*12+12] = np.array([0,0,-1])
	global c_fit, c_rigid, c_smooth
	d_e_fit *= c_fit
	d_e_rigid *= c_rigid
	d_e_smooth *= c_smooth
	return np.append(np.append(d_e_fit,d_e_rigid,axis=0),d_e_smooth,axis=0)

'''
Return the transformation matrix which align the source to target
Iteration stops if change in error is less than 'tol' or more than specified number of runs
'''
def NICP (P, w, source, edges, target, tol, run):
	global c_fit, c_rigid, c_smooth
	err = 100000000    # arbitrary large number to start
	for i in range(run):
		e = err
		dd = Deform(P,w,source)
		nn = Match(dd,target)
		lsq = optimize.leastsq(MinFunc,P,(w,source,edges,nn),D_MinFunc,full_output=1)
		P = lsq[0]
		err = (lsq[2]['fvec']*lsq[2]['fvec']).sum()
		print "{0:2d} error = {1:f} [{2:f},{3:f}]".format(i,err,c_rigid,c_smooth)
		if ((e-err)<tol):
			c_rigid /= 2.0
			c_smooth /= 2.0
			if ((c_rigid<0.1) or (c_smooth<0.1)):
				break
	#-- Apply the final deformation parameters
	dd = Deform(P,w,source)
	print P
	return dd, P

'''
Convert Numpy.Array to Maya MMatrix
'''
def ArrayToMMatrix (arr):
	return om.MMatrix((\
	(arr[0,0], arr[0,1], arr[0,2], arr[0,3]), \
	(arr[1,0], arr[1,1], arr[1,2], arr[1,3]), \
	(arr[2,0], arr[2,1], arr[2,2], arr[2,3]), \
	(arr[3,0], arr[3,1], arr[3,2], arr[3,3])))

'''
Modify vertices of mesh based on parameters
'''
def ModifyVertices (path, dset):
	#-- Sample code to change vertices position
	#-- Anchor of mesh does not change
	points = []
	for i in range(dset.shape[0]):
		points.append(om.MPoint(dset[i,0],dset[i,1],dset[i,2]))
	mesh = om.MFnMesh(path)
	mesh.setPoints(points,om.MSpace.kWorld)

'''
Main function to link everything
'''
def main():
	print "NICP - Non-rigid Iterative Closest Point"
	srcPath, tgtPath = GetModel()
	if (srcPath is None):
		print "You must select 2 models"
	else:
		print ("Source model is {0:s}".format(srcPath.partialPathName()))
		print ("Target model is {0:s}".format(tgtPath.partialPathName()))
		if (srcPath.hasFn(om.MFn.kMesh)==False) or (tgtPath.hasFn(om.MFn.kMesh)==False):
			print "Selected models are not a mesh"
		else:
			srcPts,srcEdges = GetVerticesEdges(srcPath)
			tgtPts,tgtEdges = GetVerticesEdges(tgtPath)
			print ("Source mode has {0:d} vertices, {1:d} edges".format(srcPts.shape[0],srcEdges.shape[0]))
			print ("Target mode has {0:d} vertices, {1:d} edges".format(tgtPts.shape[0],tgtEdges.shape[0]))
			P0,w = Initialise(srcPts,tgtPts)
			nn = Match(srcPts,tgtPts)
			#newPts, P = NICP(P0,w,srcPts,srcEdges,tgtPts,0.01,50)
			#ModifyVertices(srcPath,newPts)
			
# ---- Template code for Maya Plugin Command ---- #

def maya_useNewAPI():
	"""
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
	pass


# command
class PyIcp3dCmd(om.MPxCommand):
	kPluginCmdName = "icp"

	def __init__(self):
		om.MPxCommand.__init__(self)

	@staticmethod
	def cmdCreator():
		return PyIcp3dCmd()

	def doIt(self, args):
		main()


# Initialize the plug-in
def initializePlugin(plugin):
	pluginFn = om.MFnPlugin(plugin)
	try:
		pluginFn.registerCommand(
			PyIcp3dCmd.kPluginCmdName, PyIcp3dCmd.cmdCreator
		)
	except:
		sys.stderr.write(
			"Failed to register command: %s\n" % PyIcp3dCmd.kPluginCmdName
		)
		raise

# Uninitialize the plug-in
def uninitializePlugin(plugin):
	pluginFn = om.MFnPlugin(plugin)
	try:
		pluginFn.deregisterCommand(PyIcp3dCmd.kPluginCmdName)
	except:
		sys.stderr.write(
			"Failed to unregister command: %s\n" % PyIcp3dCmd.kPluginCmdName
		)
		raise

if __name__ == "__main__":
	main()

#-
# ==========================================================================
# Copyright (C) 2011 Autodesk, Inc. and/or its licensors.  All 
# rights reserved.
#
# The coded instructions, statements, computer programs, and/or related 
# material (collectively the "Data") in these files contain unpublished 
# information proprietary to Autodesk, Inc. ("Autodesk") and/or its 
# licensors, which is protected by U.S. and Canadian federal copyright 
# law and by international treaties.
#
# The Data is provided for use exclusively by You. You have the right 
# to use, modify, and incorporate this Data into other products for 
# purposes authorized by the Autodesk software license agreement, 
# without fee.
#
# The copyright notices in the Software and this entire statement, 
# including the above license grant, this restriction and the 
# following disclaimer, must be included in all copies of the 
# Software, in whole or in part, and all derivative works of 
# the Software, unless such copies or derivative works are solely 
# in the form of machine-executable object code generated by a 
# source language processor.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. 
# AUTODESK DOES NOT MAKE AND HEREBY DISCLAIMS ANY EXPRESS OR IMPLIED 
# WARRANTIES INCLUDING, BUT NOT LIMITED TO, THE WARRANTIES OF 
# NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR 
# PURPOSE, OR ARISING FROM A COURSE OF DEALING, USAGE, OR 
# TRADE PRACTICE. IN NO EVENT WILL AUTODESK AND/OR ITS LICENSORS 
# BE LIABLE FOR ANY LOST REVENUES, DATA, OR PROFITS, OR SPECIAL, 
# DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES, EVEN IF AUTODESK 
# AND/OR ITS LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY 
# OR PROBABILITY OF SUCH DAMAGES.
#
# ==========================================================================
#+

