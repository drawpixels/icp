'''
icp_3d_new.py

Iterative Closest Point algorithm in 3D.

'''

import sys
import maya as m
import maya.api.OpenMaya as om
import numpy as np

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
def GetVertices (path):
    mesh = om.MFnMesh(path)
    pts = mesh.getPoints(om.MSpace.kWorld)
    a = np.zeros([len(pts),3])
    for i in range(len(pts)):
        a[i,0] = pts[i].x
        a[i,1] = pts[i].y
        a[i,2] = pts[i].z
    return a

'''
Return the index of the closest point in the array
'''
def closest (arr, pt):
    diff = arr - pt
    dist = (diff*diff).sum(1)
    return np.argmin(dist)

'''
For each point in source, find the closest point in the target. 
Return the closest points in array
'''
def match (source, target):
    nn = np.zeros(source.shape)
    for i in range(source.shape[0]):
        #- nn[i,:] = target[closest(target,source[i,:]),:]
        c = closest(target,source[i,:])
        nn[i,:] = target[c,:]
        #- print i,c
        # Debug - link between points in source and target mesh
        str = "distanceDimension -sp %f %f %f -ep %f %f %f" % \
          (source[i,0], source[i,1], source[i,2], \
          nn[i,0], nn[i,1], nn[i,2])
        m.mel.eval(str)
    return nn

'''
Transform source to match target so that the sum sq error is minimum.
Return Transformation matrix
'''
def findTransform (source, target):
    s_mean = source.mean(0)
    s1 = source - s_mean
    t_mean = target.mean(0)
    t1 = target - t_mean
    print "s_mean = ", s_mean
    print "t_mean = ", t_mean
    #- print s1.shape, t1.shape
    #- print np.array([s1[0,:]])
    #- print np.array([t1[0,:]])
    #- print np.dot(np.array([t1[0,:]]) , np.array([s1[0,:]]).T)
    #- print np.dot(np.array([t1[0,:]]).T , np.array([s1[0,:]]))
    W = np.zeros([3,3])
    for i in range(source.shape[0]):
        W += np.dot(np.array([t1[i,:]]).T , np.array([s1[i,:]]))
    print "W = \n", W
    U, S, VT = np.linalg.svd(W)
    print "U = \n", U
    print "S = \n", S
    print "VT = \n", VT

    R = np.dot(VT, U)    #-- ???
    t = t_mean - np.dot(s_mean, R)
    print "R = \n", R
    print "t = \n", t
    T = np.identity(4)
    T[0:3,0:3] = R
    T[3,0:3] = t
    print "T = \n", T

    # DEBUG: Tranform with fixed amount
    #- T = np.identity(4)
    # Rotate about X axis in row-major matrix form
    #- T[1,1] = np.cos(1)
    #- T[1,2] = np.sin(1)
    #- T[2,1] = -np.sin(1)
    #- T[2,2] = np.cos(1)
    # Rotate about Y axis in row-major matrix form
    #- T[0,0] = np.cos(1)
    #- T[0,2] = -np.sin(1)
    #- T[2,0] = np.sin(1)
    #- T[2,2] = np.cos(1)
    # Rotate about Z axis in row-major matrix form
    #- T[0,0] = np.cos(1)
    #- T[0,1] = np.sin(1)
    #- T[1,0] = -np.sin(1)
    #- T[1,1] = np.cos(1)
    # Translate along X/Y/Z axis in row-major matrix form
    #- T[3,0] = 40
    #- T[3,1] = 40
    #- T[3,2] = 40
    return T

'''
Convert Numpy.Array to Maya MMatrix
'''
def arrayToMMatrix (arr):
    return om.MMatrix((\
    (arr[0,0], arr[0,1], arr[0,2], arr[0,3]), \
    (arr[1,0], arr[1,1], arr[1,2], arr[1,3]), \
    (arr[2,0], arr[2,1], arr[2,2], arr[2,3]), \
    (arr[3,0], arr[3,1], arr[3,2], arr[3,3])))

'''
Main function to link everything
'''
def main():
    print "ICP - Iterative Closest Point"
    srcPath, tgtPath = GetModel()
    if (srcPath is None):
        print "You must select 2 models"
    else:
        print ("Source model is %s" % srcPath.partialPathName( ))
        print ("Target model is %s" % tgtPath.partialPathName())
        if (srcPath.hasFn(om.MFn.kMesh)==False) or (tgtPath.hasFn(om.MFn.kMesh)==False):
            print "Selected models are not a mesh"
        else:
            srcPts = GetVertices(srcPath)
            tgtPts = GetVertices(tgtPath)
            print ("Source mode has %d vertices" % srcPts.shape[0])
            print ("Target mode has %d vertices" % tgtPts.shape[0])
            corrPts = match(srcPts,tgtPts)
            Trans = findTransform(srcPts, corrPts)
            transform = om.MFnTransform(srcPath)
            m = transform.transformation().asMatrix()
            print m
            t = arrayToMMatrix (Trans)
            m = m * t
            print t
            print m
            transform.setTransformation(om.MTransformationMatrix(m))

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

