'''
NICP v1

2D non-rigid ICP based on Hao Li's 2008 SIGGRAPH paper. 
Ignore the global rotation and translation. 

Deformed source is not enabled to test the basic code is running at this stage. 
Row major implementation (row vector)

'''

from scipy import optimize
import numpy as np
import matplotlib.pyplot as pt
# import matplotlib.cm as cm
# import math as m

'''
Global Constants
'''
target_len = 50
source_len = 20
rotate = 0.83
translate_x = 2.0
translate_y = 3.5
num_nn = 4
c_fit = 0.1
c_rigid = 100
c_smooth = 10

'''
Global Variables - may move to local variables
'''
P = np.empty([])
w = np.empty([])

'''
Generate target dataset
'''
def GenerateTarget (nSample):
	x = np.linspace(0,2*np.pi,nSample)
	dset = np.array([x,np.sin(x)]).T
	return dset

'''
Generate source dataset
Create random sample positions over the same range (i.e. sample points don't align)
'''
def GenerateSource (nSample, rot, tx, ty):
	x = np.sort(np.random.random(nSample))*2*np.pi
	dset = np.array([x,np.sin(x)]).T
	#-- Introduce deformatio
	for i in range(1,nSample):
		#dset[i,0] *= 0.2 + (i*4.0)/nSample
		dset[i,1] *= 0.5 + float(i)/float(nSample)
	# Rotate & translate source dataset
	R = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])
	T = np.array([tx,ty])
	dset = np.dot(dset,R) + T
	return dset

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
	return nn

'''
Iterative Closest Point
'''
'''
def ICP (source,target,tol,run):
    dim = source.shape[1]
    ss = source.copy()
    pt.ion()
    for i in range(run):
        ref = ss.copy()
        nn = Match(ss,target)
        #R,t = FindTransform(source,nn)
        T = FindTransform(source,nn)
        R = T[0:dim,0:dim]
        t = T[dim,0:dim]
        #print "R = \n", R
        #print "t = \n", t
        ss = np.dot(source, R) + t
        err = Distance(nn,ss)
        print "err = %f (%2d)" % (err,i)

        # plot
        #pt.xlim(-2,8)
        #pt.ylim(-2,15)
        pt.plot(target[:,0],target[:,1],'r.')
        pt.plot(source[:,0],source[:,1],'bo')
        pt.plot(ref[:,0],ref[:,1],'b.')
        pt.plot(ss[:,0],ss[:,1],'g^')
        pt.plot(nn[:,0],nn[:,1],'ro')
        pt.show()

        cont = raw_input("Continue ? ")
        if ((cont!="Y")and(cont!="y")):
            pt.close()
            break
        pt.close()
'''

'''
Initialise 
'''
def Initialise (source, target):
	src_len = source.shape[0]
	tgt_len = target.shape[0]
	#-- Initialize parameters: A = identity, b = zero, r = 0, t = 0
	P = np.append(np.tile(np.array([1,0,0,1,0.2,0.0]),src_len),np.array([0,0,0]))
	#P[24:28] = np.array([np.cos(0.86),np.sin(0.86),-np.sin(0.86),np.cos(0.86)])
	#-- Calculate weights over points
	w = np.zeros([src_len,src_len])
	for j in range(src_len):
		idx,dist = KNN(source,source[j],num_nn+2)
		#print idx,np.array_str(dist,100,4)
		denom = num_nn - dist[1:-1].sum()/dist[-1]
		for i in range(1,num_nn+1):
			w[idx[i],j] = (1 - dist[i]/dist[-1]) / denom
	return (P,w)

'''
Deform the source
'''
def Deform (source, P, w):
	deform = np.zeros_like(source)
	src_len = source.shape[0]
	#-- Local transformation
	for j in range(src_len):
		for i in range(src_len):
			if (w[i,j]!=0):
				A = P[(i*6):(i*6+4)].reshape(2,2)
				b = P[(i*6+4):(i*6+6)]
				deform[j] += w[i,j] * (np.dot((source[j]-source[i]),A) + source[i] + b)
	#return deform
	#-- Global transformation
	R = np.array([[np.cos(P[-3]),np.sin(P[-3])],[-np.sin(P[-3]),np.cos(P[-3])]])
	T = np.array([P[-2],P[-1]])
	return np.dot(deform,R) + T

'''
Function to minimize 
Combination of E-fit, E-rigid, E-smooth
'''
def MinFunc (P, w, srcdata, tgtdata):
	src_len = srcdata.shape[0]
	#-- E-fit
	e_fit = (tgtdata - Deform(srcdata,P,w)).flatten()
	#-- E-rigid
	e_rigid = np.zeros(src_len*3)
	e_smooth = np.zeros((src_len*3,2))
	for i in range(src_len):
		A = P[(i*6):(i*6+4)].reshape(2,2)
		b = P[(i*6+4):(i*6+6)]
		#-- E-rigid
		e_rigid[i*3] = A[0,0]*A[1,0] + A[0,1]*A[1,1]
		e_rigid[i*3+1] = 1 - A[0,0]*A[0,0] - A[0,1]*A[0,1]
		e_rigid[i*3+2] = 1 - A[1,0]*A[1,0] - A[1,1]*A[1,1]
		#-- E-smooth
		if (i!=0):
			e_smooth[i*2] = np.dot((srcdata[i-1]-source[i]),A) + source[i] + b - source[i-1] - P[(i*6-2):(i*6-1)]
		if (i!=(src_len-1)):
			e_smooth[i*2+1] = np.dot((srcdata[i+1]-source[i]),A) + source[i] + b - source[i+1] - P[(i*6+10):(i*6+11)]
	e_fit *= c_fit
	e_rigid *= c_rigid
	e_smooth *= c_smooth
	return np.append(np.append(e_fit.flatten(),e_rigid),e_smooth.flatten())

'''
MAIN PROGRAMME START
'''
target = GenerateTarget(target_len)
source = GenerateSource(source_len,rotate,translate_x,translate_y)
#ICP(source,target,0.1,20)
P0, w = Initialise(source,target)
print "Initialized"

pt.ion()
while (True):
	dd = Deform(source,P0,w)
	nn = Match(dd,target)
	lsq = optimize.leastsq(MinFunc,P0,(w,source,nn),full_output=1)
	print lsq[0], (lsq[2]['fvec']*lsq[2]['fvec']).sum()
	P0 = lsq[0]
	# plot
	#pt.xlim(-2,8)
	#pt.ylim(-2,15)
	pt.plot(target[:,0],target[:,1],'r.')
	pt.plot(source[:,0],source[:,1],'g^')
	pt.plot(dd[:,0],dd[:,1],'b.')
	pt.plot(nn[:,0],nn[:,1],'ro')
	pt.show()
	cont = raw_input("Continue ? ")
	pt.close()
	if ((cont=="N")or(cont=="n")):
		break

