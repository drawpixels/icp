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
c_rigid = 1000
c_smooth = 100

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
Initialise 
'''
def Initialise (source, target):
	src_len = source.shape[0]
	tgt_len = target.shape[0]
	#-- Initialize parameters: A = identity, b = zero, r = 0, t = 0
	P = np.append(np.tile(np.array([1,0,0,1,0,0.0]),src_len),np.array([0,0,0]))
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
	for i in range(src_len):
		for j in range(src_len):
			A = P[(j*6):(j*6+4)].reshape(2,2)
			b = P[(j*6+4):(j*6+6)]
			if (w[j,i]!=0):
				deform[i] += w[j,i] * (np.dot((source[i]-source[j]),A) + source[j] + b)
	#return deform
	#-- Global transformation
	R = np.array([[np.cos(P[-3]),np.sin(P[-3])],[-np.sin(P[-3]),np.cos(P[-3])]])
	T = np.array([P[-2],P[-1]])
	return np.dot(deform,R) + T

'''
Function to minimize 
Combination of E-fit, E-rigid, E-smooth
'''
def MinFunc (P, w, srcdata, tgtdata, deformdata):
	src_len = srcdata.shape[0]

	'''
	?????? - THE FOLLOWING VALUES ARE DIFFERENT ??????
	'''
	d = Deform(srcdata,P,w)
	print d-deformdata


	#-- E-fit
	e_fit = tgtdata - Deform(srcdata,P,w) #deformdata
	#-- E-rigid & E-smooth
	e_rigid = np.zeros(src_len*3)
	e_smooth = np.zeros((src_len*2,2))
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
Differentiate of minimization function 
Sequence of rows must follow the definition of minimization function
Sequence of columns must follow the parameter list 

WORK IN PROGRESS
'''
def D_MinFunc (P, w, srcdata, tgtdata, deformdata):
	src_len = srcdata.shape[0]
	nParam = P.shape[0]
	d_e_fit = np.zeros([src_len*2,nParam])
	d_e_rigid = np.zeros([src_len*3,nParam])
	d_e_smooth = np.zeros([src_len*2*2,nParam])
	R = np.array([[np.cos(P[-3]),np.sin(P[-3])],[-np.sin(P[-3]),np.cos(P[-3])]])
	d_R = np.array([[-np.sin(P[-3]),np.cos(P[-3])],[-np.cos(P[-3]),-np.sin(P[-3])]])
	for i in range(src_len):
		A = P[(i*6):(i*6+4)].reshape(2,2)
		#-- E-fit
		for j in range(src_len):
			if (w[j,i]!=0):
				disp = source[i] - source[j]
				d_e_fit[i*2:i*2+2,j*6]   = w[j,i] * np.dot(disp,np.array([R[0],[0,0]]))
				d_e_fit[i*2:i*2+2,j*6+1] = w[j,i] * np.dot(disp,np.array([R[1],[0,0]]))
				d_e_fit[i*2:i*2+2,j*6+2] = w[j,i] * np.dot(disp,np.array([[0,0],R[0]]))
				d_e_fit[i*2:i*2+2,j*6+3] = w[j,i] * np.dot(disp,np.array([[0,0],R[1]]))
				d_e_fit[i*2:i*2+2,j*6+4] = R[0]
				d_e_fit[i*2:i*2+2,j*6+5] = R[1]
		d_e_fit[i*2:i*2+2,-3] = np.dot(-deformdata[i],d_R)
		d_e_fit[i*2:i*2+2,-2] = np.array([-1,0])
		d_e_fit[i*2:i*2+2,-1] = np.array([0,-1])
		#-- E-rigid
		d_e_rigid[i*3,  i*6]   = A[1,0]
		d_e_rigid[i*3+1,i*6]   = -2*A[0,0]
		d_e_rigid[i*3+2,i*6]   = 0
		d_e_rigid[i*3,  i*6+1] = A[1,1]
		d_e_rigid[i*3+1,i*6+1] = -2*A[0,1]
		d_e_rigid[i*3+2,i*6+1] = 0
		d_e_rigid[i*3,  i*6+2] = A[0,0]
		d_e_rigid[i*3+1,i*6+2] = 0
		d_e_rigid[i*3+2,i*6+2] = -2*A[1,0]
		d_e_rigid[i*3,  i*6+3] = A[0,1]
		d_e_rigid[i*3+1,i*6+3] = 0
		d_e_rigid[i*3+2,i*6+3] = -2*A[1,1]		
		#-- E-smooth
		if (i!=0):
			disp = source[i-1] - source[i]
			d_e_smooth[i*4:i*4+2,i*6]   = np.array([disp[0],0])
			d_e_smooth[i*4:i*4+2,i*6+1] = np.array([0,disp[0]])
			d_e_smooth[i*4:i*4+2,i*6+2] = np.array([disp[1],0])
			d_e_smooth[i*4:i*4+2,i*6+3] = np.array([0,disp[1]])
			d_e_smooth[i*4:i*4+2,i*6+4] = np.array([1,0])
			d_e_smooth[i*4:i*4+2,i*6+5] = np.array([0,1])
			d_e_smooth[i*4:i*4+2,i*6-2] = np.array([1,0])
			d_e_smooth[i*4:i*4+2,i*6-1] = np.array([0,1])
		if (i!=(src_len-1)):
			disp = source[i+1] - source[i]
			d_e_smooth[i*4+2:i*4+4,i*6]   = np.array([disp[0],0])
			d_e_smooth[i*4+2:i*4+4,i*6+1] = np.array([0,disp[0]])
			d_e_smooth[i*4+2:i*4+4,i*6+2] = np.array([disp[1],0])
			d_e_smooth[i*4+2:i*4+4,i*6+3] = np.array([0,disp[1]])
			d_e_smooth[i*4+2:i*4+4,i*6+4] = np.array([1,0])
			d_e_smooth[i*4+2:i*4+4,i*6+5] = np.array([0,1])
			d_e_smooth[i*4+2:i*4+4,i*6+10] = np.array([1,0])
			d_e_smooth[i*4+2:i*4+4,i*6+11] = np.array([0,1])
	d_e_fit *= c_fit
	d_e_rigid *= c_rigid
	d_e_smooth *= c_smooth
	return np.append(np.append(d_e_fit,d_e_rigid,axis=0),d_e_smooth,axis=0)

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
	lsq = optimize.leastsq(MinFunc,P0,(w,source,nn,dd),D_MinFunc,full_output=1)
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
	if ((cont=="R")or(cont=="r")):
		c_rigid /= 2.0
		c_smooth /= 2.0
	if ((cont=="N")or(cont=="n")):
		break

