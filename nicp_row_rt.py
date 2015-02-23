'''
nicp_row_rt.py


Non-rigid ICP in row-major implementation. 
No deformation between source & target. Only Rotation and translation. 

'''

from scipy import optimize 
import numpy as np
import matplotlib.pyplot as pt

'''
Global Constants
'''
target_len = 50
source_len = 20
rotate = 0.875
translate_x = 2.0
translate_y = 3.5
num_nn = 4

'''
Global Variables - may move to local variables
'''
r = 0.0
tx = 0.0
ty = 0.0

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
    # Rotate & translate source dataset
    R = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])
    T = np.array([tx,ty])
    dset = np.dot(dset,R) + T
    return dset

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
Deform the source
'''
def Deform (dset, rot, tx, ty):
	deform = np.zeros_like(dset)
	R = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])
	T = np.array([tx,ty])
	#deform = np.dot((dset-T),R)
	deform = np.dot(dset,R) + T
	return deform

'''
Displacement between source and target
This is the function to be minimized
'''
def Disp (p, srcdata, tgtdata):
	d = tgtdata - Deform(srcdata,p[0],p[1],p[2])
	return d.flatten()

'''
Derivative of displacement function
Not working yet
'''
def D_Disp (p, srcdata, tgtdata):
	d_r = -1 * np.dot(srcdata, np.array([[-np.sin(p[0]),np.cos(p[0])],[-np.cos(p[0]),-np.sin(p[0])]]))
	d_tx = np.array([-1,0])
	d_ty = np.array([0,-1])
	d_d = np.zeros([np.product(srcdata.shape),3])
	d_d[:,0] = d_r.flatten()
	d_d[:,1] = np.tile(d_tx.flatten(),srcdata.shape[0])
	d_d[:,2] = np.tile(d_ty.flatten(),srcdata.shape[0])
	'''
	d_r = -1 * np.dot((srcdata-np.array([p[1],p[2]])), \
			np.array([[-np.sin(p[0]),np.cos(p[0])],[-np.cos(p[0]),-np.sin(p[0])]]))
	d_tx = np.array([np.cos(p[0]),np.sin(p[0])])
	d_ty = np.array([-np.sin(p[0]),np.cos(p[0])])
	d_d = np.zeros([np.product(srcdata.shape),3])
	d_d[:,0] = d_r.flatten()
	d_d[:,1] = np.tile(d_tx.flatten(),srcdata.shape[0])
	d_d[:,2] = np.tile(d_ty.flatten(),srcdata.shape[0])
	'''
	return d_d

target = GenerateTarget(target_len)
source = GenerateSource(source_len,rotate,translate_x,translate_y)

#--- Initial guess
p0 = np.array([r,tx,ty])

pt.ion()
while (True):
	dd = Deform(source,p0[0],p0[1],p0[2])
	nn = Match(dd,target)
	l = optimize.leastsq(Disp,p0,(source,nn),D_Disp,full_output=1)
	print l[0], (l[2]['fvec']*l[2]['fvec']).sum()
	p0 = l[0]
	pt.plot(target[:,0],target[:,1],'r.')
	pt.plot(source[:,0],source[:,1],'bo')
	pt.plot(dd[:,0],dd[:,1],'b.')
	#pt.plot(ss[:,0],ss[:,1],'g^')
	pt.plot(nn[:,0],nn[:,1],'ro')
	pt.show()
	cont = raw_input("Continue?")
	if ((cont=="n")or(cont=="N")):
		pt.close()
		break
	pt.close()

