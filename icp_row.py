'''
ICP v2

Very Basic ICP
2 sine curves with only rotation and translation
Source curve has random samples

Row major implementation (row vector)

'''

import numpy as np
import matplotlib.pyplot as pt
# import matplotlib.cm as cm
# import math as m

'''
Global Constants
'''
target_len = 20
source_len = 20
rotate = 0.8
translate_x = 2.0
translate_y = 3.5

'''
Generate target dataset
'''
def GenerateTarget (nSample):
    x = np.linspace(0,2*np.pi,target_len)
    dset = np.array([x,np.sin(x)]).T
    return dset

'''
Generate source dataset
Create random sample positions over the same range (i.e. sample points don't align)
'''
def GenerateSource (nSample, rot, tx, ty):
    # Rotate & translate source dataset
    R = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])
    T = np.array([tx,ty])
    x = np.sort(np.random.random(source_len))*2*np.pi
    dset = np.array([x,np.sin(x)]).T
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
    #- return (R,t)
    # Retun Rotation and Translation in combined matrix
    T = np.identity(dim+1)
    T[0:dim,0:dim] = R
    T[dim,0:dim] = t
    #print "T = \n", T
    return T

'''
Iterative Closest Point
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
        pt.xlim(-2,8)
        pt.ylim(-2,8)
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
MAIN PROGRAMME START
'''
target = GenerateTarget(target_len)
source = GenerateSource(source_len,rotate,translate_x,translate_y)
ICP(source,target,0.1,20)

