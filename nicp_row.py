'''
NICP v1

2D non-rigid ICP based on Hao Li's 2008 SIGGRAPH paper. 
Ignore the global rotation and translation. 

Deformed source is not enabled to test the basic code is running at this stage. 
Row major implementation (row vector)

'''

import numpy as np
import matplotlib.pyplot as pt
# import matplotlib.cm as cm
# import math as m

'''
Global Constants
'''
target_len = 50
source_len = 5
rotate = 0.0
translate_x = 2.0
translate_y = 3.5
num_nn = 2
c_fit = 0.1
c_rigid = 100
c_smooth = 10

'''
Global Variables - may move to local variables
'''
A = np.empty([])
b = np.empty([])
R = 0.0
tx = 0.0
ty = 0.0
w = np.empty([])
J = np.empty([])
r = np.empty([])
dd = np.empty([])
nn = np.empty([])

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
    #for i in range(nSample):
    #    dset[i,0] *= 1.5
    #    dset[i,1] *= 0.2 + (i*4.0)/nSample
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
Initialise 
'''
def Initialise (source, target):
    src_len = source.shape[0]
    tgt_len = target.shape[0]
    global A, b, w
    A = np.empty([src_len,2,2])
    for i in range(src_len):
        A[i] = np.identity(2)
    b = np.zeros([src_len,2])
    w = np.zeros([src_len,src_len])
    for j in range(src_len):
        idx,dist = KNN(source,source[j],num_nn+2)
        #print idx,np.array_str(dist,100,4)
        denom = num_nn - dist[1:-1].sum()/dist[-1]
        for i in range(1,num_nn+1):
            w[idx[i],j] = (1 - dist[i]/dist[-1]) / denom

'''
Deform the source
'''
def Deform (source):
    deform = np.zeros_like(source)
    src_len = source.shape[0]
    for j in range(src_len):
        for i in range(src_len):
            if (w[i,j]!=0):
                deform[j] += w[i,j] * (np.dot((source[j]-source[i]),A[i]) + source[i] + b[i])
    return deform

def Jacobian (source, target):
    global A, b, w, J
    #J = np.zeros([src_len,A.shape[0]*A.shape[1]+b.shape[0]*b.shape[1]+3,2])
    #-- Omit the global rotation and translation first --#
    #-- arrange of parameters [A00, A01, A10, A11, b0, b1] for each sample point
    src_len = source.shape[0]
    #J = np.zeros([src_len*9,src_len*6])
    J = np.zeros([src_len*5,src_len*6])

    for j in range(src_len):
        #jj = j * 9
        jj = j * 5
        for i in range(src_len):
            ii = i * 6
            if (w[i,j]!=0):
                #-- E fit (x)
                d = source[j] - source[i]
                J[jj,ii] = w[i,j] * d[0]
                J[jj,ii+1] = 0
                J[jj,ii+2] = w[i,j] * d[1]
                J[jj,ii+3] = 0
                J[jj,ii+4] = w[i,j]
                J[jj,ii+5] = 0
                J[jj] *= c_fit
                #-- E fit (y)
                #jj += 1
                J[jj+1,ii] = 0
                J[jj+1,ii+1] = w[i,j] * d[0]
                J[jj+1,ii+2] = 0
                J[jj+1,ii+3] = w[i,j] * d[1]
                J[jj+1,ii+4] = 0
                J[jj+1,ii+5] = w[i,j]
                J[jj+1] *= c_fit
        #-- E rigid (a1.T * a2)
        jj += 2
        ii = j * 6
        J[jj,ii] = A[i,1,0]
        J[jj,ii+1] = A[i,1,1]
        J[jj,ii+2] = A[i,0,0]
        J[jj,ii+3] = A[i,0,1]
        J[jj] *= c_rigid
        #-- E rigid (1 - a1.T * a1)
        jj += 1
        J[jj,ii] = -2 * A[i,0,0]
        J[jj,ii+1] = -2 * A[i,0,1]
        J[jj] *= c_rigid
        #-- E rigid (1 - a2.T * a2)
        jj += 1
        J[jj,ii+2] = -2 * A[i,1,0]
        J[jj,ii+3] = -2 * A[i,1,1]
        J[jj] *= c_rigid
'''
        #-- E smoooth (x)
        jj += 1
        if (j==0):
            jj += 1
            d = source[j+1] - source[j]
            J[jj,ii] = d[0]
            J[jj,ii+1] = 0
            J[jj,ii+2] = d[1]
            J[jj,ii+3] = 0
            J[jj,ii+4] = 1
            J[jj,ii+5] = 0
            J[jj,ii+6+4] = -1    # b[j+1,0]
            J[jj,ii+6+5] = 0     # b[j+1,1]
            J[jj] *= c_smooth
        elif (j==(src_len-1)):
            d = source[j-1] - source[j]
            J[jj,ii] = d[0]
            J[jj,ii+1] = 0
            J[jj,ii+2] = d[1]
            J[jj,ii+3] = 0
            J[jj,ii+4] = 1
            J[jj,ii+5] = 0
            J[jj,ii-6+4] = -1    # b[j-1,0]
            J[jj,ii-6+5] = 0     # b[j-1,1]
            J[jj] *= c_smooth
        else:
            d = source[j-1] - source[j]
            J[jj,ii] = d[0]
            J[jj,ii+1] = 0
            J[jj,ii+2] = d[1]
            J[jj,ii+3] = 0
            J[jj,ii+4] = 1
            J[jj,ii+5] = 0
            J[jj,ii-6+4] = -1    # b[j-1,0]
            J[jj,ii-6+5] = 0     # b[j-1,1]
            J[jj] *= c_smooth
            jj += 1
            d = source[j+1] - source[j]
            J[jj,ii] = d[0]
            J[jj,ii+1] = 0
            J[jj,ii+2] = d[1]
            J[jj,ii+3] = 0
            J[jj,ii+4] = 1
            J[jj,ii+5] = 0
            J[jj,ii+6+4] = -1    # b[j+1,0]
            J[jj,ii+6+5] = 0     # b[j+1,1]
            J[jj] *= c_smooth
        #-- E smoooth (y)
        jj += 1
        if (j==0):
            jj += 1
            d = source[j+1] - source[j]
            J[jj,ii] = 0
            J[jj,ii+1] = d[0]
            J[jj,ii+2] = 0
            J[jj,ii+3] = d[1]
            J[jj,ii+4] = 0
            J[jj,ii+5] = 1
            J[jj,ii+6+4] = 0     # b[j+1,0]
            J[jj,ii+6+5] = -1    # b[j+1,1]
            J[jj] *= c_smooth
        elif (j==(src_len-1)):
            d = source[j-1] - source[j]
            J[jj,ii] = 0
            J[jj,ii+1] = d[0]
            J[jj,ii+2] = 0
            J[jj,ii+3] = d[1]
            J[jj,ii+4] = 0
            J[jj,ii+5] = 1
            J[jj,ii-6+4] = 0     # b[j-1,0]
            J[jj,ii-6+5] = -1    # b[j-1,1]
            J[jj] *= c_smooth
        else:
            d = source[j-1] - source[j]
            J[jj,ii] = 0
            J[jj,ii+1] = d[0]
            J[jj,ii+2] = 0
            J[jj,ii+3] = d[1]
            J[jj,ii+4] = 0
            J[jj,ii+5] = 1
            J[jj,ii-6+4] = 0     # b[j-1,0]
            J[jj,ii-6+5] = -1    # b[j-1,1]
            J[jj] *= c_smooth
            jj += 1
            d = source[j+1] - source[j]
            J[jj,ii] = 0
            J[jj,ii+1] = d[0]
            J[jj,ii+2] = 0
            J[jj,ii+3] = d[1]
            J[jj,ii+4] = 0
            J[jj,ii+5] = 1
            J[jj,ii+6+4] = 0     # b[j+1,0]
            J[jj,ii+6+5] = -1    # b[j+1,1]
            J[jj] *= c_smooth
'''

def Residual (source, target):
    global A, b, w, J, r
    global dd, nn
    src_len = source.shape[0]
    #r = np.zeros(src_len*9)
    r = np.zeros(src_len*5)

    dd = Deform(source)
    nn = Match(dd,target)
    diff = dd - nn
    print (diff*diff).sum()

    for j in range(src_len):
        #jj = j * 9
        jj = j * 5
        # E-fit
        r[jj] = diff[j,0] * c_fit
        r[jj+1] = diff[j,1] * c_fit
        # E-rigid
        r[jj+2] = (A[j,0,0]*A[j,1,0] + A[j,0,1]*A[j,1,1]) * c_rigid
        r[jj+3] = (1 - A[j,0,0]*A[j,0,0] - A[j,0,1]*A[j,0,1]) * c_rigid
        r[jj+4] = (1 - A[j,1,0]*A[j,1,0] - A[j,1,1]*A[j,1,1]) * c_rigid
'''
        # E-smooth
        if (j==0):
            d = np.dot((source[j+1]-source[j]),A[j]) + source[j] + b[j] - source[j+1] - b[j+1]
            r[jj+6] = d[0] * c_smooth
            r[jj+8] = d[1] * c_smooth
        elif (j==(src_len-1)):
            d = np.dot((source[j-1]-source[j]),A[j]) + source[j] + b[j] - source[j-1] - b[j-1]
            r[jj+5] = d[0] * c_smooth
            r[jj+7] = d[1] * c_smooth
        else:
            d = np.dot((source[j-1]-source[j]),A[j]) + source[j] + b[j] - source[j-1] - b[j-1]
            r[jj+5] = d[0] * c_smooth
            r[jj+7] = d[1] * c_smooth
            d = np.dot((source[j+1]-source[j]),A[j]) + source[j] + b[j] - source[j+1] - b[j+1]
            r[jj+6] = d[0] * c_smooth
            r[jj+8] = d[1] * c_smooth
'''

'''
MAIN PROGRAMME START
'''
target = GenerateTarget(target_len)
source = GenerateSource(source_len,rotate,translate_x,translate_y)
#ICP(source,target,0.1,20)
Initialise(source,target)

pt.ion()
while (True):
    Jacobian(source,target)
    Residual(source,target)
    JT = J.transpose()
    JT_J = np.dot(JT,J)
    JT_J_INV = np.linalg.inv(JT_J)
    JT_J_INV_JT = np.dot(JT_J_INV,JT)
    delta = np.dot(JT_J_INV_JT,r)
    print (r*r).sum()
    print np.array_str(r,85,4)
    print np.array_str(delta,80,4)

    # plot
    #pt.xlim(-2,8)
    #pt.ylim(-2,15)
    pt.plot(target[:,0],target[:,1],'r.')
    pt.plot(source[:,0],source[:,1],'bo')
    pt.plot(dd[:,0],dd[:,1],'b.')
    #pt.plot(ss[:,0],ss[:,1],'g^')
    pt.plot(nn[:,0],nn[:,1],'ro')
    pt.show()
    cont = raw_input("Continue ? ")
    if ((cont!="Y")and(cont!="y")):
        pt.close()
        break
    pt.close()
    for j in range(source.shape[0]):
        jj = j * 6
        A[j,0,0] += delta[jj]
        A[j,0,1] += delta[jj+1]
        A[j,1,0] += delta[jj+2]
        A[j,1,1] += delta[jj+3]
        b[j,0] += delta[jj+4]
        b[j,1] += delta[jj+5]

