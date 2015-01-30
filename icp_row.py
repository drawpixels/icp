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

'''
Return the index of the closest point
'''
def closest (set, pt):
    disp = set - pt
    dist = (disp*disp).sum(1)
    return np.argmin(dist)

def dist (s1, s2):
    disp = s1 - s2
    dist = np.sqrt((disp*disp).sum(1))
    return dist.sum(0)

# Target dataset
x = np.linspace(0,2*np.pi,target_len)
target = np.array([x,np.sin(x)]).T

# Source dataset
#-- Picking random samples from the source (i.e. sample points align)
#-- samples = np.sort(np.random.choice(target_len,source_len,replace=False))
#-- source = np.zeros([2,source_len])
#-- for i in range(source_len):
#--     source[:,i] = target[:,samples[i]]

#-- Create random sample positions over the same range (i.e. sample points don't align)
x = np.sort(np.random.random(source_len))*2*np.pi
source = np.array([x,np.sin(x)]).T

# Rotate & translate source dataset
theta = 1.0
rot = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
tsl = np.array([2.7,1.0])
#transform = np.dot(source,rot)
#source[0] = transform[0] + 2.7
#source[1] = transform[1] + 1.0
source = np.dot(source,rot) + tsl

ss = source.copy()
nn = np.zeros([source_len,2])
err = -1
run = True

pt.ion()
while (run):
    ref = ss.copy()
    for i in range(source_len):
        cp = closest(target,ss[i])
        nn[i] = target[cp]

    # ICP
    t_mean = nn.mean(0)
    t1 = nn - t_mean
    s_mean = source.mean(0)
    s1 = source - s_mean
    W = np.zeros([2,2])
    for i in range(source_len):
        W += np.dot(np.array([t1[i]]).T, np.array([s1[i]]))
    #print W
    U, S, VT = np.linalg.svd(W)
    #- print U
    #- print VT

    R = np.dot(VT, U)
    t = t_mean - np.dot(s_mean, R)
    print "R = \n", R
    print "t = \n", t
    #print np.dot(s_mean, R)
    ss = np.dot(source, R) + t
    err = dist(nn,ss)
    print "err = ", err

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
        run = False
    pt.close()

