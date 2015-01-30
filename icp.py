'''
ICP v2

Very Basic ICP
2 sine curves with only rotation and translation
Source curve has random samples

TODO:
 - closest point is not accurate

'''

import numpy as np
import matplotlib.pyplot as pt
# import matplotlib.cm as cm
# import math as m

def closest (set, pt):
    diff = set - pt
    dist = (diff*diff).sum(0)
    return np.argmin(dist)

# Target dataset
target_len = 50
x = np.linspace(0,2*np.pi,target_len)
target = np.array([x,np.sin(x)])

# Source dataset
source_len = 20
#-- Picking random samples from the source (i.e. sample points align)
#-- samples = np.sort(np.random.choice(target_len,source_len,replace=False))
#-- source = np.zeros([2,source_len])
#-- for i in range(source_len):
#--     source[:,i] = target[:,samples[i]]

#-- Create random sample positions over the same range (i.e. sample points don't align)
x = np.sort(np.random.random(source_len))*2*np.pi
source = np.array([x,np.sin(x)])

# Rotate & translate source dataset
theta = 1.0
rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
transform = np.dot(rot,source)
source[0] = transform[0] + 2.7
source[1] = transform[1] + 1.0

ss = source.copy()

pt.ion()
ICP = np.zeros([2,source_len])
run = True
while (run):
    nn = np.zeros([2,source_len])
    for i in range(source_len):
        p = np.array([source[:,i]]).T
        nn[:,i] = target[:,closest(target,p)]

    # ICP
    t_mean = np.array([nn.mean(1)]).T
    t1 = nn - t_mean
    s_mean = np.array([source.mean(1)]).T
    s1 = source - s_mean
    #- print t_mean
    #- print s_mean
    W = np.zeros([2,2])
    for i in range(source_len):
        W += np.dot(np.array([t1[:,i]]).T , np.array([s1[:,i]]))
    #- print W
    U, S, VT = np.linalg.svd(W)
    #- print U
    #- print V

    R = np.dot(U , VT)
    t = t_mean - np.dot(R , s_mean)
    print "R = \n", R
    print "t = \n", t
    ICP = np.dot(R , source) + t

    # plot
    pt.xlim(-2,8)
    pt.ylim(-2,8)
    pt.plot(target[0],target[1],'r.')
    pt.plot(ss[0],ss[1],'bo')
    pt.plot(nn[0],nn[1],'ro')
    #- pt.plot(target[0,nn],target[1,nn],'ro')
    #- pt.plot(source[0,s],source[1,s],'bo')
    pt.plot(source[0],source[1],'b.')
    pt.plot(ICP[0],ICP[1],'g^')
    pt.show()

    cont = raw_input("Continue ? ")
    if ((cont=="Y")or(cont=="y")):
        for i in range(source_len):
            source[:,i] = ICP[:,i]
        #-- source = ICP.copy
    else:
        run = False
    pt.close()

