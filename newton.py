'''
newton.py
'''

import numpy as np
import matplotlib.pyplot as pt
# import matplotlib.cm as cm
# import math as m

'''
Global Constants
'''
data_range = 5
data_len = 20
h = 0.0001
#-- Target parameters
A = 5
B = 1
C = 10
D = 2
#-- Initial guess
a = 2
b = 1
c = 8
d = 1

'''
Generate target dataset
'''
def GenerateTarget (x):
    dset = A*np.sin(B*x) + C*np.cos(D*x)
    return dset

def Estimate (x, a, b, c, d):
    est = a*np.sin(b*x) + c*np.cos(d*x)
    return est    

def Jacobian (x, a, b, c, d):
    J = np.zeros([x.shape[0],4])
    J[:,0] = np.sin(b*x)
    J[:,1] = a*x*np.cos(b*x)
    J[:,2] = np.cos(d*x)
    J[:,3] = -c*x*np.sin(d*x)
    #J[:,0] = (((a+h)*np.sin(b*x) + c*np.cos(d*x))-((a-h)*np.sin(b*x) + c*np.cos(d*x))) / (2*h)
    #J[:,1] = ((a*np.sin((b+h)*x) + c*np.cos(d*x))-(a*np.sin((b-h)*x) + c*np.cos(d*x))) / (2*h)
    #J[:,2] = ((a*np.sin(b*x) + (c+h)*np.cos(d*x))-(a*np.sin(b*x) + (c-h)*np.cos(d*x))) / (2*h)
    #J[:,3] = ((a*np.sin(b*x) + c*np.cos((d+h)*x))-(a*np.sin(b*x) + c*np.cos((d-h)*x))) / (2*h)
    #print J[:,0] - np.sin(b*x)
    #print J[:,1] - a*x*np.cos(b*x)
    #print J[:,2] - np.cos(d*x)
    #print J[:,3] + c*x*np.sin(d*x)
    return J

'''
MAIN PROGRAMME START
'''
x = np.linspace(0,data_range, data_len)
target = GenerateTarget(x)

while (True):
    est = Estimate(x,a,b,c,d)
    r = target - est
    J = Jacobian(x,a,b,c,d)
    JT = J.transpose()
    JT_J = np.dot(JT,J)
    JT_J_INV = np.linalg.inv(JT_J)
    JT_J_INV_JT = np.dot(JT_J_INV,JT)
    delta = np.dot(JT_J_INV_JT,r)
    print "[",a,b,c,d,"]",(r*r).sum(),delta

    cont = raw_input("Continue ? ")
    if ((cont=="N")or(cont=="n")):
        break

    a += delta[0]
    b += delta[1]
    c += delta[2]
    d += delta[3]

