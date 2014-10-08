#/usr/bin/python

import numpy as np
from scipy import linalg
import cmath

np.set_printoptions(suppress=True, precision=3)

length = 4
chi = 3
d = 2

chir = [chi for n in range(length)]
chic = [chi for n in range(length)]
#chir[0] = chic[length-1] = 1
print "chir =", chir
print "chic =", chic

MPS = [np.random.rand(chir[n] * d * chic[n]) for n in range(length)]
print "MPS =", MPS

for n in range(length):
    MPS[n] = MPS[n].reshape((chir[n] * d, chic[n]))
    Q, R = linalg.qr(MPS[n], mode='economic')
    MPS[n] = Q.copy()
    print "Q[",n,"] =", Q
    print "R[",n,"] =", R

    if(n != length-1):
        MPS[n+1] = MPS[n+1].reshape((chir[n+1], chic[n+1] * d))
        MPS[n+1] = np.dot(R, MPS[n+1])
        print "shape =", MPS[n+1].shape, chir[n+1], chic[n+1], d
        MPS[n+1] = MPS[n+1].reshape((chir[n+1] * d * chic[n+1]))
    del Q, R

for n in range(length):
    print "MPS[",n,"] =", MPS[n]
    i = np.random.randint(0, chic[n])
    j = np.random.randint(0, chic[n])
    dot = np.dot(MPS[n][:,i], MPS[n][:,j])
    print "dot =", i, j, round(dot, 10)

"""
del R
print R

exit()

A = np.random.rand(chi*d*chi)
#A = A.astype(complex)
A = A.reshape((chi * d, chi))

Q, R = linalg.qr(A, mode='economic')

zrows = np.zeros((chi*d-chi, chi))
R = np.vstack((R, zrows))

print "A =", A#.shape
print "Q =", Q#.shape
print "R =", R#.shape

B = np.random.rand(chi*d*chi)
#B = A.astype(complex)
B = B.reshape((chi * d, chi))

print "B =",B

B = R * B

Q2, R2 = linalg.qr(B, mode='economic')

R2 = np.vstack((R2, zrows))

print "B =", B#.shape
print "Q2 =", Q2#.shape
print "R2 =", R2#.shape

C = np.random.rand(chi*d*chi)
#B = A.astype(complex)
C = C.reshape((chi * d, chi))

print "C =",C

C = R2 * C

Q3, R3 = linalg.qr(C, mode='economic')

R3 = np.vstack((R3, zrows))

print "C =", C#.shape
print "Q3 =", Q3#.shape
print "R3 =", R3#.shape

exit()

A = Q.copy()
"""
