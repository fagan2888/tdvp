#/usr/bin/python

import numpy as np
import cmath

length = 4
chi = 1
d = 2

A = np.random.rand(chi*d*chi)
A = A.astype(complex)
A = A.reshape((chi * d, chi))

Q, R = np.linalg.qr(A)

print "A = ", A
print "Q = ", Q

A = Q.copy()
A = A.reshape((chi, d, chi))

print "A = ", A
