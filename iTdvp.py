#!/usr/bin/python

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import cmath

np.set_printoptions(suppress=True, precision=3)

def powerMethod(MPS):
    chir, chic, aux = MPS.shape
    X = np.random.rand(chir, chic) - .5
    maxIterForPow = 1000
    print X

    for q in range(maxIterForPow):
        Y = linearOpForR(MPS, X)
        YH = np.transpose(np.conjugate(Y), (1, 0))
        YHY = np.tensordot(YH, Y, axes=([1, 0]))
        norm = np.sqrt(np.trace(YHY))
        X = Y / norm
        print q, norm

def linearOpForR(MPS, X):
    AX = np.tensordot(MPS, X, axes=([1,0]))
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    AXAH = np.tensordot(AX, AH, axes=([1,2], [2,0]))
    #AA = np.tensordot(A, MPS, axes=([2, 2]))
    #AXA = np.tensordot(AA, X, axes=([1, 3], [0, 1]))

    return AXAH

def symmNormalization(MPS, chir, chic):
    """Returns a symmetric normalized MPS.

    There is the doubt of how to reshape the tensor in order to get
    the SVD. (Look at the first two lines of the sub.) My guess is 
    that in an infinite system it does not make a difference.
    """
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    AA = np.tensordot(AH, MPS, axes=([2,2]))
    AA = np.reshape(AA, (chir*chic, chir*chic))
    print "AA =\n", AA#, AA == AA.T
    omega, R = spspla.eigs(AA, k=1, which='LR')
    print "w(1) =", omega
    Aval, Avec = spla.eig(AA)
    print Aval
    powerMethod(MPS)
    #print getRasVec(np.eye(chir, chic))
    #linearOp = spspla.LinearOperator((chir, chic), matvec=getRasVec)

def buildLocalH():
    """Builds local hamiltonian (d x d)-matrix.

    Returns local hamiltonian for a translation invariant system.
    The local H is a (d, d, d, d)-rank tensor.
    """
    localH = np.zeros((d, d, d, d))
    localH[0,0,0,0] = localH[1,1,1,1] = 1./4.
    localH[0,1,0,1] = localH[1,0,1,0] = 1./4.
    localH[1,0,0,1] = localH[0,1,1,0] = 1./2.

    return localH

def buildHElements(MPS, H):
    """Builds the matrix elements of the hamiltonian.

    Returns a tensor of the form C[a, b, s, t].
    """
    AA = np.tensordot(MPS, MPS, axes=([1,0]))
    tmp = np.tensordot(H, AA, axes=([2,3], [1,3]))
    C = np.transpose(tmp, (2, 3, 0, 1))

    return C

def calcHmeanval(MPS, Ssqrt, C):
    S = np.linalg.svd(MPS.reshape(xir, xic*d), False, False)
    print S
    pass


"""Main...
"""
length = 4
d = 2
xi = 3

xir = xic = xi
theA = np.random.rand(xir, xic, d) - .5

#theH = buildLocalH()
#print "theH\n", theH.reshape(d*d, d*d)

symmNormalization(theA, xir, xic)
exit()
print "theGamma =", map(np.shape, theGamma)

theC = buildHElements(theA, theH)
print "theC =", theC.shape

calcHmeanval(theA, theC)
