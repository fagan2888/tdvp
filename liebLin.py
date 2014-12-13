#!/usr/bin/python

import numpy as np
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import functools

np.set_printoptions(precision = 4, suppress = True)

adj = lambda X: np.transpose(np.conjugate(X))
comm = lambda X, Y: np.dot(X, Y) - np.dot(Y, X)

def supOp(A, B, way, X):
    if(way == 'R'):
        return A.dot(X).dot(adj(B))
    else:
        return adj(B).dot(X).dot(A)

def linOpForT(Q_, R_, way, X):
    chi, chi = Q_.shape
    X = X.reshape(chi, chi)
    Id = np.eye(chi, chi)

    XTX = supOp(Q_, Id, way, X) + supOp(Id, Q_, way, X) + supOp(R_, R_, way, X)

    return XTX.reshape(chi * chi)

def getLargestW(Q_, R_, way):
    """Returns density matrix by diagonalizing.

    Should it be SR or SM?
    """
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForEigs = spspla.LinearOperator((chi * chi, chi * chi), 
                                         matvec = linOpWrap, dtype = 'float64')
    omega, X = spspla.eigs(linOpForEigs, k = 1, which = 'SR', tol = expS, 
                           maxiter = maxIter)
    X = X.reshape(chi, chi)

    return omega, X

def solveLinSys(Q_, R_, way):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    guess = np.random.rand(chi * chi) - .5
    S, info = spspla.bicgstab(linOpForSol, np.zeros(chi * chi), tol = expS, 
                              x0 = guess, maxiter = maxIter)
    if(info != 0): print "\nWARNING: bicgstab failed!\n"; exit()
    S = S.reshape(chi, chi)

    return S

def fixPhase(X):
    Y = X / np.trace(X)
    norm = 1.0 #np.sqrt(np.trace(adj(Y).dot(Y)))
    Y = Y / norm

    return Y

def leftNormalization(K_, R_):
    chi, chi = K_.shape
    Q_ = - K_ - .5 * (adj(R_).dot(R_))
    print "K\n", K_, "\nR\n", R_, "\nQ\n", Q_

    #w, l_ = getLargestW(Q_, R_, 'L')
    l_ = solveLinSys(Q_, R_, 'L')
    r_ = solveLinSys(Q_, R_, 'R')
    l_ = fixPhase(l_)
    r_ = fixPhase(r_)
    fac = np.trace(l_) / chi
    print "l", np.trace(l_), "\n", l_/fac
    print "r", np.trace(r_), "\n", r_

    return Q_, r_

def rightNormalization(K_, R_):
    chi, chi = K_.shape
    Q_ = - K_ - .5 * (R_.dot(adj(R_)))
    print "K\n", K_, "\nR\n", R_, "\nQ\n", Q_

    l_ = solveLinSys(Q_, R_, 'L')
    r_ = solveLinSys(Q_, R_, 'R')
    l_ = fixPhase(l_)
    r_ = fixPhase(r_)
    fac = np.trace(r_) / chi
    print "l", np.trace(l_), "\n", l_
    print "r", np.trace(r_), "\n", r_/fac

    return Q_, l_

def getQrhsQ(Q_, R_, way, rho):
    chi, chi = Q_.shape
    commQR = comm(Q_, R_)
    RR = R_.dot(R_)
    Id = np.eye(chi, chi)

    kinE = 1./(2. * m) * supOp(commQR, commQR, way, Id)
    potE = v * supOp(R_, R_, way, Id)
    intE = w * supOp(RR, RR, way, Id)
    ham = kinE + potE + intE

    energy = np.trace(ham.dot(rho))
    rhs = ham - (energy * Id)
    print "Energy density", energy

    return rhs.reshape(chi * chi)

def linOpForF(Q_, R_, way, rho, X):
    chi, chi = Q_.shape
    X = X.reshape(chi, chi)
    Id = np.eye(chi, chi)

    FTF = linOpForT(Q_, R_, way, X) - (np.trace(X.dot(rho)) * Id)

    return FTF.reshape(chi * chi)


"""
Main...
"""
np.random.seed(0)

xi = 6
expS = 1.e-10
maxIter = 900
K = np.random.rand(xi, xi) -.5 #+ 1j * np.zeros((xi, xi))
K = .5 * (K - adj(K))
R = np.random.rand(xi, xi) -.5 #+ 1j * np.zeros((xi, xi))

Q, r = leftNormalization(K, R)
Q, l = rightNormalization(K, R)

eval, evec = spla.eig(r)
print "lambda", reduce(lambda x,y: x+y, eval), eval.real
