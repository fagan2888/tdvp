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

def linOpForSys(Q, R, way, X):
    chi = Q.shape
    X = X.reshape(chi, chi)
    Id = np.eye(chi, chi)

    XTX = SupOp(Q, Id, way, X) + SupOp(Id, Q, way, X) + SupOp(R, R, way, X)
    return XTX.reshape(chi * chi)

def linearOpForR(q, r, f):
    chi, chi = r.shape
    f = f.reshape(chi, chi)
    fRdag = np.tensordot(f, np.conjugate(r.T), axes=([1,0]))

    tmp1 = np.tensordot(q, f, axes=([1,0]))
    tmp2 = np.tensordot(f, np.conjugate(q.T), axes=([1,0]))
    tmp3 = np.tensordot(r, fRdag, axes=([1,0]))

    Tf = tmp1 + tmp2 + tmp3
    return Tf.reshape(chi * chi)

def linearOpForL(q, r, f):
    chi, chi = q.shape
    f = f.reshape(chi, chi)
    RdagF = np.tensordot(np.conjugate(r.T), f, axes=([1,0]))

    tmp1 = np.tensordot(f, q, axes=([1,0]))
    tmp2 = np.tensordot(np.conjugate(q.T), f, axes=([1,0]))
    tmp3 = np.tensordot(RdagF, r, axes=([1,0]))

    fT = tmp1 + tmp2 + tmp3
    return fT.reshape(chi * chi)

def getLargestW(q, r, way):
    chi, chi = q.shape

    if(way == 'R'):
        linOpWrapped = functools.partial(linearOpForR, q, r)
    else:
        linOpWrapped = functools.partial(linearOpForL, q, r)

    linOpForEigs = spspla.LinearOperator((chi * chi, chi * chi), 
                                         matvec = linOpWrapped, 
                                         dtype = 'float64')
    omega, X = spspla.eigs(linOpForEigs, k = 1, which = 'LR', tol = expS, 
                           maxiter = maxIter)
    X = X.reshape(chi, chi)

    return omega, X

def solveLinSys(q, r, way):
    chi, chi = q.shape

    if(way == 'R'):
        linOpWrapped = functools.partial(linearOpForR, q, r)
    else:
        linOpWrapped = functools.partial(linearOpForL, q, r)

    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrapped, 
                                        dtype = 'float64')
    guess = np.random.rand(chi * chi) - .5

    S, info = spspla.bicgstab(linOpForSol, np.zeros(chi * chi), tol = expS, 
                              x0 = guess, maxiter = maxIter)
    if(info != 0): print "\nWARNING: bicgstab failed!\n"; exit()
    S = S.reshape(chi, chi)

    return S

def fixPhase(X):
    Y = X / np.trace(X)
    norm = np.tensordot(np.transpose(np.conjugate(Y)), Y, axes=([1,0]))
    norm = np.sqrt(np.trace(norm))
    Y = Y / norm

    return Y

def leftNormalization(chi):
    K = np.random.rand(chi, chi) -.5 #+ 1j * np.zeros((chi, chi))
    K = .5 * (K - adj(K))
    R = np.random.rand(chi, chi) -.5 #+ 1j * np.zeros((chi, chi))

    Q = - K - .5 * (adj(R).dot(R))
    print "K\n", K, "\nR\n", R, "\nQ\n", Q

    l = solveLinSys(Q, R, 'L')
    r = solveLinSys(Q, R, 'R')
    fac = np.trace(l) / chi
    print "L", np.trace(l), "\n", l/fac
    print "R", np.trace(r), "\n", r

    eval, evec = spla.eig(r)
    print "lambda", eval

def rightNormalization(chi):
    K = np.random.rand(chi, chi) -.5 #+ 1j * np.zeros((chi, chi))
    K = 0.5 * (K - np.conjugate(K.T))

    R = np.random.rand(chi, chi) -.5 #+ 1j * np.zeros((chi, chi))
    Rdag = np.conjugate(R.T)

    Q = - K - .5 * (np.tensordot(R, Rdag, axes=([1,0])))
    print "K\n", K, "\nR\n", R, "\nQ\n", Q

    l = solveLinSys(Q, R, 'L')
    r = solveLinSys(Q, R, 'R')
    fac = np.trace(r) / chi
    print "L", np.trace(l), "\n", l
    print "R", np.trace(r), "\n", r/fac

    eval, evec = spla.eig(l)
    print "lambda", eval


"""
Main...
"""
xi = 3
expS = 1.e-10
maxIter = 500

np.random.seed(0)

leftNormalization(xi)
rightNormalization(xi)
