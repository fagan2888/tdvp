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

    return omega, X.reshape(chi, chi)

def solveLinSys(Q_, R_, way):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    guess = np.random.rand(chi * chi) - .5
    S, info = spspla.bicgstab(linOpForSol, np.zeros(chi * chi), tol = expS, 
                              x0 = guess, maxiter = maxIter)
    if(info != 0): print "\nWARNING: bicgstab failed!\n"; exit()

    return S.reshape(chi, chi)

def fixPhase(X):
    Y = X / np.trace(X)
    norm = 1.0 #np.sqrt(np.trace(adj(Y).dot(Y)))
    Y = Y / norm

    return Y

def leftNormalization(K_, R_):
    chi, chi = K_.shape
    Q_ = K_ - .5 * (adj(R_).dot(R_))
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
    Q_ = K_ - .5 * (R_.dot(adj(R_)))
    print "K\n", K_, "\nR\n", R_, "\nQ\n", Q_

    l_ = solveLinSys(Q_, R_, 'L')
    r_ = solveLinSys(Q_, R_, 'R')
    l_ = fixPhase(l_)
    r_ = fixPhase(r_)
    fac = np.trace(r_) / chi
    print "l", np.trace(l_), "\n", l_
    print "r", np.trace(r_), "\n", r_/fac

    return Q_, l_

def getQrhsQ(Q_, R_, way, rho_):
    chi, chi = Q_.shape
    commQR = comm(Q_, R_)
    RR = R_.dot(R_)
    Id = np.eye(chi, chi)

    kinE = 1./(2. * m) * supOp(commQR, commQR, way, Id)
    potE = v * supOp(R_, R_, way, Id)
    intE = w * supOp(RR, RR, way, Id)
    ham = kinE + potE + intE

    energy = np.trace(ham.dot(rho_))
    rhs = ham - (energy * Id)
    print "Energy density", energy

    return rhs.reshape(chi * chi)

def linOpForF(Q_, R_, way, rho_, X):
    chi, chi = Q_.shape
    X = X.reshape(chi, chi)
    Id = np.eye(chi, chi)
    shift = np.trace(X.dot(rho_))

    FTF = linOpForT(Q_, R_, way, X).reshape(chi, chi) - (shift * Id)

    return FTF.reshape(chi * chi)

def calcF(Q_, R_, way, rho_):
    chi, chi = Q_.shape
    rhs = - getQrhsQ(Q_, R_, way, rho_)

    linOpWrap = functools.partial(linOpForF, Q_, R_, way, rho_)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    guess = np.random.rand(chi * chi) - .5
    F, info = spspla.bicgstab(linOpForSol, rhs, tol = expS, 
                              x0 = guess, maxiter = maxIter)
    if(info != 0): print "\nWARNING: bicgstab failed!\n"; exit()

    return F.reshape(chi, chi)

def rhoVersions(rho_):
    eval_, evec_ = spla.eig(rho_)
    print "lambda", reduce(lambda x, y: x+y, eval_), eval_.real

    eval_ = eval_.real
    evalI = 1. / eval_
    evalSr = map(np.sqrt, eval_)
    evalSrI = 1. / np.asarray(evalSr)
    print "evalI  ", evalI, "\nevalSr ", evalSr, "\nevalSrI", evalSrI

    rhoI_ = evec_.dot(np.diag(evalI)).dot(adj(evec_))
    rhoSr_ = evec_.dot(np.diag(evalSr)).dot(adj(evec_))
    rhoSrI_ = evec_.dot(np.diag(evalSrI)).dot(adj(evec_))

    return rhoI_, rhoSr_, rhoSrI_

def calcYstar(Q_, R_, F_, rho_, rhoI_, rhoSr_, rhoSrI_):
    fContrib = - R_.dot(F_).dot(rhoSr_) + F_.dot(R_).dot(rhoSr_)

    common = comm(Q_, R_).dot(rho_)
    partOne = comm(adj(Q_), common).dot(rhoSrI_)
    partTwo = R_.dot(comm(common, adj(R_))).dot(rhoSrI_)

    kContrib = (partOne - partTwo) / (2. * m)
    pContrib = v * (R_.dot(rhoSr))

    RR = R_.dot(R_)
    partOne = RR.dot(rho_).dot(adj(R_)).dot(rhoSrI_)
    partTwo = adj(R_).dot(RR).dot(rhoSr_)

    iContrib = w * (partOne + partTwo)

    Ystar_ = fContrib + kContrib + pContrib + iContrib
    print "Ystar\n", Ystar_

    return Ystar_

def getUpdateVandW(R_, rhoSrI_, Ystar_):
    Vstar_ = - adj(R_).dot(Ystar_).dot(rhoSrI_)
    Wstar_ = Ystar_.dot(rhoSrI_)
    print "Vstar\n", Vstar_, "\nWstar\n", Wstar_

    return Vstar_, Wstar_


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

m, v, w = .5, 1., 2.

Q, rho = leftNormalization(K, R)
#Q, rho = rightNormalization(K, R)

rhoI, rhoSr, rhoSrI = rhoVersions(rho)

F = calcF(Q, R, 'L', rho)

Ystar = calcYstar(Q, R, F, rho, rhoI, rhoSr, rhoSrI)

Vstar, Wstar = getUpdateVandW(R, rhoSrI, Ystar)
