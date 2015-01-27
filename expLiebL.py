#!/usr/bin/python

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import functools

#np.set_printoptions(precision = 4, suppress = True)

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

def getLargestW(Q_, R_, way, myWhich, myGuess):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForEigs = spspla.LinearOperator((chi * chi, chi * chi), 
                                         matvec = linOpWrap, dtype = 'float64')
    omega, X = spspla.eigs(linOpForEigs, k = 1, which = myWhich, tol = expS, 
                           maxiter = maxIter, v0 = myGuess, ncv = 10)

    return omega, X.reshape(chi, chi)

def solveLinSys(Q_, R_, way, myGuess, method = None):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    if(method == 'bicgstab'):
        S, info = spspla.lgmres(linOpForSol, np.zeros(chi * chi), tol = expS, 
                                x0 = myGuess, maxiter = maxIter, outer_k = 6)
    else:#if(method == 'gmres'):
        S, info = spspla.gmres(linOpForSol, np.zeros(chi * chi), tol = expS, 
                               x0 = myGuess, maxiter = maxIter)

    return info, S.reshape(chi, chi)

def tryGetBestSol(Q_, R_, way, guess):
    chi, chi = Q_.shape
    if(way == 'L'): guess = np.eye(chi, chi)
    if(guess == None): guess = np.random.rand(chi, chi) - .5
    #print "GUESS\n", guess
    guess = guess.reshape(chi * chi)

    try:
        joke, sol = getLargestW(Q_, R_, way, 'SM', guess)
    except:
        print "getLargestW: eigs failed, trying random seed"
        guess = np.random.rand(chi * chi) - .5

        try:
            joke, sol = getLargestW(Q_, R_, way, 'SM', guess)
        except:
            print "getLargestW: eigs failed, trying bicgstab"
            sol = np.random.rand(chi, chi) - .5

    print "SOL\n", sol
    guess = sol.reshape(chi * chi).real

    try:
        joke, sol = solveLinSys(Q_, R_, way, guess, 'bicgstab')
    except (ArpackError, ArpackNoConvergence):
        print "solveLinSys: bicgstab failed, trying gmres"
        guess = sol.reshape(chi * chi) if joke > 0 else guess

        try:
            joke, sol = solveLinSys(Q_, R_, way, guess, 'gmres')
        except (ArpackError, ArpackNoConvergence):
            print "solveLinSys: gmres failed, lame solution"
            sol = 0.5 * (sol + guess)

    return sol

def fixPhase(X):
    Y = X / np.trace(X)
    norm = 1.0 #np.sqrt(np.trace(adj(Y).dot(Y)))
    Y = Y / norm

    return Y

def leftNormalization(K_, R_, guess):
    chi, chi = K_.shape
    Q_ = K_ - .5 * (adj(R_).dot(R_))
    print "K\n", K_, "\nR\n", R_, "\nQ\n", Q_

    #l_ = tryGetBestSol(Q_, R_, 'L', guess)
    #l_ = fixPhase(l_)
    #fac = np.trace(l_) / chi
    #print "l", np.trace(l_), "\n", l_/fac

    r_ = tryGetBestSol(Q_, R_, 'R', guess)
    r_ = fixPhase(r_)
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

def rhoVersions(rho_):
    eval_, evec_ = spla.eigh(rho_)
    print "lambda", reduce(lambda x, y: x+y, eval_), eval_

    eval_ = abs(eval_)
    evalI = 1. / eval_
    evalSr = np.sqrt(eval_)
    evalSrI = 1. / np.sqrt(eval_)
    print "evalI  ", evalI, "\nevalSr ", evalSr, "\nevalSrI", evalSrI

    ULUd = evec_.dot(np.diag(eval_)).dot(adj(evec_))
    print "r - UdU+\n", rho_ - ULUd

    rhoI_ = evec_.dot(np.diag(evalI)).dot(adj(evec_))
    rhoSr_ = evec_.dot(np.diag(evalSr)).dot(adj(evec_))
    rhoSrI_ = evec_.dot(np.diag(evalSrI)).dot(adj(evec_))
    print "rr^-1\n", rho_.dot(rhoI_)
    print "rSrrSr^-1\n", rhoSr_.dot(rhoSrI_)

    return rhoI_, rhoSr_, rhoSrI_

def getQrhsQ(Q_, R_, way, rho_):
    chi, chi = Q_.shape
    commQR = comm(Q_, R_)
    #RR = R_.dot(R_)
    Id = np.eye(chi, chi)

    kinE = 1./(2. * m) * supOp(commQR, commQR, way, Id)
    potE = v * supOp(R_, R_, way, Id)
    intE = w * supOp(R_, R_, way, supOp(R_, R_, way, Id))#supOp(RR, RR, way, Id)
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

def calcF(Q_, R_, way, rho_, guess):
    chi, chi = Q_.shape
    rhs = - getQrhsQ(Q_, R_, way, rho_)

    if(guess == None): guess = np.random.rand(chi * chi) - .5
    guess = guess.reshape(chi * chi)

    linOpWrap = functools.partial(linOpForF, Q_, R_, way, rho_)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    try:
        F, info = spspla.lgmres(linOpForSol, rhs, tol = expS, x0 = guess, 
                                maxiter = maxIter, outer_k = 6)
    except (ArpackError, ArpackNoConvergence):
        print "calcF: bicgstab failed, trying gmres\n"
        guess = F if info > 0 else guess

        try:
            F, info = spspla.bicgstab(linOpForSol, rhs, tol = expS, x0 = guess, 
                                      maxiter = maxIter)
        except (ArpackError, ArpackNoConvergence):
            print "calcF: gmres failed, taking lame solution\n"
            F = F if info > 0 else guess

    print "F\n", F.reshape(chi, chi)
    return F.reshape(chi, chi)

def calcYstar(Q_, R_, F_, rho_, rhoI_, rhoSr_, rhoSrI_):
    fContrib = - R_.dot(F_).dot(rhoSr_) + F_.dot(R_).dot(rhoSr_)

    #commQR = comm(Q_, R_)
    #partOne = adj(Q_).dot(commQR).dot(rhoSr_) 
    #- commQR.dot(rho_).dot(adj(Q_)).dot(rhoSrI_)
    #partTwo = R_.dot(commQR).dot(rho_).dot(adj(R_)).dot(rhoSrI_) 
    #- R_.dot(adj(R_)).dot(commQR).dot(rhoSr_)

    common = comm(Q_, R_).dot(rho_)
    partOne = comm(adj(Q_), common).dot(rhoSrI_)
    partTwo = R_.dot(comm(common, adj(R_))).dot(rhoSrI_)

    kContrib = (partOne - partTwo) / (2. * m)
    pContrib = v * (R_.dot(rhoSr_))

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
    conver = np.trace(adj(Ystar_).dot(Ystar_))
    print "Vstar\n", Vstar_, "\nWstar\n", Wstar_, "\nConver", conver,

    return Vstar_, Wstar_

def doUpdateQandR(K_, R_, Wstar_):
    tmp = adj(R_).dot(Wstar_) - adj(Wstar_).dot(R_)

    R__ = R_ - dTau * Wstar_
    K__ = K_ + .5 * dTau * tmp
    #Q__ = K__ - .5 * (adj(R__).dot(R__))
    #print "K\n", K__, "\nR\n", R__, "\nQ\n", Q__

    return K__, R__

def calcQuantities(Q_, R_, rho_, way):
    tmp = comm(Q_, R_)
    RR = R_.dot(R_)
    #if(way == 'L'): way = 'R' else: way = 'L'
    way = 'R' if way == 'L' else 'L'

    density = np.trace(R_.dot(rho_).dot(adj(R_)))
    eFixedN = np.trace(tmp.dot(rho_).dot(adj(tmp)) / (2. * m) 
                       + w * RR.dot(rho_).dot(adj(RR)))
    print "<n>", density, "e", eFixedN, 

    density = np.trace(supOp(R_, R_, way, rho_))
    eFixedN = np.trace(supOp(tmp, tmp, way, rho_) / (2. * m) 
                       + w * supOp(RR, RR, way, rho_))
    print "e/<n>^3", eFixedN/density**3, "g/<n>", w/density



"""
Main...
"""

#np.random.seed(2)
xi = 40
expS = 1.e-6
maxIter = 9000
dTau = .01
K = 1 * (np.random.rand(xi, xi) - .5) #+ 1j * np.zeros((xi, xi))
K = .5 * (K - adj(K))
R = 1 * (np.random.rand(xi, xi) - .5) #+ 1j * np.zeros((xi, xi))
rho = fixPhase(np.random.rand(xi, xi) - .5)
F = np.random.rand(xi, xi) - .5

m, v, w = .5, -.5, 2.

I = 0
while (I != maxIter):
    print "\t\t\t\t\t\t\t############# ITERATION", I, "#############"

    Q, rho = leftNormalization(K, R, rho)
    #Q, rho = rightNormalization(K, R, ...)

    rhoI, rhoSr, rhoSrI = rhoVersions(rho)

    F = calcF(Q, R, 'L', rho, F)

    Ystar = calcYstar(Q, R, F, rho, rhoI, rhoSr, rhoSrI)

    Vstar, Wstar = getUpdateVandW(R, rhoSrI, Ystar)

    calcQuantities(Q, R, rho, 'L')

    K, R = doUpdateQandR(K, R, Wstar)

    I += 1
