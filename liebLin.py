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

def getLargestW(Q_, R_, way, myWhich = 'SR', myGuess = None):
    """Returns density matrix by diagonalizing.

    Should it be SR or SM?
    """
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForEigs = spspla.LinearOperator((chi * chi, chi * chi), 
                                         matvec = linOpWrap, dtype = 'float64')
    omega, X = spspla.eigs(linOpForEigs, k = 1, which = myWhich, tol = expS, 
                           maxiter = maxIter, v0 = myGuess)

    return omega, X.reshape(chi, chi)

def solveLinSys(Q_, R_, way, myGuess = None, method):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    if(method == 'bicgstab'):
        S, info = spspla.bicgstab(linOpForSol, np.zeros(chi * chi), tol = expS, 
                                  x0 = myGuess, maxiter = maxIter)
    else:#if(method == 'gmres'):
        S, info = spspla.gmres(linOpForSol, np.zeros(chi * chi), tol = expS, 
                                  x0 = myGuess, maxiter = maxIter)

    return info, S.reshape(chi, chi)

def tryGetBestSol(Q_, R_, way, guess = None):
    chi, chi = Q_.shape
    if(guess != None): guess = np.random.rand(chi * chi) - .5
    guess = guess.reshape(chi * chi)

    try:
        joke, sol = solveLinSys(Q_, R_, way, guess, 'bicgstab')
    except (ArpackError, ArpackNoConvergence):
        print "solveLinSys failed, trying getLargestW\n"
        sol = sol.reshape(chi * chi) if joke > 0 else guess

        try:
            joke, sol = solveLinSys(Q_, R_, way, guess, 'gmres')
        except (ArpackError, ArpackNoConvergence):
            print "getLargestW failed; restarting\n"
            sol = sol.reshape(chi * chi)

            try:
                joke, sol = getLargestW(Q_, R_, way, guess)
            except (ArpackError, ArpackNoConvergence):
                print "Continuing with lame solution\n"
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

    l_ = tryGetBestSol(Q_, R_, 'L', guess)
    r_ = tryGetBestSol(Q_, R_, 'R', guess)
    l_ = fixPhase(l_)
    r_ = fixPhase(r_)
    fac = np.trace(l_) / chi
    print "l", np.trace(l_), "\n", l_/fac
    print "r", np.trace(r_), "\n", r_

    RR = R_.dot(R_)
    Rd = adj(R_)
    print "(R^+)^2 - (R^2)^+\n", Rd.dot(Rd) - adj(RR)

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
    eval_, evec_ = spla.eig(rho_)
    print "lambda", reduce(lambda x, y: x+y, eval_), eval_.real

    eval_ = eval_.real
    evalI = 1. / eval_
    evalSr = map(np.sqrt, eval_)
    evalSrI = 1. / np.asarray(evalSr)
    print "evalI  ", evalI, "\nevalSr ", evalSr, "\nevalSrI", evalSrI

    ULUd = evec_.dot(np.diag(eval_)).dot(adj(evec_))
    print "r - UdU+\n", rho_-ULUd

    rhoI_ = evec_.dot(np.diag(evalI)).dot(adj(evec_))
    rhoSr_ = evec_.dot(np.diag(evalSr)).dot(adj(evec_))
    rhoSrI_ = evec_.dot(np.diag(evalSrI)).dot(adj(evec_))
    print "rr^-1\n", rho_.dot(rhoI_)
    print "rSrrSr^-1\n", rhoSr_.dot(rhoSrI_)

    return rhoI_, rhoSr_, rhoSrI_

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
    if(info != 0): print "\nWARNING: bicgstab failed!", info, way, "\n"; exit()

    return F.reshape(chi, chi)

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
    eFixedN = np.trace(tmp.dot(rho_).dot(adj(tmp)) / (2. * m) + w * RR.dot(rho_).dot(adj(RR)))
    print "<n>", density, "e", eFixedN, 

    density = np.trace(supOp(R_, R_, way, rho_))
    eFixedN = np.trace(supOp(tmp, tmp, way, rho_) / (2. * m) + w * supOp(RR, RR, way, rho_))
    print "<n>", density, "e", eFixedN



"""
Main...
"""

np.random.seed(0)
xi = 4
expS = 1.e-6
maxIter = 190000
dTau = .01
K = 1 * (np.random.rand(xi, xi) - .5) #+ 1j * np.zeros((xi, xi))
K = .5 * (K - adj(K))
R = 1 * (np.random.rand(xi, xi) - .5) #+ 1j * np.zeros((xi, xi))
rho = np.random.rand(xi, xi) - .5

m, v, w = .5, -.5, 2.

I = 0
while (I != maxIter):
    print "\t\t\t\t\t\t\t############# ITERATION", I, "#############"

    Q, rho = leftNormalization(K, R, rho)
    #Q, rho = rightNormalization(K, R, ...)

    rhoI, rhoSr, rhoSrI = rhoVersions(rho)
    exit()
    F = calcF(Q, R, 'L', rho)

    Ystar = calcYstar(Q, R, F, rho, rhoI, rhoSr, rhoSrI)

    Vstar, Wstar = getUpdateVandW(R, rhoSrI, Ystar)

    calcQuantities(Q, R, rho, 'L')

    K, R = doUpdateQandR(K, R, Wstar)

    I += 1
