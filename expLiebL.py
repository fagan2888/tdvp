#!/usr/bin/python

import sys
import math
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import scipy.integrate as spig
import functools

#np.set_printoptions(precision = 4, suppress = True)

adj = lambda X: np.transpose(np.conjugate(X))
comm = lambda X, Y: np.dot(X, Y) - np.dot(Y, X)
trNorm = lambda X: np.sqrt(np.trace(adj(X).dot(X)))

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
    if(guess == None): guess = fixPhase(np.random.rand(chi, chi) - .5)
    #print "GUESS\n", guess
    guess = guess.reshape(chi * chi)

    """
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
    """
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

    #print "SOL\n", sol, 
    print "\n(l|T", \
        trNorm(linOpForT(Q_, R_, 'L', np.eye(chi, chi)).reshape(chi, chi)), \
        "T|r)", trNorm(linOpForT(Q_, R_, 'R', sol.real).reshape(chi, chi))

    return sol.real

def fixPhase(X):
    Y = X / np.trace(X)
    norm = 1.0 #np.sqrt(np.trace(adj(Y).dot(Y)))
    Y = Y / norm

    return Y

def leftNormalization(K_, R_, guess):
    chi, chi = K_.shape
    Q_ = K_ - .5 * (adj(R_).dot(R_))
    #print "K\n", K_, "\nR\n", R_, "\nQ\n", Q_

    #l_ = tryGetBestSol(Q_, R_, 'L', guess)
    #l_ = fixPhase(l_)
    #fac = np.trace(l_) / chi
    #print "l", np.trace(l_), "\n", l_/fac

    r_ = tryGetBestSol(Q_, R_, 'R', guess)
    r_ = fixPhase(r_)
    #print "r", np.trace(r_), "\n", r_

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
    #print "lambda", reduce(lambda x, y: x+y, eval_), eval_

    eval_ = abs(eval_)
    evalI = 1. / eval_
    evalSr = np.sqrt(eval_)
    evalSrI = 1. / np.sqrt(eval_)
    #print "evalI  ", evalI, "\nevalSr ", evalSr, "\nevalSrI", evalSrI

    ULUd = evec_.dot(np.diag(eval_)).dot(adj(evec_))
    #print "r - UdU+\n", rho_ - ULUd

    rhoI_ = evec_.dot(np.diag(evalI)).dot(adj(evec_))
    rhoSr_ = evec_.dot(np.diag(evalSr)).dot(adj(evec_))
    rhoSrI_ = evec_.dot(np.diag(evalSrI)).dot(adj(evec_))
    #print "rr^-1\n", rho_.dot(rhoI_)
    #print "rSrrSr^-1\n", rhoSr_.dot(rhoSrI_)

    return rhoI_, rhoSr_, rhoSrI_

def getQrhsQ(Q_, R_, way, rho_, muL_):
    chi, chi = Q_.shape
    commQR = comm(Q_, R_)
    Id = np.eye(chi, chi)

    kinE = 1./(2. * m) * supOp(commQR, commQR, way, Id)
    potE = v * supOp(R_, R_, way, Id)
    intE = w * supOp(R_, R_, way, muL_)
    ham = kinE + potE + intE

    energy = np.trace(ham.dot(rho_))
    rhs = ham - (energy * Id)
    print "GDB: Energy density", trNorm(commQR), energy, trNorm(kinE), \
        trNorm(potE), trNorm(intE)

    return rhs.reshape(chi * chi)

def linOpForF(Q_, R_, way, rho_, X):
    chi, chi = Q_.shape
    X = X.reshape(chi, chi)
    Id = np.eye(chi, chi)
    shift = np.trace(X.dot(rho_))

    FTF = linOpForT(Q_, R_, way, X).reshape(chi, chi) - (shift * Id)

    return FTF.reshape(chi * chi)

def calcF(Q_, R_, way, rho_, muL_, guess):
    chi, chi = Q_.shape
    rhs = - getQrhsQ(Q_, R_, way, rho_, muL_)

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

    #print "F\n", F.reshape(chi, chi)
    return F.reshape(chi, chi)

def calcYstar(Q_, R_, F_, rho_, rhoI_, rhoSr_, rhoSrI_, muL_, muR_):
    fContrib = - R_.dot(F_).dot(rhoSr_) + F_.dot(R_).dot(rhoSr_)

    common = comm(Q_, R_).dot(rho_)
    partOne = comm(adj(Q_), common).dot(rhoSrI_)
    partTwo = R_.dot(comm(common, adj(R_))).dot(rhoSrI_)

    kContrib = (partOne - partTwo) / (2. * m)
    pContrib = v * (R_.dot(rhoSr_))

    #RR = R_.dot(R_)
    partOne = R_.dot(muR_).dot(rhoSrI_)
    partTwo = muL_.dot(R_).dot(rhoSr_)

    iContrib = w * (partOne + partTwo)

    partOne = - R_.dot(muL_).dot(muR_).dot(rhoSrI_)
    partTwo = muL_.dot(R_).dot(muR_).dot(rhoSrI_)

    lrContrib = w * (partOne + partTwo)

    Ystar_ = fContrib + kContrib + pContrib + iContrib + lrContrib
    print "GDB: Y contrib", trNorm(fContrib), trNorm(kContrib), \
        trNorm(pContrib), trNorm(iContrib), trNorm(lrContrib)
    #print "Ystar\n", Ystar_

    return Ystar_

def getUpdateVandW(R_, rhoSrI_, Ystar_):
    Vstar_ = - adj(R_).dot(Ystar_).dot(rhoSrI_)
    Wstar_ = Ystar_.dot(rhoSrI_)
    conver = np.sqrt(np.trace(adj(Ystar_).dot(Ystar_)))
    #print "Vstar\n", Vstar_, "\nWstar\n", Wstar_
    print "GDB: conver", I, conver, dTau,

    return Vstar_, Wstar_

def doUpdateQandR(K_, R_, Wstar_, rho__, F__, muL__, muR__):
    effUpK = - .5 * (adj(R_).dot(Wstar_) - adj(Wstar_).dot(R_))
    effUpR = Wstar_

    K__ = K_ - .5 * dTau * effUpK
    R__ = R_ - .5 * dTau * effUpR

    ldTau, lTol, lEta, zeta, J = dTau, expS, 1.e9, 1.e9, 0
    while (zeta > lTol and J < 30):
        Q__, rho__ = leftNormalization(K__, R__, rho__)
        rhoI__, rhoSr__, rhoSrI__ = rhoVersions(rho__)
        muL__, muR__ = calcMuS(Q, R, rho, muL__, muR__)
        F__ = calcF(Q__, R__, 'L', rho__, muL__, F__)
        Ystar__ = calcYstar(Q__, R__, F__, rho__, rhoI__, rhoSr__, rhoSrI__, muL__, muR__)
        lEta, ldTau = evaluateStep(Ystar__, lEta, ldTau)
        Vstar__, Wstar__ = getUpdateVandW(R__, rhoSrI__, Ystar__)
        calcQuantities(Q__, R__, rho__, 'L')

        effUpK__ = - .5 * (adj(R__).dot(Wstar__) - adj(Wstar__).dot(R__))
        effUpR__ = Wstar__

        dK__ = .5 * dTau * (effUpK - effUpK__)
        dR__ = .5 * dTau * (effUpR - effUpR__)

        K__ += dK__
        R__ += dR__

        effUpK, effUpR = effUpK__, effUpR__
        zeta = np.sqrt(np.trace(supOp(dR__, dR__, 'R', rho__)))

        print "GDB:", J, ldTau, "zeta", zeta, "|dR__|", trNorm(dR__), "\nGDB: ---"
        J += 1

    #Q__ = K__ - .5 * (adj(R__).dot(R__))
    #print "K\n", K__, "\nR\n", R__, "\nQ\n", Q__

    K__ = K__ - .5 * dTau * effUpK__
    R__ = R__ - .5 * dTau * effUpR__

    return K__, R__

def calcQuantities(Q_, R_, rho_, way):
    tmp = comm(Q_, R_)
    RR = R_.dot(R_)
    #if(way == 'L'): way = 'R' else: way = 'L'
    way = 'R' if way == 'L' else 'L'

    density = np.trace(supOp(R_, R_, way, rho_))
    eFixedN = np.trace(supOp(tmp, tmp, way, rho_) / (2. * m) 
                       + w * supOp(RR, RR, way, rho_))

    print "<n>", density, "e", eFixedN, 
    print "e/<n>^3", eFixedN/density**3, "g/<n>", w/density
    print "GDB: ***"

def evaluateStep(Ystar_, oldEta_, dTau_):
    newEta = np.sqrt(np.trace(adj(Ystar_).dot(Ystar_)))
    ratio = oldEta_ / newEta
    if(ratio < 1. and dTau_ > dTauMin): dTau_ = dTau_ * ratio / 1.001
    if(ratio > 1. and dTau_ < dTauMax and \
           I != 0 and I % 100 == 0): dTau_ = dTau_ * ratio * 1.001

    return newEta, dTau_

def linOpForOde(X, x, Q_, R_, way):
    return linOpForT(Q_, R_, way, X)

def onePartCorr(Q_, R_, rho_):
    chi, chi = Q_.shape
    Id = np.eye(chi, chi)
    x = np.linspace(0, 100, 300)

    ket = supOp(Id, R_, 'R', rho_)
    l0 = supOp(R_, Id, 'L', Id).reshape(chi * chi)

    n = np.trace(l0.reshape(chi, chi).dot(ket))
    print "#init cond", n

    bra = spig.odeint(linOpForOde, l0, x, args = (Q_, R_, 'L'), rtol = expS, 
                      atol = expS)
    bra = bra.reshape(len(x), chi, chi)

    corr = [bra[i].dot(ket) for i in range(len(x))]
    corr = np.array(map(np.trace, corr)) / n
    print "#bra", bra.shape, x.shape, corr.shape, "\n#corr\n", \
        '\n'.join(map(str, corr))

def rhoRhoCorr(Q_, R_, rho_):
    chi, chi = Q_.shape
    Id = np.eye(chi, chi)
    x = np.linspace(0, 100, 300)

    ket = supOp(R_, R_, 'R', rho_)
    l0 = supOp(R_, R_, 'L', Id).reshape(chi * chi)

    n = np.trace(l0.reshape(chi, chi).dot(ket))
    print "\n#init cond", n

    bra = spig.odeint(linOpForOde, l0, x, args = (Q_, R_, 'L'), rtol = expS, 
                      atol = expS)
    bra = bra.reshape(len(x), chi, chi)

    corr = [bra[i].dot(ket) for i in range(len(x))]
    corr = np.array(map(np.trace, corr))
    print "#bra", bra.shape, x.shape, corr.shape, "\n#corr\n", \
        '\n'.join(map(str, corr))

def linOpForExp(Q_, R_, way, X):
    chi, chi = Q_.shape
    XTX = mu * X - linOpForT(Q_, R_, way, X)
    #print "XTX", XTX

    return XTX

def solveForMu(Q_, R_, way, myGuess, method, X):
    chi, chi = Q_.shape
    inhom = supOp(R_, R_, way, X).reshape(chi * chi)
    #print "solveForMu:", way

    linOpWrap = functools.partial(linOpForExp, Q_, R_, way)
    linOpForSol = spspla.LinearOperator((chi * chi, chi * chi), 
                                        matvec = linOpWrap, dtype = 'float64')
    if(method == 'bicgstab'):
        S, info = spspla.lgmres(linOpForSol, inhom, tol = expS, x0 = myGuess, 
                                maxiter = maxIter, outer_k = 6)
    else:#if(method == 'gmres'):
        S, info = spspla.gmres(linOpForSol, inhom, tol = expS, x0 = myGuess, 
                               maxiter = maxIter)

    return info, S.reshape(chi, chi)

def calcMuS(Q_, R_, rho_, guessL, guessR):
    chi, chi = Q_.shape
    guessL = guessL.reshape(chi * chi)
    guessR = guessR.reshape(chi * chi)

    info, muL_ = solveForMu(Q_, R_, 'L', guessL, 'bicgstab', np.eye(chi, chi))
    #print "info", info, "muL\n", muL_

    info, muR_ = solveForMu(Q_, R_, 'R', guessR, 'bicgstab', rho_)
    #print "info", info, "muR\n", muR_

    return muL_, muR_

def writeFiles(K_, R_):
    aux = "xi" + str(xi) + "mu" + str(mu) + "w" + str(w)
    np.save('K_' + aux, K_)
    np.save('R_' + aux, R_)

def readFiles(chi__):
    K__ = .5 * (np.random.rand(chi__, chi__) - .5)
    R__ = .5 * (np.random.rand(chi__, chi__) - .5)
    rho__ = fixPhase(np.random.rand(chi__, chi__) - .5)
    F__ = np.random.rand(chi__, chi__) - .5
    muL__ = np.random.rand(chi__, chi__) - .5
    muR__ = np.random.rand(chi__, chi__) - .5

    try:
        K_ = np.load('K_.npy')
    except IOError:
        K_ = K__
    try:
        R_ = np.load('R_.npy')
    except IOError:
        R_ = R__

    Q_ = K_ - .5 * (np.dot(adj(R_), R_))
    eigv, pho = getLargestW(Q_, R_, 'R', 'SM', None)
    pho = fixPhase(pho.real)
    eval__, evec__ = spla.eigh(pho)
    print "pho\n", pho
    print "tr(pho)", np.trace(pho), "\neval", eval__
    K_ = np.dot(adj(evec__), np.dot(K_, evec__))
    R_ = np.dot(adj(evec__), np.dot(R_, evec__))

    Q_ = K_ - .5 * (np.dot(adj(R_), R_))
    eigv, pho = getLargestW(Q_, R_, 'R', 'SM', None)
    pho = fixPhase(pho.real)
    print "pho\n", pho

    chi_, chi_ = K_.shape
    if(K_.shape < K__.shape):
        print "Expanding", chi_, "->", chi__
        K__[chi__-chi_:, chi__-chi_:] = K_
        R__[chi__-chi_:, chi__-chi_:] = R_
#        K__[:chi_, :chi_] = K_
#        R__[:chi_, :chi_] = R_
    elif(K_.shape > K__.shape):
        print "Shrinking", chi_, "->", chi__
        K__ = K_[chi_-chi__:, chi_-chi__:]
        R__ = R_[chi_-chi__:, chi_-chi__:]
#        K__ = K_[:chi__, :chi__]
#        R__ = R_[:chi__, :chi__]
    else:
        print "Equating", chi_, "==", chi__
        K__ = K_
        R__ = R_

    K__ = .5 * (K__ - adj(K__))
    print "K_\n", K_, "\nK__\n", K__
    print "R_\n", R_, "\nR__\n", R__

    return K__, R__, rho__, F__, muL__, muR__

def readParams(file):
    f = open(file, 'r')
    chi_, dTau_, w_, mu_ = f.read().split()
    f.close()

    return int(chi_), float(dTau_), float(w_), float(mu_)

def vonEntropy(rho_):
    eval_, evec_ = spla.eigh(rho_)
    eval_ = abs(eval_)
    print "eval_ = ", eval_
    print "#tr(rho)", np.sum(w for w in eval_),
    print "S(rho)", np.sum(-w * math.log(w) for w in eval_),
    print "S(rho)", np.sum(-w * math.log(w, 2) for w in eval_)

def getLargestVals(Q_, R_, way, myWhich, myGuess):
    chi, chi = Q_.shape

    linOpWrap = functools.partial(linOpForT, Q_, R_, way)
    linOpForEigs = spspla.LinearOperator((chi * chi, chi * chi),
                                         matvec = linOpWrap, dtype = 'float64')
    omega, X = spspla.eigs(linOpForEigs, k = 6, which = myWhich, tol = expS,
                           maxiter = maxIter, v0 = myGuess, ncv = 10)

    return omega, X



"""
Main...
"""

#np.random.seed(2)
m, v, w, mu = .5, -2., 25., 1.
maxIter, expS, xi = 90000, 1.e-12, 24
oldEta, dTau, dTauMin, dTauMax = 1.e9, .125/100, 1.e-3, 0.125
xi, dTau, w, mu = readParams(sys.argv[1])
K, R, rho, F, muL, muR = readFiles(xi)

#Q = K - .5 * (np.dot(adj(R), R))
#tVals, tVecs = getLargestVals(Q, R, 'L', 'SM', None)
#print >> sys.stderr, "tVals", tVals
#print >> sys.stderr, "|tVals|", np.abs(tVals)

#exit()

I, flag, measCorr = 0, False, 1.e-1
while (not flag):#I != maxIter):
    print 5*"\t", 15*"#", "ITERATION =", I, 15*"#"
    if(I % 1000 == 0): print "xi", xi, "v", v, "w", w, "mu", mu, "m", m

    Q, rho = leftNormalization(K, R, rho)
    #Q, rho = rightNormalization(K, R, ...)

    rhoI, rhoSr, rhoSrI = rhoVersions(rho)

    muL, muR = calcMuS(Q, R, rho, muL, muR)

    F = calcF(Q, R, 'L', rho, muL, F)

    Ystar = calcYstar(Q, R, F, rho, rhoI, rhoSr, rhoSrI, muL, muR)

    oldEta, dTau = evaluateStep(Ystar, oldEta, dTau)

    Vstar, Wstar = getUpdateVandW(R, rhoSrI, Ystar)

    calcQuantities(Q, R, rho, 'L')

    if(oldEta < measCorr):
        vonEntropy(rho)
	onePartCorr(Q, R, rho)
        rhoRhoCorr(Q, R, rho)
        writeFiles(K, R)
        measCorr /= 10

    K, R = doUpdateQandR(K, R, Wstar, rho, F, muL, muR)

    I += 1

    if(oldEta < 100 * expS): flag = True#break
