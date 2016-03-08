#!/usr/bin/python

import numpy as np
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import functools
import cmath

np.set_printoptions(suppress=True)#, precision=3)

def powerMethod(MPS, way):
    eVal = 1234.5678
    chir, chic, aux = MPS.shape
    X = np.random.rand(chir * chic) - .5

    for q in range(maxIter):
        if way == 'R': Y = linearOpForR(MPS, X)
        else:          Y = linearOpForL(MPS, X)

        norm = np.linalg.norm(Y)
        X = Y / norm

        if np.abs(eVal - norm) < expS:
            return np.array([norm]), X.reshape(chir, chic)
        else: eVal = norm

    print "\nWARNING: powerMethod did not converge\n"
    return 1234.5678, np.zeros(chir, chic)

def linearOpForR(MPS, R):
    chir, chic, aux = MPS.shape
    R = np.reshape(R, (chir, chic))
    AR = np.tensordot(MPS, R, axes=([1,0]))
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    ARAH = np.tensordot(AR, AH, axes=([1,2], [2,0]))

    return ARAH.reshape(chir * chic)

def linearOpForL(MPS, L):
    chir, chic, aux = MPS.shape
    L = np.reshape(L, (chir, chic))
    LA = np.tensordot(L, MPS, axes=([1,0]))
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    AHLA = np.tensordot(AH, LA, axes=([1,2], [0,2]))

    return AHLA.reshape(chir * chic)

def buildLargeE(MPS):
    chir, chic, aux = MPS.shape
    AC = np.conjugate(MPS)
    AA = np.tensordot(MPS, AC, axes=([2,2]))
    AA = np.transpose(AA, (0, 2, 1, 3))
    AA = np.reshape(AA, (chir * chir, chic * chic))
    #print "AA", AA.shape, "\n", AA

    return AA

def getLargestW(MPS, way):
    chir, chic, aux = MPS.shape

    if way == 'R':
        linOpWrapped = functools.partial(linearOpForR, MPS)
    else:
        linOpWrapped = functools.partial(linearOpForL, MPS)

    linOpForEigs = spspla.LinearOperator((chir * chir, chic * chic), 
                                         matvec = linOpWrapped, 
                                         dtype = MPS.dtype)
    try:
        omega, X = spspla.eigs(linOpForEigs, k = 1, which = 'LR', tol = expS, 
                               maxiter = maxIter, ncv = 12)
    except:
        omega, X = powerMethod(MPS, way)
    else:
        X = X.reshape(chir, chic)

    return omega, X

def fixPhase(X):
    Y = X / np.trace(X)
    norm = np.linalg.norm(Y)
    Y = Y / norm

    return Y

def symmNormalization(MPS, chir, chic):
    omega, R = getLargestW(MPS, 'R')
    if np.isreal(R).all(): omega, R = omega.real, R.real
    print "wR", omega, np.isreal(R).all(), "R\n", R

    assym = np.linalg.norm(R - R.T.conj())
    print "assym R", assym

    Rvals, Rvecs = spla.eigh(R)
    Rvals_s = np.sqrt(abs(Rvals))
    Rvals_si = 1. / Rvals_s
    print "Rvals", Rvals

    Rs = np.dot(Rvecs, np.dot(np.diag(Rvals_s), Rvecs.T.conj()))#Rs = spla.sqrtm(R)
    ARs = np.tensordot(MPS, Rs, axes=([1,0]))
    Rsi = np.dot(Rvecs, np.dot(np.diag(Rvals_si), Rvecs.T.conj()))#Rsi = spla.inv(Rs)
    A1 = np.tensordot(Rsi, ARs, axes=([1,0]))
    A1 = np.transpose(A1, (0, 2, 1))

    omega, L = getLargestW(A1, 'L')
    if np.isreal(L).all(): omega, L = omega.real, L.real
    print "wL", omega, np.isreal(L).all(), "L\n", L

    assym = np.linalg.norm(L - L.T.conj())
    print "assym L", assym

    Lambda2, U = spla.eigh(L)
    A1U = np.tensordot(A1, U, axes=([1,0]))
    A2 = np.tensordot(U.T.conj(), A1U, axes=([1,0]))#Udag = np.transpose(np.conjugate(U))
    A2 = np.transpose(A2, (0, 2, 1))
    print "Lambda**2", Lambda2#, "\n", U

    Lambda = np.sqrt(abs(Lambda2))
    Lambdas = np.sqrt(Lambda)
    Lambdasi = 1. / Lambdas
    A2Lsi = np.tensordot(A2, np.diag(Lambdasi), axes=([1,0]))
    A3 = np.tensordot(np.diag(Lambdas), A2Lsi, axes=([1,0]))
    A3 = np.transpose(A3, (0, 2, 1))
    print "Lambda", Lambda

    nMPS = A3 / np.sqrt(omega)
    RealLambda = fixPhase(np.diag(Lambda))
    #print "nMPS", nMPS.shape, "\n", nMPS
    print "RealLambda", RealLambda.shape, "\n", RealLambda

    ######### CHECKING RESULT #########
    Trace = np.linalg.norm(RealLambda)#trace(np.dot(RealLambda, np.conjugate(RealLambda)))
    ELambda = linearOpForR(nMPS, RealLambda).reshape(chir, chic)
    LambdaE = linearOpForL(nMPS, RealLambda).reshape(chir, chic)
    print "Trace(RealLambda)", Trace, "\nE|r)\n", ELambda, "\n(l|E\n", LambdaE
    exit()

    return RealLambda, nMPS

def buildLocalH():
    """Builds local hamiltonian (d x d)-matrix.

    Returns local hamiltonian for a translation invariant system.
    The local H is a (d, d, d, d)-rank tensor.
    """
    # S = 1 (d = 3)
    Sx = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    Sy = np.array([[0., -1.j, 0.], [1.j, 0., -1.j], [0., 1.j, 0.]])
    Sz, Id = np.diag([1., 0., -1.]), np.eye(d)
    localH = .5 * (np.kron(Sx, Sx) + np.kron(Sy, Sy)) + np.kron(Sz, Sz)
    # S = 1/2 (d = 2)
    # Sx, Sy = np.array([[0., 1.], [1., 0.]]), np.array([[0., -1.j], [1.j, 0.]])
    # Sz, Id = np.diag([1., -1.]), np.eye(d)
    # localH = .25 * (np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz))

    return localH.real.reshape(d, d, d, d)

def buildHElements(MPS, H):
    """Builds the matrix elements of the hamiltonian.

    Returns a tensor of the form C[a, b, s, t].
    """
    AA = np.tensordot(MPS, MPS, axes=([1,0]))
    tmp = np.tensordot(H, AA, axes=([2,3], [1,3]))
    C = np.transpose(tmp, (2, 3, 0, 1))
    #print "C\n", C

    return C

def getQHaaaaR(MPS, Lambda, C):
    """Returns either the energy density or the RHS of eq. for |K).
    """
    L = R = Lambda
    chir, chic, aux = MPS.shape
    AA = np.tensordot(MPS, MPS, axes=([1,0]))
    AAH = np.transpose(np.conjugate(AA), (2, 1, 0, 3))
    CR = np.tensordot(C, R, axes=([1,0]))

    HAR = np.tensordot(CR, AAH, axes=([1,2,3], [1,3,0]))
    h = np.trace(np.tensordot(L, HAR, axes=([1,0])))
    rhs = HAR - (h * R)

    print "Energy density", h, "\nL", L.shape, "HAAAAR", HAR.shape,
    print "rhs", rhs.shape, "rhs\n", rhs

    return rhs.reshape(chir * chic)

def linearOpForK(MPS, Lambda, K):
    """Returns ...

    Instead of defining QEQ as the regularized operator, we could 
    define QEQ = E - S = Q - |r)(l|; therefore avoiding more matrix 
    multiplications. Is this true?
    """
    L = R = Lambda
    chir, chic, aux = MPS.shape
    K = np.reshape(K, (chir, chic))
    EK = linearOpForR(MPS, K.reshape(chir * chic)).reshape(chir, chic)
    trLK = np.trace(np.tensordot(L, K, axes=([1,0])))

    lhs = K - EK + (trLK * R)
    #print "EK", EK.shape, "lhs", lhs.shape

    return lhs.reshape(chir * chic)

def calcHmeanval(MPS, R, C):
    chir, chic, aux = MPS.shape
    QHAAAAR = getQHaaaaR(MPS, R, C)

    linOpWrapped = functools.partial(linearOpForK, MPS, R)
    linOpForBicg = spspla.LinearOperator((chir * chir, chic * chic), 
                                         matvec = linOpWrapped, 
                                         dtype = 'complex128')
    K, info = spspla.bicgstab(linOpForBicg, QHAAAAR, tol = expS, 
                              maxiter = maxIter)
    if info != 0: print "\nWARNING: bicgstab failed!\n"; exit()

    K = np.reshape(K, (chir, chic))
    print "QHAAAAR", QHAAAAR.shape, "K\n", K

    return K

def nullSpaceR(MPS, Lambda):
    R = np.diag(Lambda)
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    RR = np.tensordot(np.diag(map(np.sqrt, R)), AH, axes=([1,0]))
    chir, chic, aux = RR.shape

    RR = np.transpose(RR, (0, 2, 1))
    RR = np.reshape(RR, (chir * aux, chic))
    U, S, V = np.linalg.svd(RR, full_matrices=True)

    #dimS, = S.shape
    #extraS = np.zeros((chir * aux) - dimS)
    #Sp = np.append(S, extraS, 0)
    #maskp = (Sp < epsS) #try: np.select(...)

    mask = np.empty(chir * aux, dtype=bool)
    mask[:] = False; mask[chic:] = True
    VRdag = np.compress(mask, U, axis=1)

    RR = np.conjugate(np.transpose(RR))
    Null = np.tensordot(RR, VRdag, axes=([1,0]))
    Id = np.tensordot(np.conjugate(VRdag.T), VRdag, axes=([1,0]))

    tmp = np.conjugate(np.transpose(VRdag))
    lpr, lmz = tmp.shape
    tmp = np.reshape(tmp, (lpr, chir, aux))

    print "mask =", mask, "\n", S, "\nU\n", U, "\nVRdag\n", VRdag, \
        "\nNull\n", Null, "\nVV+\n", Id, "\nVR =", tmp.shape
    del R, U, S, V, VRdag, Null, Id

    return tmp

def calcFs(MPS, C, Lambda, K, VR):
    VRdag = np.transpose(np.conjugate(VR), (1, 0, 2))
    Ld = np.diag(Lambda)
    Lsqrt = Rsqrt = np.diag(map(np.sqrt, Ld))
    Lsqrti = Rsqrti = np.diag(map(np.sqrt, 1./Ld))
    A = np.transpose(np.conjugate(MPS), (1, 0, 2))

    RsiVRdag = np.tensordot(Rsqrti, VRdag, axes=([1,0]))
    ARsiVRdag = np.tensordot(A, RsiVRdag, axes=([1,0]))
    RARsiVRdag = np.tensordot(Lambda, ARsiVRdag, axes=([1,0]))
    CRARsiVRdag = np.tensordot(C, RARsiVRdag, axes=([1,2,3], [0,3,1]))
    tmp1 = np.tensordot(Lsqrt, CRARsiVRdag, axes=([1,0]))
    #print tmp1

    RsVRdag = np.tensordot(Rsqrt, VRdag, axes=([1,0]))
    CRsVRdag = np.tensordot(C, RsVRdag, axes=([1,3], [0,2]))
    LCRsVRdag = np.tensordot(Lambda, CRsVRdag, axes=([1,0]))
    ALCRsVRdag = np.tensordot(A, LCRsVRdag, axes=([1,2], [0,1]))
    tmp2 = np.tensordot(Lsqrti, ALCRsVRdag, axes=([1,0]))
    #print tmp2

    RsiVRdag = np.tensordot(Rsqrti, VRdag, axes=([1,0]))
    KRsiVRdag = np.tensordot(K, RsiVRdag, axes=([1,0]))
    AKRsiVRdag = np.tensordot(MPS, KRsiVRdag, axes=([1,2], [0,2]))
    tmp3 = np.tensordot(Lsqrt, AKRsiVRdag, axes=([1,0]))
    #print tmp3

    tmp = tmp1 + tmp2 + tmp3
    print "F\n", tmp

    return tmp

def getUpdateB(Lambda, x, VR):
    Rsqrti = Lsqrti = np.diag(map(np.sqrt, 1./np.diag(Lambda)))
    row, col, aux = VR.shape

    if row * col == 0:
        row, row = Lsqrti.shape
        tmp = np.zeros((row, col, aux))
    else:
        VRRsi = np.tensordot(VR, Rsqrti, axes=([1,0]))
        VRRsi = np.transpose(VRRsi, (0, 2, 1))
        xVRRsi = np.tensordot(x, VRRsi, axes=([1,0]))
        tmp = np.tensordot(Lsqrti, xVRRsi, axes=([1,0]))

    #print Lsqrti.shape, x.shape, VR.shape, tmp.shape
    print "B\n", tmp

    return tmp

def doUpdateForA(MPS, B):
    """It does the actual update to the MPS state for given time step.

    The update is done according to the formula:
    A(t + dTau) = A(t) - dTau * B(x*),
    where dTau is the corresponding time step.
    """
    nMPS = MPS - dTau * B
    print "cmp shapes =", nMPS.shape, B.shape

    return nMPS



"""Main...
"""
np.random.seed(9)
d, xi = 3, 4
maxIter, expS = 1800, 1.e-12
dTau = 0.1

xir = xic = xi
theMPS = np.random.rand(xir, xic, d) - .5# + 1j * np.zeros((xir, xic, d))
#theMPS = np.random.rand(xir, xic, d) - .5 + 1j * (np.random.rand(xir, xic, d) - .5)
print "theMPS", type(theMPS), theMPS.dtype, "\n", theMPS

theH = buildLocalH()
print "theH\n", theH.reshape(d*d, d*d)

I = 0
while I != maxIter:
    print 5*"\t", 15*"#", "ITERATION =", I, 15*"#"

    theL, theMPS = symmNormalization(theMPS, xir, xic)
    print "theMPS\n", theMPS, "\ntheL\n", theL

    theC = buildHElements(theMPS, theH)
    print "theC =", theC.shape

    theK = calcHmeanval(theMPS, theL, theC)
    print "theK =", theK.shape

    theVR = nullSpaceR(theMPS, theL)
    print "theVR =", theVR.shape

    theF = calcFs(theMPS, theC, theL, theK, theVR)
    print "theF =", theF.shape

    theB = getUpdateB(theL, theF, theVR)
    print "theB =", theB.shape

    theMPS = doUpdateForA(theMPS, theB)

    eta = np.linalg.norm(theF)
    print "eta", I, eta
    if eta < 100 * expS: break
    I += 1
