#!/usr/bin/python

import numpy as np
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import functools
import cmath

np.set_printoptions(suppress=True, precision=3)

def powerMethod(MPS, dir):
    eval = 1234.5678
    chir, chic, aux = MPS.shape
    X = np.random.rand(chir * chic) - .5

    for q in range(maxIter):
        if(dir == 'R'): Y = linearOpForR(MPS, X)
        else: Y = linearOpForL(MPS, X)

        Y = np.reshape(Y, (chir, chic))
        YH = np.transpose(np.conjugate(Y), (1, 0))
        YHY = np.tensordot(YH, Y, axes=([1, 0]))
        norm = np.sqrt(np.trace(YHY))
        X = Y / norm

        if(np.abs(eval - norm) < expS): return norm, X
        else: eval = norm

    print "\nWARNING: powerMethod did not converge\n"
    return -1, np.zeros(chir, chic)

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

def symmNormalization(MPS, chir, chic):
    """Returns a symmetric normalized MPS.

    This is really tricky business!
    """
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    AA = np.tensordot(MPS, AH, axes=([2,2]))
    AA = np.transpose(AA, (0, 1, 3, 2))
    AA = np.transpose(AA, (0, 2, 1, 3))
    AA = np.reshape(AA, (chir * chic, chir * chic))
    ##print "AA =\n", AA

    omega, R = spspla.eigs(AA, k=1, which='LR', tol=expS, 
                           maxiter=maxIter)
    print "w(1) =", omega, "\nR\n", R.reshape(chir, chic)

    linOpWrapped = functools.partial(linearOpForR, MPS)
    linOpForEigs = spspla.LinearOperator((chir * chic, chir * chic), 
                                         matvec = linOpWrapped)
    omega, R = spspla.eigs(linOpForEigs, k=1, which='LR', tol=expS, 
                           maxiter=maxIter)
    R = R.reshape(chir, chic)
    print "w(1) =", omega, "\nR\n", R


    R = spla.sqrtm(R)
    A = np.tensordot(MPS, R, axes=([1,0]))
    R = spla.inv(R)
    A = np.tensordot(R, A, axes=([1,0]))
    MPS = np.transpose(A, (0, 2, 1))


    linOpWrapped = functools.partial(linearOpForL, MPS)
    linOpForEigs = spspla.LinearOperator((chir * chic, chir * chic), 
                                         matvec = linOpWrapped)
    omega, L = spspla.eigs(linOpForEigs, k=1, which='LR', tol=expS, 
                           maxiter=maxIter)
    L = L.reshape(chir, chic)
    print "w(1) =", omega, "\nL\n", L


    eval, evec = spla.eig(L)
    print eval, "\n", evec
    A = np.tensordot(MPS, evec, axes=([1,0]))
    evec = np.conjugate(evec.T)
    MPS = np.tensordot(evec, A, axes=([1,0]))
    MPS = np.transpose(MPS, (0, 2, 1))


    eval = map(np.sqrt, map(np.sqrt, eval))
    A = np.tensordot(np.diag(eval), MPS, axes=([1,0]))
    eval = 1/np.asarray(eval)
    MPS = np.tensordot(A, np.diag(eval), axes=([1,0]))
    MPS = np.transpose(MPS, (0, 2, 1))

    MPS /= np.sqrt(omega)
    print "New MPS =", MPS.shape

    linOpWrapped = functools.partial(linearOpForL, MPS)
    linOpForEigs = spspla.LinearOperator((chir * chic, chir * chic), 
                                         matvec = linOpWrapped)
    omega, L = spspla.eigs(linOpForEigs, k=1, which='LR', tol=expS, 
                           maxiter=maxIter)
    print "chk =", omega, "\nL\n", L.reshape(chir, chic)

    linOpWrapped = functools.partial(linearOpForR, MPS)
    linOpForEigs = spspla.LinearOperator((chir * chic, chir * chic), 
                                         matvec = linOpWrapped)
    omega, R = spspla.eigs(linOpForEigs, k=1, which='LR', tol=expS, 
                           maxiter=maxIter)
    print "chk =", omega, "\nR\n", R.reshape(chir, chic)

    return np.diag(map(np.abs, L))

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

def getQHaaaa(MPS, R, H, getE = False):
    chi, = R.shape
    AA = np.tensordot(MPS, MPS, axes=([1,0]))
    AAH = np.transpose(np.conjugate(AA), (2, 1, 0, 3))
    AAR = np.tensordot(AA, np.diag(R), axes=([2,0]))
    AARAAH = np.tensordot(AAR, AAH, axes=([3,0]))
    tmp = np.tensordot(H, AARAAH, axes=([0,1,2,3], [1,2,3,5]))

    if(getE):
        Q = np.transpose(np.diag(np.conjugate(R)), (1, 0))
        QHAAAA = np.tensordot(Q, tmp, axes=([1,0]))
        return np.trace(QHAAAA)
    else:
        Q = np.diag(np.power(R, 2))
        QHAAAA = np.tensordot(Q, tmp, axes=([1,0]))
        print "Q\n", Q, "\nQHAAAA\n", QHAAAA
        return QHAAAA.reshape(chi * chi)

def linearOpForK(MPS, R, K):
    chir, chic, aux = MPS.shape
    K = np.reshape(K, (chir, chic))
    Q = np.diag(np.power(R, 2))
    QK = np.tensordot(Q, K, axes=([1,0]))
    EQK = linearOpForR(MPS, QK.reshape(chir * chic))
    EQK = np.reshape(EQK, (chir, chic))
    QEQK = np.tensordot(Q, EQK, axes=([1,0]))
    tmp = K - QEQK

    return tmp.reshape(chir * chic)

def calcHmeanval(MPS, R, H):
    QHAAAA = getQHaaaa(MPS, R, H)
    chir, chic, aux = MPS.shape

    linOpWrapped = functools.partial(linearOpForK, MPS, R)
    linOpForBicg = spspla.LinearOperator((chir * chir, chic * chic), 
                                         matvec = linOpWrapped, 
                                         dtype = 'complex64')
    K, info = spspla.bicgstab(linOpForBicg, QHAAAA, tol=expS, 
                              maxiter=maxIter)
    if(info != 0): print "\nWARNING: bicgstab failed!\n"
    K = np.reshape(K, (chir, chic))

    print "Energy density =", getQHaaaa(MPS, R, H, True)
    return K

def nullSpaceR(MPS, R):
    print "IN NULL SPACE R"
    RR = np.transpose(np.conjugate(MPS), (1, 0, 2))
    RR = np.tensordot(np.diag(map(np.sqrt, R)), RR, axes=([1,0]))
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
    Id = np.tensordot(VRdag.T, VRdag, axes=([1,0]))

    tmp = np.conjugate(VRdag.T)
    lpr, lmz = tmp.shape
    tmp = np.reshape(tmp, (lpr, chir, aux))
 
    print "mask =", mask, S, "\nU\n", U, "\nVRdag\n", VRdag, \
        "\nNull\n", Null, "\nVV+\n", Id, "\nVR =", tmp.shape
    del R, U, S, V, VRdag, Null, Id

    return tmp

def calcFs(MPS, C, R, K, VR):
    tmp1 = tmp2 = tmp3 = 0.
    VRdag = np.transpose(np.conjugate(VR), (1, 0, 2))
    Lsqrt = Rsqrt = np.diag(map(np.sqrt, R))
    Lsqrti = Rsqrti = np.diag(map(np.sqrt, 1./R))

    A = np.transpose(np.conjugate(MPS), (1, 0, 2))
    RsiVRdag = np.tensordot(Rsqrti, VRdag, axes=([1,0]))
    ARsiVRdag = np.tensordot(A, RsiVRdag, axes=([1,0]))
    RARsiVRdag = np.tensordot(np.diag(R), ARsiVRdag, axes=([1,0]))
    CRARsiVRdag = np.tensordot(C, RARsiVRdag, axes=([1,2,3], [0,3,1]))
    tmp1 = np.tensordot(Lsqrt, CRARsiVRdag, axes=([1,0]))
    #print tmp1

    RsVRdag = np.tensordot(Rsqrt, VRdag, axes=([1,0]))
    CRsVRdag = np.tensordot(C, RsVRdag, axes=([1,3], [0,2]))
    LCRsVRdag = np.tensordot(np.diag(R), CRsVRdag, axes=([1,0]))
    A = np.transpose(np.conjugate(MPS), (1, 0, 2))
    ALCRsVRdag = np.tensordot(A, LCRsVRdag, axes=([1,2], [0,1]))
    tmp2 = np.tensordot(Lsqrti, ALCRsVRdag, axes=([1,0]))
    #print tmp2

    RsiVRdag = np.tensordot(Rsqrti, VRdag, axes=([1,0]))
    KRsiVRdag = np.tensordot(K, RsiVRdag, axes=([1,0]))
    AKRsiVRdag = np.tensordot(MPS, KRsiVRdag, axes=([1,2], [0,2]))
    tmp3 = np.tensordot(Lsqrt, AKRsiVRdag, axes=([1,0]))
    #print tmp3

    tmp = tmp1 + tmp2 + tmp3

    return tmp

def getUpdateB(R, x, VR):
    Rsqrti = Lsqrti = np.diag(map(np.sqrt, 1./R))

    row, col, aux = VR.shape
    if(row*col == 0):
        row, row = Lsqrti.shape
        tmp = np.zeros((row, col, aux))
    else:
        VRRsi = np.tensordot(VR, Rsqrti, axes=([1,0]))
        VRRsi = np.transpose(VRRsi, (0, 2, 1))
        xVRRsi = np.tensordot(x, VRRsi, axes=([1,0]))
        tmp = np.tensordot(Lsqrti, xVRRsi, axes=([1,0]))

    print Lsqrti.shape, x.shape, VR.shape, tmp.shape

    return tmp

def doUpdateForA(MPS, B):
    """
    It does the actual update to the MPS state for given time step.

    The update is done according to the formula:
    A[n, t + dTau] = A[n, t] - dTau * B[x*](n),
    where dTau is the corresponding time step.
    """
    MPS -= dTau * B
    print "cmp shapes =", MPS.shape, B.shape

    return MPS



"""Main...
"""
d = 2
xi = 3
expS = 1e-12
maxIter = 200
dTau = 0.1

xir = xic = xi
theA = np.random.rand(xir, xic, d) - .5 + 1j * np.zeros((xir, xic, d))

theH = buildLocalH()
print "theH\n", theH.reshape(d*d, d*d)

I = 0
while (I != maxIter):

    theR = symmNormalization(theA, xir, xic)
    print "theR =", theR
    exit()
    theC = buildHElements(theA, theH)
    print "theC =", theC.shape

    theK = calcHmeanval(theA, theR, theH)
    print "theK =", theK.shape

    theVR = nullSpaceR(theA, theR)
    print "theVR =", theVR.shape

    theF = calcFs(theA, theC, theR, theK, theVR)
    print "theF =", theF.shape

    theB = getUpdateB(theR, theF, theVR)
    print "theB =", theB.shape

    doUpdateForA(theA, theB)

    I += 1
