#!/opt/local/bin/python2.7
# #!/usr/bin/python

import numpy as np
import scipy.sparse.linalg as spspla
import scipy.linalg as spla
import functools
import cmath
import sys

np.set_printoptions(suppress=True)#, precision=3)

def powerMethod(MPS, way, guess=None):
    eVal = 1234.5678
    chir, chic, aux = MPS.shape
    if guess is None or not guess.size: X = np.random.rand(chir * chic) - .5
    else: print "guess", guess.shape, guess.size, "\n", guess

    for q in range(maxIter):
        if way == 'R': Y = linearOpForR(MPS, X)
        else:          Y = linearOpForL(MPS, X)

        norm = np.linalg.norm(Y)
        X = Y / norm

        if abs(eVal - norm) < expS:
            return np.array([norm]), X.reshape(chir, chic)
        else: eVal = norm

    print >> sys.stderr, "powerMethod: Error powerMethod did not converge"
    return np.array([norm]), X.reshape(chir, chic)

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

def getLargestW(MPS, way, guess):
    chir, chic, aux = MPS.shape

    if way == 'R': linOpWrapped = functools.partial(linearOpForR, MPS)
    else:          linOpWrapped = functools.partial(linearOpForL, MPS)

    linOpForEigs = spspla.LinearOperator((chir * chir, chic * chic), 
                                         matvec = linOpWrapped, 
                                         dtype = MPS.dtype)
    try:
        omega, X = spspla.eigs(linOpForEigs, k = 1, which = 'LR', tol = 1.e-14, 
                               maxiter = maxIter, ncv = 12, v0 = guess)
    except spspla.ArpackNoConvergence as err:
        print >> sys.stderr, "getLargestW: Error", I, err
        omega, X = powerMethod(MPS, way, err.eigenvectors)
    except ValueError as err:
        print >> sys.stderr, "getLargestW: Error", I, err
        omega, X = powerMethod(MPS, way)
    else:
        X = X.reshape(chir, chic)

    return omega, X

def fixPhase(X):
    Y = X / np.trace(X)
    norm = 1.#np.linalg.norm(Y)
    Y = Y / norm

    return Y

def symmNormNew(MPS, chir, chic, guess, thld = 1.e-10):
    print 5*"\t", 15*"#", "ITERATION =", I, 15*"#"

    omega, R = getLargestW(MPS, 'R', guess)
    R = fixPhase(R)
    if np.isreal(R).all(): omega, R = omega.real, R.real
    print "wR", omega, np.isreal(R).all(), "R\n", R

    assym = np.linalg.norm(R - R.T.conj())
    print "assym R", I, assym

    try:
        Rvals, Rvecs = spla.eigh(R)
    except spla.LinAlgError as err:
        print >> sys.stderr, "symmNormalization: Error R", I, err

    print "Rvals", Rvals
    noZeros = np.count_nonzero(Rvals > thld)
    Rvals[:Rvals.size - noZeros] = 0.
    Rvals_s = np.sqrt(Rvals)
    Rvals_si = np.zeros(Rvals.shape, dtype=Rvals.dtype)
    Rvals_si[-noZeros:] = 1. / Rvals_s[-noZeros:]

    right = np.dot(Rvecs, np.diag(Rvals_s))
    left = np.dot(np.diag(Rvals_si), Rvecs.conj().T)
    B = np.tensordot(left, MPS, axes=([1,0]))
    B = np.tensordot(B, right, axes=([1,0]))
    B = np.transpose(B, (0, 2, 1))
    B /= np.sqrt(omega)
    print "Rvals", Rvals, "\nRvals_s", Rvals_s, "\nRvals_si", Rvals_si

    omega, L = getLargestW(B, 'L', guess)
    L = fixPhase(L)
    if np.isreal(L).all(): omega, L = omega.real, L.real
    print "wL", omega, np.isreal(L).all(), "L\n", L

    assym = np.linalg.norm(L - L.T.conj())
    print "assym L", I, assym

    try:
        Lambda, U = spla.eigh(L)
    except spla.LinAlgError as err:
        print >> sys.stderr, "symmNormalization: Error L", I, err

    print "Lambda", Lambda
    noZeros = np.count_nonzero(Lambda > thld)
    Lambda[:Lambda.size - noZeros] = 0.
    Lambda_s = np.sqrt(np.sqrt(Lambda))
    Lambda_si = np.zeros(Lambda.shape, dtype=Lambda.dtype)
    Lambda_si[-noZeros:] = 1. / Lambda_s[-noZeros:]

    right = np.dot(U, np.diag(Lambda_si))
    left = np.dot(np.diag(Lambda_s), U.conj().T)
    C = np.tensordot(left, B, axes=([1,0]))
    C = np.tensordot(C, right, axes=([1,0]))
    C = np.transpose(C, (0, 2, 1))
    C /= np.sqrt(omega)
    print "Lambda", Lambda, "\nLambda_s", Lambda_s, "\nLambda_si", Lambda_si

    g_s = np.diag(np.sqrt(Lambda))
    g_s /= np.sqrt(np.trace(np.dot(g_s, g_s)))
    print "g_s", np.diag(g_s)

    trg_s, normg_s, trlr = np.trace(g_s), np.linalg.norm(g_s), np.trace(np.dot(g_s, g_s))
    Eg_s = linearOpForR(C, g_s).reshape(chir, chic)
    g_sE = linearOpForL(C, g_s).reshape(chir, chic)
    print "Trace(g_s)", trg_s, "Norm(g_s)", normg_s, "Trace(l * r)", trlr
    print "E|r)-|r)", np.linalg.norm(Eg_s-g_s), "(l|E-(l|", np.linalg.norm(g_sE-g_s)
    print "E|r)\n", Eg_s, "\n(l|E\n", g_sE

    g_s, C, __ = truncation(C, g_s)

    return g_s, C

def symmNormalization(MPS, chir, chic):
    print 5*"\t", 15*"#", "ITERATION =", I, 15*"#"

    omega, R = getLargestW(MPS, 'R')
    R = fixPhase(R)
    if np.isreal(R).all(): omega, R = omega.real, R.real
    print "wR", omega, np.isreal(R).all(), "R\n", R

    assym = np.linalg.norm(R - R.T.conj())
    print "assym R", assym

    try:
        Rvals, Rvecs = spla.eigh(R)
    except spla.LinAlgError as err:
        print >> sys.stderr, "symmNormalization: Error R", I, err

    Rvals_s = np.sqrt(abs(Rvals))
    Rvals_si = 1. / Rvals_s
    print "Rvals", Rvals

    Rs = np.dot(Rvecs, np.dot(np.diag(Rvals_s), Rvecs.T.conj()))
    ARs = np.tensordot(MPS, Rs, axes=([1,0]))
    Rsi = np.dot(Rvecs, np.dot(np.diag(Rvals_si), Rvecs.T.conj()))
    A1 = np.tensordot(Rsi, ARs, axes=([1,0]))
    A1 = np.transpose(A1, (0, 2, 1))

    omega, L = getLargestW(A1, 'L')
    L = fixPhase(L)
    if np.isreal(L).all(): omega, L = omega.real, L.real
    print "wL", omega, np.isreal(L).all(), "L\n", L

    assym = np.linalg.norm(L - L.T.conj())
    print "assym L", assym

    try:
        Lambda2, U = spla.eigh(L)
    except spla.LinAlgError as err:
        print >> sys.stderr, "symmNormalization: Error L", I, err

    A1U = np.tensordot(A1, U, axes=([1,0]))
    A2 = np.tensordot(U.T.conj(), A1U, axes=([1,0]))
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
    print "Finite", np.isfinite(nMPS).all(), np.isfinite(RealLambda).all()

    ######### CHECKING RESULT #########
    Trace = np.linalg.norm(RealLambda)
    ELambda = linearOpForR(nMPS, RealLambda).reshape(chir, chic)
    LambdaE = linearOpForL(nMPS, RealLambda).reshape(chir, chic)
    print "Trace(RealLambda)", Trace, "E|r)-|r)", np.linalg.norm(ELambda - RealLambda),
    print "(l|E-(l|", np.linalg.norm(LambdaE - RealLambda)

    return RealLambda, nMPS

def buildLocalH(Jh, Hz):
    """Builds local hamiltonian (d x d)-matrix.

    Returns local hamiltonian for a translation invariant system.
    The local H is a (d, d, d, d)-rank tensor.
    """
    # S = 1 (d = 3)
    # Sx = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    # Sy = np.array([[0., -1.j, 0.], [1.j, 0., -1.j], [0., 1.j, 0.]])
    # Sz, Id = np.diag([1., 0., -1.]), np.eye(d)
    # localH = .5 * (np.kron(Sx, Sx) + np.kron(Sy, Sy)) + np.kron(Sz, Sz)
    # S = 1/2 (d = 2)
    dLoc = d#int(np.sqrt(d))
    Sx, Sy = np.array([[0., 1.], [1., 0.]]), np.array([[0., -1.j], [1.j, 0.]])
    Sz, Id = np.diag([1., -1.]), np.eye(dLoc)
    # XY + transverse field
    localH = Jh * np.kron(Sx, Sx) + Jh * np.kron(Sy, Sy) \
             + (Hz / 2.) * (np.kron(Sz, Id) + np.kron(Id, Sz))
    # # Heisenberg chain
    # localH = .25 * (np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz))
    print "theH", "dLoc", dLoc, "Jex", Jh, "Hz", Hz, "\n", localH.real

    return localH.real.reshape(dLoc, dLoc, dLoc, dLoc)

    SxId, IdSx = np.kron(Sx, Id), np.kron(Id, Sx)
    SyId, IdSy = np.kron(Sy, Id), np.kron(Id, Sy)
    SzId, IdSz = np.kron(Sz, Id), np.kron(Id, Sz)
    IdId = np.kron(Id, Id)

    bulkHl = Jh * np.kron(SxId, SxId) + Jh * np.kron(SyId, SyId) \
             + (Hz / 2.) * (np.kron(SzId, IdId) + np.kron(IdId, SzId))
    bulkHr = Jh * np.kron(IdSx, IdSx) + Jh * np.kron(IdSy, IdSy) \
             + (Hz / 2.) * (np.kron(IdSz, IdId) + np.kron(IdId, IdSz))
    print "Hmpo", "d", d, "Jex", Jh, "Hz", Hz, "\n", (bulkHl + bulkHr).real

    return (bulkHl + bulkHr).real.reshape(d, d, d, d)


def buildHElements(MPS, H):
    """Builds the matrix elements of the hamiltonian.

    Returns a tensor of the form C[a, b, s, t].
    """
    AA = np.tensordot(MPS, MPS, axes=([1,0]))
    tmp = np.tensordot(H, AA, axes=([2,3], [1,3]))
    # C = np.transpose(tmp, (2, 3, 0, 1))
    tmp = np.transpose(tmp, (2, 1, 0, 3))
    C = np.transpose(tmp, (0, 3, 2, 1))
    print "C", C.shape#, "\n", C

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

    print "Energy density", I, h.real, xi, xiTilde, "\nL", L.shape,
    print "HAAAAR", HAR.shape, "rhs", rhs.shape, "rhs\n", rhs

    return rhs.reshape(chir * chic)

def linearOpForK(MPS, Lambda, K):
    """Returns ...

    Instead of defining QEQ as the regularized operator, we could 
    define QEQ = E - S = Q - |r)(l|; therefore avoiding more matrix 
    multiplications. Is this true?
    """
    L = R = Lambda
    chir, chic, aux = MPS.shape

    EK = linearOpForR(MPS, K).reshape(chir, chic)
    K = np.reshape(K, (chir, chic))
    trLK = np.trace(np.tensordot(L, K, axes=([1,0])))

    lhs = K - EK + (trLK * R)
    #print "EK", EK.shape, "lhs", lhs.shape

    return lhs.reshape(chir * chic)

def calcHmeanval(MPS, R, C, guess):
    chir, chic, aux = MPS.shape
    QHAAAAR = getQHaaaaR(MPS, R, C)
    if (chir, chic) < guess.shape: guess = guess[-chir:, -chic:]

    linOpWrapped = functools.partial(linearOpForK, MPS, R)
    linOpForLsol = spspla.LinearOperator((chir * chir, chic * chic), 
                                         matvec = linOpWrapped, 
                                         dtype = MPS.dtype)

    K, info = spspla.lgmres(linOpForLsol, QHAAAAR, tol = 1.e-14, 
                            maxiter = maxIter, x0 = guess.reshape(chir * chic))
    if info > 0:
        K, info = spspla.gmres(linOpForLsol, QHAAAAR, tol = expS, x0 = K, 
                               maxiter = maxIter)
    elif info < 0:
        K, info = spspla.gmres(linOpForLsol, QHAAAAR, tol = expS, 
                               maxiter = maxIter)
    if info != 0:
        print >> sys.stderr, "calcHmeanval: Error", I, "lgmres and gmres failed"

    K = np.reshape(K, (chir, chic))
    print "QHAAAAR", QHAAAAR.shape, "info", info, np.isfinite(K).all(), "K\n", K

    return K

def nullSpaceR(MPS, Lambda):
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    RR = np.tensordot(np.sqrt(Lambda), AH, axes=([1,0]))
    chir, chic, aux = RR.shape

    RR = np.transpose(RR, (0, 2, 1))
    RR = np.reshape(RR, (chir * aux, chic))

    try:
        U, S, V = spla.svd(RR, full_matrices=True)
    except spla.LinAlgError as err:
        print >> sys.stderr, "nullSpaceR: Error", I, err

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
        "\nNull\n", Null, "\nVV+\n", Id#, "\nVR =", tmp.shape
    del U, S, V, VRdag, Null, Id

    return tmp

def nullSpaceL(MPS, Lambda):
    AH = np.transpose(np.conjugate(MPS), (1, 0, 2))
    LL = np.tensordot(AH, np.sqrt(Lambda), axes=([1,0]))
    chir, aux, chic = LL.shape

    LL = np.reshape(LL, (chir, aux * chic))

    try:
        U, S, V = spla.svd(LL, full_matrices=True)
    except spla.LinAlgError as err:
        print >> sys.stderr, "nullSpaceL: Error", I, err

    mask = np.empty(aux * chic, dtype=bool)
    mask[:] = False; mask[chir:] = True
    VLdag = np.compress(mask, V, axis=0)

    LL = np.conjugate(np.transpose(LL))
    Null = np.tensordot(VLdag, LL, axes=([1,0]))
    Id = np.tensordot(VLdag, VLdag.T.conj(), axes=([1,0]))

    tmp = np.conjugate(VLdag.T)
    lpr, lmz = tmp.shape
    tmp = np.reshape(tmp, (aux, chic, lmz))
    tmp = np.transpose(np.transpose(tmp, (2, 1, 0)), (1, 0, 2))

    print "mask =", mask, "\n", S, "\nV\n", V, "\nVLdag\n", VLdag, \
        "\nNull\n", Null, "\nV+V\n", Id#, "\nVL =", tmp.shape
    del U, S, V, VLdag, Null, Id

    return tmp

def calcFs(MPS, C, Lambda, K, VR, thld = 1.e-10):
    VRdag = np.transpose(np.conjugate(VR), (1, 0, 2))
    A = np.transpose(np.conjugate(MPS), (1, 0, 2))

    Ld = np.diag(Lambda)
    noZeros = np.count_nonzero(Ld > thld)
    Ld_s = np.sqrt(Ld)
    Lsqrt = Rsqrt = np.diag(Ld_s)
    Ld_si = np.zeros(Ld.shape, dtype=Ld.dtype)
    Ld_si[-noZeros:] = 1. / Ld_s[-noZeros:]
    Lsqrti = Rsqrti = np.diag(Ld_si)#np.diag(np.sqrt(1. / Ld))

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

def getUpdateB(Lambda, x, VR, thld = 1.e-10):
    Ld = np.diag(Lambda)
    noZeros = np.count_nonzero(Ld > thld)
    Ld_s = np.sqrt(Ld)
    Ld_si = np.zeros(Ld.shape, dtype=Ld.dtype)
    Ld_si[-noZeros:] = 1. / Ld_s[-noZeros:]
    Rsqrti = Lsqrti = np.diag(Ld_si)#np.diag(np.sqrt(1. / np.diag(Lambda)))
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

def supOp(A, B, way, Op, X):
    Bdag = np.transpose(np.conjugate(B), (1, 0, 2))
    print "Shapes A", A.shape, "B", B.shape, "Op", Op.shape, "X", X.shape

    if way == 'R':
        XBdag = np.tensordot(X, Bdag, axes=([1,0]))
        AXBdag = np.tensordot(A, XBdag, axes=([1,0]))
        OpAXBdag = np.tensordot(Op, AXBdag, axes=([0,1], [3,1]))
        # print "Shape supOp R", OpAXBdag.shape, "\n", OpAXBdag

        return OpAXBdag
    else:
        XA = np.tensordot(X, A, axes=([1,0]))
        BdagXA = np.tensordot(Bdag, XA, axes=([1,0]))
        OpBdagXA = np.tensordot(Op, BdagXA, axes=([0,1], [1,3]))
        # print "Shape supOp L", OpBdagXA.shape, "\n", OpBdagXA

        return OpBdagXA

def meanVals(A, Lambda):
    R = L = Lambda
    Sz = np.diag([1., -1.])#np.kron(np.diag([1., -1.]), np.eye(np.sqrt(d)))
    toR = supOp(A, A, 'R', Sz, R)
    mvSz = np.trace(np.dot(L, toR))
    #toL = supOp(A, A, 'L', Sz, L)
    #mvSz = np.trace(np.dot(toL, R))
    print "<Sz>", mHz, mvSz

def calcYforZZs(C, Lambda, VL, VR):
    VLdag = np.transpose(np.conjugate(VL), (1, 0, 2))
    VRdag = np.transpose(np.conjugate(VR), (1, 0, 2))
    L_s = R_s = np.sqrt(Lambda)
    # print "Lambda Lala\n", Lambda, "\nL_s\n", L_s, "\nC\n", C

    RsVRdag = np.tensordot(R_s, VRdag, axes=([1,0]))
    CRsVRdag = np.tensordot(C, RsVRdag, axes=([1,3], [0,2]))
    LsCRsVRdag = np.tensordot(L_s, CRsVRdag, axes=([1,0]))
    Y = np.tensordot(VLdag, LsCRsVRdag, axes=([1,2], [0,1]))
    print "theY", Y.shape, "\n", Y

    return Y

def calcZ01andZ10(Y, MPS):
    try:
        U, S, V = spla.svd(Y, full_matrices=True)
    except spla.LinAlgError as err:
        if 'empty' in err.message:
            row, col = Y.shape
            Z01 = np.array([], dtype=Y.dtype).reshape(row, 0)
            Z10 = np.array([], dtype=Y.dtype).reshape(0, col)
            print "Empty", Z01.shape, Z10.shape
        else:
            print >> sys.stderr, "calcZ01andZ10: Error", I, err
            raise
    else:
        print "S", S, "\nU", U, "\nV", V
        __, chi, __ = MPS.shape
        mask = (S > expS) #np.array([True] * S.shape[0])
        mask[xiTilde - chi:] = False
        U = np.compress(mask, U, 1)
        S = np.compress(mask, S, 0)
        V = np.compress(mask, V, 0)

        Ssq = np.diag(np.sqrt(S))
        Z01 = np.dot(U, Ssq)
        Z10 = np.dot(Ssq, V)
        print "Fill ", U.shape, V.shape, "mask", mask

    eps = np.linalg.norm(np.dot(Z01, Z10))
    print "eps", I, eps
    print "Z01", Z01.shape, "\n", Z01, "\nZ10", Z10.shape, "\n", Z10

    return Z01, Z10

def getB01andB10(Z01, Z10, Lambda, VL, VR, thld = 1.e-10):
    Ld = np.diag(Lambda)
    noZeros = np.count_nonzero(Ld > thld)
    Ld_s = np.sqrt(Ld)
    Ld_si = np.zeros(Ld.shape, dtype=Ld.dtype)
    Ld_si[-noZeros:] = 1. / Ld_s[-noZeros:]
    L_si = R_si = np.diag(Ld_si)#np.diag(1. / np.sqrt(np.diag(Lambda)))
    # print "Lambda_si\n", L_si

    row, col = Z01.shape
    if row * col == 0:
        row, __ = Lambda.shape
        __, __, aux = VL.shape
        B01 = np.array([], dtype=Z01.dtype).reshape(row, col, aux)
    else:
        L_siVL = np.tensordot(L_si, VL, axes=([1,0]))
        B01 = np.tensordot(L_siVL, Z01, axes=([1,0]))
        B01 = np.transpose(B01, (0, 2, 1))

    print "Done01", L_si.shape, VL.shape, Z01.shape, B01.shape

    row, col = Z10.shape
    if row * col == 0:
        __, col = Lambda.shape
        __, __, aux = VR.shape
        B10 = np.array([], dtype=Z10.dtype).reshape(row, col, aux)
    else:
        z10VR = np.tensordot(Z10, VR, axes=([1,0]))
        B10 = np.tensordot(z10VR, R_si, axes=([1,0]))
        B10 = np.transpose(B10, (0, 2, 1))

    print "Done10", R_si.shape, VR.shape, Z10.shape, B10.shape
    print "B01\n", B01, "\nB10\n", B10

    return B01, B10

def doUpdateAndExpandA(B, B01, B10, MPS):
    rB, cB, aB = B.shape
    rB01, cB01, __ = B01.shape
    rB10, cB10, __ = B10.shape

    A = np.zeros((rB + rB10, cB + cB01, aB), dtype=B.dtype)
    A[:rB, :cB, :] = MPS - dTau * B
    A[:rB01, cB:, :] = np.sqrt(dTau) * B01
    A[rB:, :cB10, :] = - np.sqrt(dTau) * B10
    nDr, nDc, __ = A.shape
    print "Expanding", B.shape, B01.shape, B10.shape, A.shape, "\n", A

    return A, nDr, nDc

def doDynamicExpansion(MPS, Lambda, C, VR, B):
    VL = nullSpaceL(MPS, Lambda)

    Y = calcYforZZs(C, Lambda, VL, VR)
    Z01, Z10 = calcZ01andZ10(Y, MPS)
    B01, B10 = getB01andB10(Z01, Z10, Lambda, VL, VR)
    nA, nXir, nXic = doUpdateAndExpandA(B, B01, B10, MPS)
    print "hex(MPS)", hex(id(MPS)), "hex(newA)", hex(id(nA))
    print "Before", MPS.shape, "After ", nA.shape

    return nA, nXir, nXic

def appendThem(A, L, K):
    chir, chic, __ = A.shape
    oxir, oxic = K.shape

    nL, nK = np.zeros((chir, chic)), np.zeros((chir, chic))
    nL[:oxir, :oxic] = L
    nK[:oxir, :oxic] = K
    # print "append L\n", nL, "\nappend K\n", nK

    return nL, nK

def truncation(MPS, Lambda, thld = 1.e-10):
    global xi, xir, xic, xiTilde
    truncate, whenToTrunc = False, d
    chi, chi, __ = MPS.shape
    newChi = np.count_nonzero(Lambda > thld)

    if newChi < chi and chi - newChi > whenToTrunc:
        truncate = True
        Lambda = np.diag(Lambda)
        newLambda = np.diag(Lambda[-newChi:])
        newMPS = MPS[-newChi:, -newChi:, :]
        xir, xic, __ = newMPS.shape
        xi, xi, __ = newMPS.shape
        xiTilde = xi * d
        print "truncation: reducing from", chi, "to", newChi, xi
        newLambda, newMPS = symmNormNew(newMPS, xir, xic, newLambda)

        return newLambda, newMPS, truncate

    return Lambda, MPS, truncate



"""Main...
"""
#np.random.seed(9)
d, xi, xiTilde = 2, 4, 8
Jex, mHz = 1.0, float(sys.argv[1])
I, maxIter, expS, dTau = 0, 9000, 1.e-12, 0.1

xir = xic = xi
theMPS = np.ones((xir, xic, d))
theMPS = np.random.rand(xir, xic, d) - .5# + 1j * (np.random.rand(xir, xic, d) - .5)
theL, theK = np.random.rand(xir, xic) - .5, np.random.rand(xir, xic) - .5; theL += theL.T
print "theMPS", type(theMPS), theMPS.dtype, "\n", theMPS

theH = buildLocalH(Jex, mHz)

while True:#I != maxIter:
    theL, theMPS = symmNormNew(theMPS, xir, xic, theL)
    print "theMPS\n", theMPS, "\ntheL\n", theL

    meanVals(theMPS, theL)

    theC = buildHElements(theMPS, theH)
    print "theC =", theC.shape

    theK = calcHmeanval(theMPS, theL, theC, theK)
    print "theK =", theK.shape

    theVR = nullSpaceR(theMPS, theL)
    print "theVR =", theVR.shape

    theF = calcFs(theMPS, theC, theL, theK, theVR)
    print "theF =", theF.shape

    theB = getUpdateB(theL, theF, theVR)
    print "theB =", theB.shape

    eta, thold = np.linalg.norm(theF), 1.e-5 if xi == 1 else 1.e8 * expS
    print "eta", I, eta, xi, xiTilde
    if eta < thold:
        if xiTilde < 1965:
            theMPS, xir, xic = doDynamicExpansion(theMPS, theL, theC, theVR, theB)
            xi, xiTilde = xir, xir * d
            print "InMain", xi, xir, xic, xiTilde, theMPS.shape
            theL, theK = appendThem(theMPS, theL, theK)
        else:
            break
    else:
        theMPS = doUpdateForA(theMPS, theB)

    I += 1
