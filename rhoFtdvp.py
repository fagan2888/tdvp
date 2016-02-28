#/usr/bin/python

import numpy as np
#from scipy import linalg
import cmath
import sys

np.set_printoptions(suppress=True, precision=13)

adj = lambda X: np.transpose(np.conjugate(X))

def leftNormalization(MPS, chir, chic):
    """Returns a left-normalized MPS.
    """
    for n in range(length):
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", n
        print "MPS shape", MPS[n].shape
        MPS[n] = np.transpose(MPS[n], (0, 2, 1))
        print "MPS shape", MPS[n].shape
        MPS[n] = MPS[n].reshape((chir[n] * d, chic[n]))
        print "MPS shape", MPS[n].shape

        U, S, V = np.linalg.svd(MPS[n], full_matrices=False)

        print "U\n", U; print "V\n", V
        print "S =", S, xi
        mask = (S > epsS)
        newXi = mask.tolist().count(True)
        print "S =", mask, newXi
        if(newXi < xi): mask[newXi:] = False
        else: mask[xi:] = False
        print "S =", mask

        U = np.compress(mask, U, 1)
        S = np.diag(np.compress(mask, S, 0))
        V = np.compress(mask, V, 0)
        SV = np.tensordot(S, V, axes=([1,0]))

        MPS[n] = U.copy()
        aux, chic[n] = MPS[n].shape

        #i = np.random.randint(0, chic[n])
        #j = np.random.randint(0, chic[n])
        for i in range(chic[n]):
            for j in range(i, chic[n]):
                dot = np.dot(U[:,i], U[:,j])
                print "dot =", i, j, round(dot, 10)

        print "MPS shape", MPS[n].shape
        MPS[n] = MPS[n].reshape((chir[n], d, chic[n]))
        print "MPS shape", MPS[n].shape
        MPS[n] = np.transpose(MPS[n], (0, 2, 1))
        print "MPS shape", MPS[n].shape

        print "U[",n,"] =\n", U
        print "V[",n,"] =\n", V

        if(n != length-1):
            print "shape =", MPS[n+1].shape
            MPS[n+1] = np.tensordot(SV, MPS[n+1], axes=([1, 0]))
            chir[n+1], aux, aux = MPS[n+1].shape
            print "shape =", MPS[n+1].shape
            del U, S, V, SV, aux, i, j

def rightNormalization(MPS, chir, chic):
    """Returns a left-normalized MPS.
    """
    for n in range(length):
        ip = length-n-1
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", ip
        print "MPS shape", MPS[ip].shape
        MPS[ip] = MPS[ip].reshape((chir[ip], chic[ip] * d))
        print "MPS shape", MPS[ip].shape

        U, S, V = np.linalg.svd(MPS[ip], full_matrices=False)

        print "U\n", U; print "V\n", V
        print "S =", S, xi
        mask = (S > epsS)
        newXi = mask.tolist().count(True)
        print "S =", mask, newXi
        if(newXi < xi): mask[newXi:] = False
        else: mask[xi:] = False
        print "S =", mask

        U = np.compress(mask, U, 1)
        S = np.diag(np.compress(mask, S, 0))
        V = np.compress(mask, V, 0)
        US = np.tensordot(U, S, axes=([1,0]))

        MPS[ip] = V.copy()
        chir[ip], aux = MPS[ip].shape

        #i = np.random.randint(0, chir[ip])
        #j = np.random.randint(0, chir[ip])
        for i in range(chir[ip]):
            for j in range(i, chir[ip]):
                dot = np.dot(V[i,:], V[j,:])
                print "dot =", i, j, round(dot, 10)

        print "MPS shape", MPS[ip].shape
        MPS[ip] = MPS[ip].reshape((chir[ip], chic[ip], d))
        print "MPS shape", MPS[ip].shape

        print "U[",n,"] =\n", U
        print "V[",n,"] =\n", V

        if(n != length-1):
            print "shape =", MPS[ip-1].shape
            MPS[ip-1] = np.tensordot(MPS[ip-1], US, axes=([1, 0]))
            print "shape =", MPS[ip-1].shape
            MPS[ip-1] = np.transpose(MPS[ip-1], (0, 2, 1))
            aux, chic[ip-1], aux = MPS[ip-1].shape
            print "shape =", MPS[ip-1].shape
            del U, S, V, US, aux, i, j

def buildLocalH(Jh, hz):
    """Builds local hamiltonian (dxd matrix).

    This function should call several ones depending on the 
    number of terms in H.
    Returns a list of local hamiltonians, where each term 
    is a (d, d, d, d)-rank tensor.
    """
    Sx, Sy = np.array([[0., 1.], [1., 0.]]), np.array([[0., -1.j], [1.j, 0.]])
    Sz, Id = np.diag([1., -1.]), np.eye(d)
    bulkLocH = Jh * np.kron(Sx, Sx) + Jh * np.kron(Sy, Sy) \
               + (hz / 2.) * (np.kron(Sz, Id) + np.kron(Id, Sz))
    lBryLocH = Jh * np.kron(Sx, Sx) + Jh * np.kron(Sy, Sy) \
               + hz * np.kron(Sz, Id) + (hz / 2.) * np.kron(Id, Sz)
    rBryLocH = Jh * np.kron(Sx, Sx) + Jh * np.kron(Sy, Sy) \
               + (hz / 2.) * np.kron(Sz, Id) + hz * np.kron(Id, Sz)

    h = [bulkLocH.real.reshape(d, d, d, d) for n in range(length-1-2)]
    h = [lBryLocH.real.reshape(d, d, d, d)] + h + [rBryLocH.real.reshape(d, d, d, d)]
    print "theH", type(h), len(h)

    return h

def buildHElements(MPS, H):
    """Builds the matrix elements of the hamiltonian.

    Not sure how many of these are: N or N-1?
    Returns a tensor of the form C[a, b, s, t].
    """
    C = []
    for n in range(length-1):
        print H[n].reshape(d * d, d * d)

    for n in range(length-1):
        AA = np.tensordot(MPS[n], MPS[n+1], axes=([1,0]))
        tmp = np.tensordot(H[n], AA, axes=([2,3], [1,3]))
        tmp = np.transpose(tmp, (2, 3, 0, 1))

        C.append(tmp)
        del AA, tmp

    #print C
    return C

def calcLs(MPS):
    """Calculates the list of L matrices.

    Association of matrix multiplication different as in calcRs.
    SYMMETRIZATION OF L(n) BY HAND (UNNECESSARY)
    """
    L = []#np.ones((1,1))]

    for n in range(length):
        if(n == 0):
            LA = MPS[n]
        else:
            LA = np.tensordot(np.diag(L[n-1]), MPS[n], axes=([1,0]))

        A = np.transpose(np.conjugate(MPS[n]), (1, 0, 2))
        tmp = np.tensordot(A, LA, axes=([2,1], [2,0]))
        tmp = 0.5 * (tmp + tmp.T)
        print "A =", MPS[n].shape, "LA =", LA.shape, "tmp =", \
            tmp.shape, np.trace(tmp), "\nl = \n", tmp

        L.append(tmp.diagonal())
        del LA, A, tmp

    return L

def calcRs(MPS):
    """Calculates the list of R matrices.

    Appends the R's as in calcLs and then reverses the list.
    The multiplication of the matrices is associated differently.
    SYMMETRIZATION OF R(n) BY HAND (SAME THING HERE).
    """
    R = []#np.ones((1,1))]

    for n in range(length):
        ip = length-n-1; #print "ip =", ip

        if(n == 0):
            AR = MPS[ip]
            AR = np.transpose(AR, (0, 2, 1))
        else:
            AR = np.tensordot(MPS[ip], np.diag(R[n-1]), axes=([1,0]))

        A = np.transpose(np.conjugate(MPS[ip]), (1, 0, 2))
        tmp = np.tensordot(AR, A, axes=([1,2], [2,0]))
        tmp = 0.5 * (tmp + tmp.T)
        print "A =", MPS[ip].shape, "AR =", AR.shape, "tmp =", \
            tmp.shape, np.trace(tmp), "\nr = \n", tmp

        R.append(tmp.diagonal())
        del AR, A, tmp

    R.reverse()
    return R

def calcHmeanval(MPS, C):
    """Computes the matrcies K(n) which give the mean value of H.

    This is done in two steps: one for each term defining K(n).
    """
    dim, aux, aux = MPS[length-1].shape
    K = [np.zeros((dim,dim))]
    print "shape =-1", length-1, K[0].shape

    for n in range(length-1):
        ip = length-n-1-1

        AA = np.tensordot(MPS[ip], MPS[ip+1], axes=([1,0]))
        AA = np.transpose(AA, (0, 2, 1, 3))
        AA = np.transpose(AA, (1, 0, 2, 3))
        AA = np.conjugate(AA)
        tmp = np.tensordot(C[ip], AA, axes=([1,2,3], [0,2,3]))

#        print "shape =", MPS[ip].shape, K[n].shape
        AK = np.tensordot(MPS[ip], K[n], axes=([1,0]))
        A = np.transpose(MPS[ip], (1, 0, 2))
        A = np.conjugate(A)
        tmp += np.tensordot(AK, A, axes=([1,2], [2,0]))
        print "shape =", n, ip, tmp.shape

        K.append(tmp)
        del AA, tmp, A

    K.reverse()
    return K

def nullSpaceR(MPS):
    """Calculates the auxiliary matrix R and its null space VR.

    The VR matrices are returned as the tensor VR[a, b, s].
    Given a fixed gauge condition the R gets simplified.
    Notice that if VR.size = 0 there's no update B[x](n) because
    VR(n) will be an empty array. However, all the operations
    associated with VR are still "well-defined".
    """
    VR = []

    for n in range(length):
        R = np.transpose(MPS[n], (1, 0, 2))
        R = np.conjugate(R)
        chir, chic, aux = R.shape

        R = np.transpose(R, (0, 2, 1))
        R = np.reshape(R, (chir * aux, chic))
        U, S, V = np.linalg.svd(R, full_matrices=True)

        #dimS, = S.shape
        #extraS = np.zeros((chir * aux) - dimS)
        #Sp = np.append(S, extraS, 0)
        #maskp = (Sp < epsS) #try: np.select(...)

        mask = np.empty(chir * aux, dtype=bool)
        mask[:] = False; mask[chic:] = True
        VRdag = np.compress(mask, U, axis=1)

        R = np.conjugate(np.transpose(R))
        Null = np.tensordot(R, VRdag, axes=([1,0]))
        Id = np.tensordot(VRdag.T, VRdag, axes=([1,0]))

        tmp = np.conjugate(VRdag.T)
        lpr, lmz = tmp.shape
        tmp = np.reshape(tmp, (lpr, chir, aux))
        VR.append(tmp)

        #print "D =", n, R.T.shape, U.shape, Sp.shape, V.shape, VRdag.shape
        print "mask =", mask, S, "\nU\n", U, "\nVRdag\n", VRdag, \
            "\nNull\n", Null, "\nVV+\n", Id
        del R, U, S, V, VRdag, Null, Id

    return VR

def calcFs(MPS, C, L, K, VR):
    """Returns the list of matrices x*(n) = F(n).

    The inversion and sqrt of the L/R is trivial since one of them is
    the identity and the other one is a diagonal matrix containing the
    eigenvalues of the reduced density matrix.
    """
    F = []

    for n in range(length):
        tmp1 = tmp2 = tmp3 = 0.
        VRdag = np.transpose(np.conjugate(VR[n]), (1, 0, 2))

        if(n == 0): Lsqrt = np.ones((1,1))
        else:       Lsqrt = np.diag(map(np.sqrt, L[n-1]))
        print "Lsqrt =", Lsqrt.shape

        if(n < length-1):
            A = np.transpose(np.conjugate(MPS[n+1]), (1, 0, 2))
            AVRdag = np.tensordot(A, VRdag, axes=([1,0]))
            CAVRdag = np.tensordot(C[n], AVRdag, axes=([1,2,3], [0,3,1]))
            tmp1 = np.tensordot(Lsqrt, CAVRdag, axes=([1,0]))
            #print tmp1

        if(n > 0):
            if(n == 1): Ltmp = np.ones((1,1))
            else:       Ltmp = np.diag(L[n-2])
            Lsqrti = np.diag(map(np.sqrt, 1./L[n-1]))

            CVRdag = np.tensordot(C[n-1], VRdag, axes=([1,3], [0,2]))
            LCVRdag = np.tensordot(Ltmp, CVRdag, axes=([1,0]))
            A = np.transpose(np.conjugate(MPS[n-1]), (1, 0, 2))
            ALCVRdag = np.tensordot(A, LCVRdag, axes=([1,2], [0,1]))
            tmp2 = np.tensordot(Lsqrti, ALCVRdag, axes=([1,0]))
            #print tmp2

        if(n < length-2):
            KVRdag = np.tensordot(K[n+1], VRdag, axes=([1,0]))
            AKVRdag = np.tensordot(MPS[n], KVRdag, axes=([1,2], [0,2]))
            tmp3 = np.tensordot(Lsqrt, AKVRdag, axes=([1,0]))
            #print tmp3

        tmp = tmp1 + tmp2 + tmp3
        F.append(tmp)
        print tmp
        del tmp1, tmp2, tmp3, tmp

    return F

def getUpdateB(L, x, VR):
    """Returns the list of updates B to be added to the current MPS.

    Be careful with the dot product when there is no update to be 
    performed. Set those tensors to zero with proper shape.
    """
    B = []

    for n in range(length):
        if(n == 0): Lsqrti = np.ones((1,1))
        else:       Lsqrti = np.diag(map(np.sqrt, 1./L[n-1]))

        row, col, aux = VR[n].shape
        if(row*col == 0):
            row, row = Lsqrti.shape
            tmp = np.zeros((row, col, aux))
        else:
            xVR = np.tensordot(x[n], VR[n], axes=([1,0]))
            tmp = np.tensordot(Lsqrti, xVR, axes=([1,0]))

        print n, Lsqrti.shape, x[n].shape, VR[n].shape, tmp.shape
        B.append(tmp)

    return B

def doUpdateForA(MPS, B):
    """
    It does the actual update to the MPS state for given time step.

    The update is done according to the formula:
    A[n, t + dTau] = A[n, t] - dTau * B[x*](n),
    where dTau is the corresponding time step.
    """
    for n in range(length):
        MPS[n] -= dTau * B[n]
        print "cmp shapes =", n, MPS[n].shape, B[n].shape

    return MPS

def supOp(A, B, way, Op, X):
    Bdag = np.transpose(np.conjugate(B), (1, 0, 2))
    print "Shapes A", A.shape, "B", B.shape, "Op", Op.shape, "X", X.shape

    if(way == 'R'):
        XBdag = np.tensordot(X, Bdag, axes=([1,0]))
        AXBdag = np.tensordot(A, XBdag, axes=([1,0]))
        OpAXBdag = np.tensordot(Op, AXBdag, axes=([0,1], [3,1]))
        print "Shape supOp R", OpAXBdag.shape
        print OpAXBdag
        
        return OpAXBdag
    else:
        XA = np.tensordot(X, A, axes=([1,0]))
        BdagXA = np.tensordot(Bdag, XA, axes=([1,0]))
        OpBdagXA = np.tensordot(Op, BdagXA, axes=([0,1], [1,3]))
        print "Shape supOp L", OpBdagXA.shape
        print OpBdagXA

        return OpBdagXA



"""Main...
"""
np.random.seed(9)
d, xi = 2, 8
length, Jex, mGh = 6, 1.0, float(sys.argv[1])
maxIter, epsS, dTau = 10000, 1e-12, 0.051

xir = [xi * d for n in range(length)]
xir[0] = 1
xic = np.roll(xir, -1).tolist()
theMPS = [np.random.rand(xir[n], xic[n], d)-0.5 for n in range(length)]
#[np.random.randint(1, 1e9, (xir[n], xic[n], d)) for n in range(length)]
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

theH = buildLocalH(Jex, mGh)
print "Jex", Jex, "hz", mGh, "theH =\n", theH[1].reshape((d*d, d*d))#"h =", theH

I = 0
while (I != maxIter):

    leftNormalization(theMPS, xir, xic)
    print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

    rightNormalization(theMPS, xir, xic)
    print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

    theR = calcRs(theMPS)
    print "theR =", len(theR)
    theL = calcLs(theMPS)
    print "theL =", len(theL)

    theL.append(np.asarray([1.]))
    theR.append(np.asarray([1.]))
    Sz, mvSz = np.diag([1., -1.]), .0
    for k in range(length):
        toR = supOp(theMPS[k], theMPS[k], 'R', Sz, np.diag(theR[k+1]))
        szk = np.trace(np.diag(theL[k-1]).dot(toR))
        mvSz += szk
        print "Something", k, szk
        #toL = supOp(theMPS[k], theMPS[k], 'L', Sz, np.diag(theL[k-1]))
        #print "Something", k, np.trace(toL.dot(np.diag(theR[k+1])))
    # break
    print "<Sz>", mGh, mvSz
    
    theC = buildHElements(theMPS, theH)
    print "theC =", len(theC)

    theK = calcHmeanval(theMPS, theC)
    print "theK =", theK

    theVR = nullSpaceR(theMPS)
    print "theVR =", map(np.shape, theVR)

    theF = calcFs(theMPS, theC, theL, theK, theVR)
    print "theF =", map(np.shape, theF)

    theB = getUpdateB(theL, theF, theVR)
    print "theB =", map(np.shape, theB)

    doUpdateForA(theMPS, theB)

    eta = np.linalg.norm(map(np.linalg.norm, theF))
    print "eta", I, eta
    if eta < 100 * epsS: break
    I += 1
