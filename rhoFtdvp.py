#/usr/bin/python

import copy as cp
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

    return max(chir)

def rightNormalization(MPS, chir, chic):
    """Returns a right-normalized MPS.
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

    return max(chic)

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

    print "theH", type(h), len(h), "Jex", Jh, "hz", hz
    for n in range(length-1): print h[n].reshape(d * d, d * d)

    return h

def buildHElements(MPS, H):
    """Builds the matrix elements of the hamiltonian.

    Not sure how many of these are: N or N-1?
    Returns a tensor of the form C[a, b, s, t].
    """
    C = []

    for n in range(length-1):
        AA = np.tensordot(MPS[n], MPS[n+1], axes=([1,0]))
        tmp = np.tensordot(H[n], AA, axes=([2,3], [1,3]))
        tmp = np.transpose(tmp, (2, 3, 0, 1))

        C.append(tmp)
        print "C.shape", tmp.shape
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

    L.append(np.asarray([1.]))
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
    R.append(np.asarray([1.]))
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
    print "Energy", I, K[0][0,0]
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

def nullSpaceL(MPS, rho_):
    """Calculates the auxiliary matrix L and its null space VL.

    The VL matrices are returned as the tensor VL[a, b, s].
    Given a fixed gauge condition the L gets simplified.
    Notice that if VL.size = 0 there's no update B[x](n) because
    VL(n) will be an empty array. However, all the operations
    associated with VL are still "well-defined".
    """
    VL = []

    for n in range(length):
        Adag = np.transpose(np.conjugate(MPS[n]), (1, 0, 2))
        if n == 0: Lsqrt = np.ones((1,1))
        else:      Lsqrt = np.diag(map(np.sqrt, rho_[n-1]))
        L = np.tensordot(Adag, Lsqrt, axes=([1,0]))
        chir, aux, chic = L.shape

        L = np.reshape(L, (chir, aux * chic))
        U, S, V = np.linalg.svd(L, full_matrices=True)

        mask = np.empty(aux * chic, dtype=bool)
        mask[:] = False; mask[chir:] = True
        VLdag = np.compress(mask, V, axis=0)

        L = np.conjugate(np.transpose(L))
        Null = np.tensordot(VLdag, L, axes=([1,0]))
        Id = np.tensordot(VLdag, VLdag.T, axes=([1,0]))

        tmp = np.conjugate(VLdag.T)
        lpr, lmz = tmp.shape
        tmp = np.reshape(tmp, (aux, chic, lmz))
        tmp = np.transpose(np.transpose(tmp, (2, 1, 0)), (1, 0, 2))
        VL.append(tmp)

        #print "D =", n, L.T.shape, U.shape, Sp.shape, V.shape, VLdag.shape
        print "mask =", mask, S, "\nV\n", V, "\nVLdag\n", VLdag, \
            "\nNull\n", Null, "\nV+V\n", Id
        del L, U, S, V, VLdag, Null, Id

    return VL

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
        # print "Lsqrt =", Lsqrt.shape

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
        print "F", n, "\n", tmp
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

        print "B", n, Lsqrti.shape, x[n].shape, VR[n].shape, tmp.shape
        print tmp
        B.append(tmp)

    return B

def doUpdateForA(MPS, B):
    """It does the actual update to the MPS state for given time step.

    The update is done according to the formula:
    A[n, t + dTau] = A[n, t] - dTau * B[x*](n),
    where dTau is the corresponding time step.
    """
    A = [None] * length

    for n in range(length):
        A[n] = MPS[n] - dTau * B[n]
        print "cmp shapes =", n, A[n].shape, B[n].shape

    nA = cp.deepcopy(A)
    nXir = cp.deepcopy(xir)
    nXic = cp.deepcopy(xic)
    xiLoc = leftNormalization(nA, nXir, nXic)
    xiLoc = rightNormalization(nA, nXir, nXic)
    faith = np.eye(1)

    for n in range(length):
        faith = supOp(nA[n], MPS[n], 'L', np.eye(d), faith)
        # print "Fee", faith.shape
    print "Fee", I, np.abs(np.trace(faith))

    return A

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

def meanVals(A, L, R):
    Sz, mvSz = np.diag([1., -1.]), .0
    for k in range(length):
        toR = supOp(A[k], A[k], 'R', Sz, np.diag(R[k+1]))
        szk = np.trace(np.diag(L[k-1]).dot(toR))
        mvSz += szk
        print "Something", k, szk
        #toL = supOp(A[k], A[k], 'L', Sz, np.diag(L[k-1]))
        #print "Something", k, np.trace(toL.dot(np.diag(R[k+1])))
    print "<Sz>", mGh, mvSz

def calcYforZZs(C, L, VL, VR):
    """Calculates the quantity G(n,n+1) defined in J.H. thesis.

    Auxiliary routine used in the dynamic expansion of the
    variational manifold.

    Y(n,n+1) or G(n,n+1) basically corresponds to the two-site
    Hamiltonian applied to the MPS as done in TEBD. Notice that
    there are length-1 of such Y matrices. Its dimensions
    correspond to (d*D[n-1]-D[n]) x (q*D[n+1]-D[n]).

    When either left or right null spaces, VR and VL respectively,
    are empty imply that Y will be empty as well. In this case we
    define Y as an empty array of the proper shape. Alternatively
    we could also define Y as None.

    Be especially careful in the way such empty/None matrices are
    treated when calculating the corresponding Z's or B's in order
    to have properly define matrix operations.
    """
    Y = []

    for n in range(length-1):
        if n == 0: Lsqrt = np.ones((1,1))
        else:      Lsqrt = np.diag(map(np.sqrt, L[n-1]))
        VLdag = np.transpose(np.conjugate(VL[n]), (1, 0, 2))
        VRdag = np.transpose(np.conjugate(VR[n+1]), (1, 0, 2))
        print "For Y", VLdag.shape, Lsqrt.shape, C[n].shape, VRdag.shape

        CVRdag = np.tensordot(C[n], VRdag, axes=([1,3], [0,2]))
        LsCVRdag = np.tensordot(Lsqrt, CVRdag, axes=([1,0]))
        currY = np.tensordot(VLdag, LsCVRdag, axes=([1,2], [0,1]))

        Y.append(currY)

    print "theY =", len(Y), map(np.shape, Y)
    return Y

def calcZ01andZ10(Y, MPS):
    """Returns the parametrizations Z01[n] and Z10[n].

    Auxiliary routine that calculates the matrices Z01[n] and Z10[n]
    used in the calculation of the tangent vectors B01[n] and B10[n].
    These vectors are necessary when expanding the dynamics of the
    time evolution, a.k.a. expansion of the manifold.

    The dimensions of the matrices are (q*D[n-1]-D[n]) x g[n] for
    Z01[n] and g[n] x (q*D[n+1]-D[n]) for Z10[n].

    Notice that we haven't defined a tildeD[n] that sets how large
    the variational manifold can grow locally. Ideally,
    g[n] = min(tildeD[n]-D[n], gamma[n]),
    where gamma[n] is the dimension resulting from the Y[n]'s svd.

    !!!! WARNING: !!!! We haven't defined such tildeD[n] rendering
    the entire evolution "exact", since there's no truncation.
    IS THIS TRUE?

    The routine doesn't treat the boundaries leave the Z's as None.
    Now it DOES...
    Similar as in calcYforZZs(...) we treat the Z's with empty null
    spaces as empty arrays of appropriate dimensions.
    """
    Z01, Z10, EPS = [None] * length, [None] * length, []

    for n in range(length-1):
        #print "Doing", n

        try:
            U, S, V = np.linalg.svd(Y[n], full_matrices=False)
        except np.linalg.LinAlgError as err:
            if 'empty' in err.message:
                row, col = Y[n].shape
                Z01[n] = np.array([], dtype=Y[n].dtype).reshape(row, 0)
                Z10[n+1] = np.array([], dtype=Y[n].dtype).reshape(0, col)
                print "Empty", n, Z01[n].shape, n+1, Z10[n+1].shape
            else:
                raise
        else:
            print "S", S, "\nU", V, "\nU", V
            __, chic, __ = MPS[n].shape
            mask = (S > epsS) #np.array([True] * S.shape[0])
            mask[xiTilde-chic:] = False
            U = np.compress(mask, U, 1)
            S = np.compress(mask, S, 0)
            V = np.compress(mask, V, 0)

            Ssq = np.diag(np.sqrt(S))
            Z01[n] = U.dot(Ssq)
            Z10[n+1] = Ssq.dot(V)
            print "Fill ", n, U.shape, n+1, V.shape, "mask", mask
        EPS.append(np.dot(Z01[n], Z10[n+1]))

    # treating boundaries as empty arrays not as None
    Z01[-1] = np.array([], dtype=Y[-1].dtype).reshape(0, 0)
    Z10[0] = np.array([], dtype=Y[0].dtype).reshape(0, 0)
    print "Z01", map(np.shape, Z01), "\nZ10", map(np.shape, Z10)

    eps = np.linalg.norm(map(np.linalg.norm, EPS))
    print "eps", I, eps

    return Z01, Z10

def getB01andB10(Z01, Z10, rho, VL, VR):
    """Returns tangent vectors for expanding the manifold.

    Be careful with the tensor product when there is no expansion
    to be performed. Set those tensors to be empty. It's perfectly
    OK to do so, since patching an empty array with some zero
    dimensions and one with finite dimensions is well defined.
    """
    B01, B10 = [], []

    for n in range(length):
        if n == 0: Lsi = np.ones((1,1))
        else:      Lsi = np.diag(map(np.sqrt, 1./rho[n-1]))

        row, col = Z01[n].shape
        if row * col == 0:
            row, row = Lsi.shape
            lpr, lmz, aux = VL[n].shape
            tmp01 = np.array([], dtype=Z01[n].dtype).reshape(row, col, aux)
        else:
            LsiVL = np.tensordot(Lsi, VL[n], axes=([1,0]))
            tmp01 = np.tensordot(LsiVL, Z01[n], axes=([1,0]))
            tmp01 = np.transpose(tmp01, (0, 2, 1))

        print "Done01", n, Lsi.shape, VL[n].shape, Z01[n].shape, tmp01.shape
        B01.append(tmp01)

    for n in range(length):
        row, col = Z10[n].shape
        if row * col == 0:
            lpr, col, aux = VR[n].shape
            tmp10 = np.array([], dtype=Z10[n].dtype).reshape(row, col, aux)
        else:
            tmp10 = np.tensordot(Z10[n], VR[n], axes=([1,0]))

        print "Done10", n, Lsi.shape, VR[n].shape, Z10[n].shape, tmp10.shape
        B10.append(tmp10)

    return B01, B10

def doUpdateAndExpandA(B, B01, B10, MPS):
    """Does last update for A[n] and expansion of manifold.

    Does the update: A[n, t + dTau] = A[n, t] - dTau * B[x* = F](n),
    and in addition also expands A[n, t + dTau] with the corresponding
    updates proportional to sqrt(dTau) * [B01[n] or B10[n]].
    """
    print "B  ", map(np.shape, B)
    print "B01", map(np.shape, B01)
    print "B10", map(np.shape, B10)

    newAs, newXir, newXic = [], [None] * length, [None] * length

    for n in range(length):
        rB, cB, aB = B[n].shape
        rB01, cB01, aB01 = B01[n].shape
        rB10, cB10, aB10 = B10[n].shape

        tmp = np.zeros((rB + rB10, cB + cB01, aB), dtype=B[n].dtype)
        tmp[:rB, :cB, :] = MPS[n] - dTau * B[n]
        tmp[:rB01, cB:, :] = np.sqrt(dTau) * B01[n]
        tmp[rB:, :cB10, :] = -np.sqrt(dTau) * B10[n]

        newAs.append(tmp)
        newXir[n], newXic[n], aux = tmp.shape
        print "Expanding", n, B[n].shape, B01[n].shape, B10[n].shape, tmp.shape
        print tmp

    nA = cp.deepcopy(newAs)
    nXir = cp.deepcopy(newXir)
    nXic = cp.deepcopy(newXic)
    xiLoc = leftNormalization(nA, nXir, nXic)
    xiLoc = rightNormalization(nA, nXir, nXic)
    faith = np.eye(1)

    for n in range(length):
        faith = supOp(nA[n], MPS[n], 'L', np.eye(d), faith)
        # print "Fee", faith.shape
    print "Fee", I, np.abs(np.trace(faith)), xiTilde

    # MPS = newAs
    return newAs, newXir, newXic

def doDynamicExpansion(MPS, L, C, VR, B):
    """Auxiliary calling function to perform the dynamic expansion.

    This routine should be used instead of doUpdateForA(...) when
    trying to perform an update and expansion of the variational
    manifold defined by the MPS A[n,t] at some given time t.

    The idea is to use this routine and doUpdateForA(...) together
    with the flags |x*| and dynexp, in the following way:

    if |x*| > eps and dynexp == False:
        doUpdateForA(...)
    else if |x*| < eps and dynexp == True:
        doDynamicExpansion(...)
    else if |x*| < eps and dynexp == False:
        doMeasuremnts(...)
        break  #we're done!
    """
    VL = nullSpaceL(MPS, L)
    print "VL =", map(np.shape, VL)

    Y = calcYforZZs(C, L, VL, VR)
    Z01, Z10 = calcZ01andZ10(Y, MPS)
    B01, B10 = getB01andB10(Z01, Z10, L, VL, VR)
    newA, newXr, newXc = doUpdateAndExpandA(B, B01, B10, MPS)
    print "hex(MPS)", hex(id(MPS)), "hex(newA)", hex(id(newA))
    print "Before", map(np.shape, MPS), "\nAfter ", map(np.shape, newA)
    return newA, newXr, newXc



"""Main...
"""
np.random.seed(10)
d, xi, xiTilde = 2, 1, 2
length, Jex, mGh = 6, 1.0, float(sys.argv[1])
maxIter, epsS, dTau = 10000, 1e-12, 0.051

xir = [xi * d for n in range(length)]
xir[0] = 1
xic = np.roll(xir, -1).tolist()
theMPS = [np.random.rand(xir[n], xic[n], d)-0.5 for n in range(length)]
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

theH = buildLocalH(Jex, mGh)

I = 0
while I != maxIter:
    xi = leftNormalization(theMPS, xir, xic)
    print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

    xi = rightNormalization(theMPS, xir, xic)
    print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

    theR = calcRs(theMPS)
    print "theR =", len(theR)
    theL = calcLs(theMPS)
    print "theL =", len(theL)

    # meanVals(theMPS, theL, theR)

    theC = buildHElements(theMPS, theH)
    print "theC =", len(theC)

    theK = calcHmeanval(theMPS, theC)
    print "theK =", theK, "\ntheK =", map(np.shape, theK)

    theVR = nullSpaceR(theMPS)
    print "theVR =", map(np.shape, theVR)

    theF = calcFs(theMPS, theC, theL, theK, theVR)
    print "theF =", map(np.shape, theF)

    theB = getUpdateB(theL, theF, theVR)
    print "theB =", map(np.shape, theB)

    eta = np.linalg.norm(map(np.linalg.norm, theF))
    print "eta", I, eta, xi, xiTilde
    if eta < 100 * epsS:
        if xiTilde < 11:
            theMPS, xir, xic = doDynamicExpansion(theMPS, theL, theC, theVR, theB)
            xi, xiTilde = xiTilde, xiTilde + d
            print "InMain", map(np.shape, theMPS)
        else:
            break
    else: #if eta > 100 * epsS:
        theMPS = doUpdateForA(theMPS, theB)

    I += 1
