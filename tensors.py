#/usr/bin/python

import numpy as np
from scipy import linalg
import cmath

#np.set_printoptions(suppress=True, precision=13)

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

def buildLocalH():
    """Builds local hamiltonian (dxd matrix).

    This function should call several ones depending on the 
    number of terms in H.
    Returns a list of local hamiltonians, where each term 
    is a (d, d, d, d)-rank tensor.
    """
    localH = np.zeros((d, d, d, d))
    localH[0,0,0,0] = localH[1,1,1,1] = 1./4.
    localH[0,1,0,1] = localH[1,0,1,0] = -1./4.
    localH[1,0,0,1] = localH[0,1,1,0] = 1./2.
    h = [localH for n in range(length)]

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
        del AA, tmp

    #print C
    return C

def calcLs(MPS):
    """Calculates the list of L matrices.

    Association of matrix multiplication different as in calcRs.
    SYMMETRIZATION OF L(n) BY HAND.
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
    SYMMETRIZATION OF R(n) BY HAND.
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
            Lsqrti = 1./L[n-1]; Lsqrti = np.diag(map(np.sqrt, Lsqrti))

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


"""Main...
"""
length = 8
d = 2
xi = 5
epsS = 1e-12

xir = [d**(length/2) for n in range(length)]
xir[0] = 1
xic = np.roll(xir, -1).tolist()
theMPS = [np.random.rand(xir[n], xic[n], d)-0.5 for n in range(length)]
#[np.random.randint(1, 1e9, (xir[n], xic[n], d)) for n in range(length)]
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

leftNormalization(theMPS, xir, xic)
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

rightNormalization(theMPS, xir, xic)
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

theH = buildLocalH()
print "theH =", theH[0].reshape((d*d, d*d))#"h =", theH
theC = buildHElements(theMPS, theH)
print "theC =", len(theC)

theR = calcRs(theMPS)
print "theR =", len(theR)
theL = calcLs(theMPS)
print "theL =", len(theL)

theK = calcHmeanval(theMPS, theC)
print "theK =", theK

theVR = nullSpaceR(theMPS)
print "theVR =", map(np.shape, theVR)#, theVR

theF = calcFs(theMPS, theC, theL, theK, theVR)
print "theF =", map(np.shape, theF)

exit()

"""
del R
print R

exit()

A = np.random.rand(chi*d*chi)
#A = A.astype(complex)
A = A.reshape((chi * d, chi))

Q, R = linalg.qr(A, mode='economic')

zrows = np.zeros((chi*d-chi, chi))
R = np.vstack((R, zrows))

print "A =", A#.shape
print "Q =", Q#.shape
print "R =", R#.shape

B = np.random.rand(chi*d*chi)
#B = A.astype(complex)
B = B.reshape((chi * d, chi))

print "B =",B

B = R * B

Q2, R2 = linalg.qr(B, mode='economic')

R2 = np.vstack((R2, zrows))

print "B =", B#.shape
print "Q2 =", Q2#.shape
print "R2 =", R2#.shape

C = np.random.rand(chi*d*chi)
#B = A.astype(complex)
C = C.reshape((chi * d, chi))

print "C =",C

C = R2 * C

Q3, R3 = linalg.qr(C, mode='economic')

R3 = np.vstack((R3, zrows))

print "C =", C#.shape
print "Q3 =", Q3#.shape
print "R3 =", R3#.shape

exit()

A = Q.copy()
"""
