#/usr/bin/python

import numpy as np
#from scipy import linalg
import cmath

np.set_printoptions(suppress=True, precision=3)

#left-canonical orthonormalization
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

        Q, R = np.linalg.qr(MPS[n])

        MPS[n] = Q.copy()
        aux, chic[n] = MPS[n].shape

        i = np.random.randint(0, chic[n])
        j = np.random.randint(0, chic[n])
        dot = np.dot(Q[:,i], Q[:,j])
        print "dot =", i, j, round(dot, 10)

        print "MPS shape", MPS[n].shape
        MPS[n] = MPS[n].reshape((chir[n], d, chic[n]))
        print "MPS shape", MPS[n].shape
        MPS[n] = np.transpose(MPS[n], (0, 2, 1))
        print "MPS shape", MPS[n].shape

        print "Q[",n,"] =", Q
        print "R[",n,"] =", R

        if(n != length-1):
            print "shape =", MPS[n+1].shape
            MPS[n+1] = np.tensordot(R, MPS[n+1], axes=([1, 0]))
            chir[n+1], aux, aux = MPS[n+1].shape
            print "shape =", MPS[n+1].shape
            del Q, R, aux, i, j

#right-canonical orthonormalization
def rightNormalization(MPS, chir, chic):
    """Returns a left-normalized MPS.
    """
    for n in range(length):
        ip = length-n-1
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", ip
        print "MPS shape", MPS[ip].shape
        MPS[ip] = MPS[ip].reshape((chir[ip], chic[ip] * d))
        print "MPS shape", MPS[ip].shape

        MPS[ip] = np.conjugate(MPS[ip].T)
        print "MPS+ shape", MPS[ip].shape
        Q, R = np.linalg.qr(MPS[ip])

        Q = np.conjugate(Q.T)
        R = np.conjugate(R.T)

        MPS[ip] = Q.copy()
        chir[ip], aux = MPS[ip].shape

        i = np.random.randint(0, chir[ip])
        j = np.random.randint(0, chir[ip])
        dot = np.dot(Q[i,:], Q[j,:])
        print "dot =", i, j, round(dot, 10)

        print "MPS shape", MPS[ip].shape
        MPS[ip] = MPS[ip].reshape((chir[ip], chic[ip], d))
        print "MPS shape", MPS[ip].shape
    #MPS[n] = np.transpose(MPS[n], (0, 2, 1))
    #print "MPS shape", MPS[n].shape

        print "Q[",n,"] =", Q
        print "R[",n,"] =", R

        if(n != length-1):
            print "shape =", MPS[ip-1].shape
            MPS[ip-1] = np.tensordot(MPS[ip-1], R, axes=([1, 0]))
            print "shape =", MPS[ip-1].shape
            MPS[ip-1] = np.transpose(MPS[ip-1], (0, 2, 1))
            aux, chic[ip-1], aux = MPS[ip-1].shape
            print "shape =", MPS[ip-1].shape, chir[ip-1], chic[ip-1]
            del Q, R, aux

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

    return C

def calcLs(MPS):
    """Calculates the list of L matrices.

    Association of matrix multiplication different as in calcRs.
    """
    L = []#np.ones((1,1))]

    for n in range(length):
        if(n == 0):
            LA = MPS[n]
        else:
            LA = np.tensordot(L[n-1], MPS[n], axes=([1,0]))

        A = np.transpose(MPS[n], (1, 0, 2))
        A = np.conjugate(A)
        tmp = np.tensordot(A, LA, axes=([1,2], [0,2]))
        print "A =", MPS[n].shape, "LA =", LA.shape, "tmp =", tmp.shape
        print tmp

        L.append(tmp)

    return L

def calcRs(MPS):
    """Calculates the list of R matrices.

    Appends the R's as in calcLs and then reverses the list.
    The multiplication of the matrices is associated differently.
    """
    R = []#np.ones((1,1))]

    for n in range(length):
        ip = length-n-1; #print "ip =", ip

        if(n == 0):
            AR = MPS[ip]
            AR = np.transpose(AR, (0, 2, 1))
        else:
            AR = np.tensordot(MPS[ip], R[n-1], axes=([1,0]))

        A = np.transpose(MPS[ip], (1, 0, 2))
        A = np.conjugate(A)
        tmp = np.tensordot(AR, A, axes=([1,2], [2,0]))
        print "A =", MPS[ip].shape, "AR =", AR.shape, "tmp =", tmp.shape
        print tmp

        R.append(tmp)

    R.reverse()
    return R

def calcHmeanval(MPS, C):
    """Computes the matrcies K(n) which give the mean value of H.

    This is done in two steps: one for each term defining K(n).
    """
    dim, aux, aux = MPS[length-1].shape
    K = [np.zeros((dim,dim))]

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

    K.reverse()
    return K

def nullSpaceR(MPS):
    """Calculates the auxiliary matrix R to get its null space.

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

        if((chir * aux) != chic): #not square
            dimS, = S.shape
            VRdag = U[:,dimS:]
        else: #square matrix
            mask = (S < epsS) #try: np.select(...)
            VRdag = np.compress(mask, U, axis=1)

        R = np.conjugate(np.transpose(R))
        Null = np.tensordot(R, VRdag, axes=([1,0]))
        #VR.append(VRdag.size)
        VR.append(np.transpose(VRdag))

        print "D =", n, R.T.shape, U.shape, S.shape, V.shape, VRdag.shape
        print U; print S; print VRdag; print Null

    return VR



"""Main...
"""
length = 4
d = 2
xi = 3
epsS = 1e-15

xir = [xi for n in range(length)]
xic = [xi for n in range(length)]
xir[0] = xic[length-1] = 1
theMPS = [np.random.rand(xir[n], xic[n], d) for n in range(length)]
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

leftNormalization(theMPS, xir, xic)
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

rightNormalization(theMPS, xir, xic)
print "xir =", xir, "\nxic =", xic, "\ntheMPS =", theMPS

theH = buildLocalH()
print "theH =", theH[0].reshape((d*d, d*d))#"h =", theH
theC = buildHElements(theMPS, theH)
print "theC =", len(theC)

theL = calcLs(theMPS)
print "theL =", len(theL)
theR = calcRs(theMPS)
print "theR =", len(theR)

theK = calcHmeanval(theMPS, theC)
print "theK =", theK

theVR = nullSpaceR(theMPS)
print "theVR =", theVR

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
