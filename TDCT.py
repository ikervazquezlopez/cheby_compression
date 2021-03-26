import numpy as np
import math
import sys

import config
import utils

from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs

BLOCK_SIZE = config.BLOCK_SIZE
VERBOSE = config.VERBOSE

#Initializes the required data for the TDCT
def init_data():
    # Generate the basis Chebysev polynomials in triangle
    dtt = createTriangleDCT(BLOCK_SIZE);
    dot = createTriangleDCT_ortho(dtt)

    dttp = createTriangleDCT_p(BLOCK_SIZE);
    dotp = createTriangleDCT_ortho_p(dttp)

    idx = np.tril(np.ones((BLOCK_SIZE,BLOCK_SIZE))) == 1
    idxp = np.tril(np.ones((BLOCK_SIZE-1,BLOCK_SIZE-1))) == 1

    data = {"dot": dot, "dotp": dotp, "idx": idx, "idxp": idxp}
    return data

# Evaluate the Chebysev B2 polynomials on zeros of ideal (common zeros of all basis)
def evalChebyshevB2(idxK,idxL,arg1,arg2):
    value = 0.25* (np.cos(2*math.pi*(idxK * arg1 + idxL * arg2)) +
                np.cos(2*math.pi*((idxK + idxL) * arg1 - idxL * arg2)) +
                np.cos(2*math.pi*(idxK * arg1 - (2*idxK + idxL) * arg2)) +
                np.cos(2*math.pi*((idxK + idxL)*arg1 - (2*idxK + idxL)*arg2)))
    return value


# Creates the DCT on triangles matrix
def createTriangleDCT(sizeN):
    # build all possible combinations for (k,j)
    tmp = np.array(np.meshgrid(list(range(sizeN)), list(range(1,2*sizeN,2))))
    # filter out the ones who satisfy constraint
    idx = 2*tmp[0] <= tmp[1]
    #create set of zeros of T_n
    z0 = tmp[0,idx] / (2*sizeN)
    z1 = tmp[1,idx] / (4*sizeN)
    zerosOfIdeal = np.array([z0,z1])

    chebIdx = np.array([[0],[0]])
    for k in range(1,sizeN):
        # build all possible indices (k,l)
        tmp = np.array(np.meshgrid(list(range(k+1)), list(range(k+1))))
        # filter out the ones who satisfy constraint
        idx = tmp[0,:] + tmp[1,:] == k
        tmp = tmp[:,idx]
        sortedtmp = tmp[tmp[:,0].argsort()]

        chebIdx = np.hstack((chebIdx, sortedtmp))

    # evaluate B2 Chebyshev polynomials on zeros to obtain DCT on triangles
    triangleDCT = []
    for i in range(0,chebIdx.shape[1]):
        triangleDCT.append(evalChebyshevB2(chebIdx[0,i],chebIdx[1,i],zerosOfIdeal[0,:],zerosOfIdeal[1,:]))
    triangleDCT = np.round(np.array(triangleDCT),decimals=4)
    return triangleDCT

def createTriangleDCT_p(sizeN):
    # build all possible combinations for (k,j)
    tmp = np.array(np.meshgrid(list(range(sizeN-1)), list(range(1,2*(sizeN-1),2))))
    # filter out the ones who satisfy constraint
    idx = 2*tmp[0] <= tmp[1]
    #create set of zeros of T_n
    z0 = tmp[0,idx] / (2*(sizeN-1))
    z1 = tmp[1,idx] / (4*(sizeN-1))
    zerosOfIdeal = np.array([z0,z1])

    chebIdx = np.array([[0],[0]])
    for k in range(1,sizeN-1):
        # build all possible indices (k,l)
        tmp = np.array(np.meshgrid(list(range(k+1)), list(range(k+1))))
        # filter out the ones who satisfy constraint
        idx = tmp[0,:] + tmp[1,:] == k
        tmp = tmp[:,idx]
        sortedtmp = tmp[tmp[:,0].argsort()]

        chebIdx = np.hstack((chebIdx, sortedtmp))

    # evaluate B2 Chebyshev polynomials on zeros to obtain DCT on triangles
    triangleDCT = []
    for i in range(0,chebIdx.shape[1]):
        triangleDCT.append(evalChebyshevB2(chebIdx[0,i],chebIdx[1,i],zerosOfIdeal[0,:],zerosOfIdeal[1,:]))
    triangleDCT = np.round(np.array(triangleDCT),decimals=4)
    return triangleDCT

# Creates the Orthogonal DCT on triangles matrix
def createTriangleDCT_ortho(dtt):
    np.set_printoptions(threshold=sys.maxsize)

    # Generate Hplus matrix
    H = [np.sqrt(2)]
    for sz in range(1,BLOCK_SIZE):
        H.append(np.sqrt(8))
        for _ in range(1,sz):
            H.append(4)
        H.append(np.sqrt(8))
    Hplus = np.diag(np.array(H))

    # Generate D matrix
    D = []
    for sz in range(0,BLOCK_SIZE):
        D.append(1/(np.sqrt(2)*BLOCK_SIZE))
        for _ in range(0,sz):
            D.append(1/BLOCK_SIZE)
    D = np.diag(np.array(D))

    # Compute matrix multiplication D*dtt*H
    triangleDCT = np.matmul(D, np.matmul(dtt, Hplus))

    return triangleDCT

def createTriangleDCT_ortho_p(dtt):
    np.set_printoptions(threshold=sys.maxsize)

    # Generate Hplus matrix
    H = [np.sqrt(2)]
    for sz in range(1,BLOCK_SIZE-1):
        H.append(np.sqrt(8))
        for _ in range(1,sz):
            H.append(4)
        H.append(np.sqrt(8))
    Hplus = np.diag(np.array(H))

    # Generate D matrix
    D = []
    for sz in range(0,BLOCK_SIZE-1):
        D.append(1/(np.sqrt(2)*BLOCK_SIZE-1))
        for _ in range(0,sz):
            D.append(1/(BLOCK_SIZE-1))
    D = np.diag(np.array(D))

    # Compute matrix multiplication D*dtt*H
    triangleDCT = np.matmul(D, np.matmul(dtt, Hplus))

    return triangleDCT


#=====================================================
# Triangular transforms of triangles
#=====================================================
def transform_triangle_forward(triangle, dtt, idx):
    if VERBOSE > 0:
        print("Transforming block to frequency domain...")

    transform = np.matmul(dtt,triangle[idx].astype(np.double))
    transform = np.reshape(transform,(transform.shape[0],1))

    if VERBOSE > 1:
        block = utils.reshape_array_to_block(transform)
        print("Transformed block (rounded for visualization):")
        print(np.round(block))
        print("=====================")

    return transform

def transform_triangle_inverse(transform, dtt, idx):
    if VERBOSE > 0:
        print("Inversing transform to image domain...")

    #transform = reshape_block_to_array(transform, transform.shape[0])
    reconstructed = linalg.lstsq(dtt, transform)[0]
    reconstructed = np.reshape(reconstructed,(reconstructed.shape[0],1))

    if VERBOSE > 1:
        block = utils.reshape_array_to_block(reconstructed)
        print("Inversed block:")
        print(block)
        print("=====================")

    return reconstructed



# Transform block into the triangle block frequency domain
def transform_block_TDCT_forward(block, data):
    J = utils.get_lower_triangle(block)
    Jp = utils.get_upper_triangle(block)

    lowerT = transform_triangle_forward(J,data['dot'],data['idx'])
    upperT = transform_triangle_forward(Jp,data['dotp'],data['idxp'])

    return lowerT, upperT

def transform_block_TDCT_inverse(lowerT, upperT, data):

    lowerT = transform_triangle_inverse(lowerT,data['dot'],data['idx'])
    upperT = transform_triangle_inverse(upperT,data['dotp'],data['idxp'])

    return lowerT, upperT
