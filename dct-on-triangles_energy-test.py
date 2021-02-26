import numpy as np
import cv2
import math
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
import sys
from os.path import isfile, isdir, join
from os import listdir
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


SHOW_BASIS = False
VERBOSE = 0

BLOCK_SIZE = 8

ORTHOGONAL_TRANSFORM = True

SAMPLES = 100





#=====================================================
# Visualization of the T_k_l basis
#=====================================================
def generate_base_img(Tkl_array,triangle_h):
    triangle = np.zeros((triangle_h,triangle_h))
    i=0
    for y in range(0,triangle_h):
        for x in range(0,y+1):
            triangle[y,x] = Tkl_array[i]
            i = i+1
    return triangle

def generate_basis_img(dtt, triangle_h):
    i = 0
    basis = np.zeros((triangle_h*triangle_h, triangle_h*triangle_h))
    for l in range(0,triangle_h):
        for k in range(0,l+1):
            y_start = l*triangle_h
            y_end = y_start + triangle_h
            x_start = k*triangle_h
            x_end = x_start + triangle_h
            basis[y_start:y_end,x_start:x_end] = generate_base_img(dtt[i],triangle_h)
            i = i+1
    return basis






#=====================================================
# Utility functions
#=====================================================

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



def reshape_array_to_block(triangle_array):
    block = np.zeros((BLOCK_SIZE,BLOCK_SIZE))
    i = 0
    for y in range(0, BLOCK_SIZE):
        for x in range(0, y+1):
            block[y,x] = triangle_array[i]
            i = i+1
    return block


#=====================================================
# Transforms of the triangles
#=====================================================
def transform_triangle_forward(triangle, dtt, idx):
    if VERBOSE > 0:
        print("Transforming block to frequency domain...")

    transform = np.matmul(dtt,J[idx].astype(np.double))
    transform = np.reshape(transform,(transform.shape[0],1))

    if VERBOSE > 1:
        block = reshape_array_to_block(transform)
        print("Transformed block (rounded for visualization):")
        print(np.round(block))
        print("=====================")

    return transform

def transform_triangle_inverse(transform, dtt, idx):
    if VERBOSE > 0:
        print("Inversing transform to image domain...")

    reconstructed = linalg.lstsq(dtt, transform)[0]
    reconstructed = np.reshape(reconstructed,(reconstructed.shape[0],1))

    if VERBOSE > 1:
        block = reshape_array_to_block(reconstructed)
        print("Inversed block:")
        print(block)
        print("=====================")

    return reconstructed







if __name__ == '__main__':

    if not isdir(sys.argv[1]):
        exit("in_dir_path is not valid!")
    in_dir = sys.argv[1]

    if VERBOSE > 1:
        if ORTHOGONAL_TRANSFORM:
            print("ORHTOGONAL")
        else:
            print("NOT ORTHOGONAL")
        print("+++++++++++++++++++++++++++++")

    # Generate the basis Chebysev polynomials in triangle
    dtt = createTriangleDCT(BLOCK_SIZE);
    dot = createTriangleDCT_ortho(dtt)
    idx = np.tril(np.ones((BLOCK_SIZE,BLOCK_SIZE))) == 1



    # Show basis for visual understanding of frequencies
    if SHOW_BASIS:
        b = generate_basis_img(dtt, BLOCK_SIZE)
        b = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("basis", b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    energy = []
    for f in tqdm(files):
        img = cv2.imread(join(in_dir,f),cv2.IMREAD_GRAYSCALE)
        # Perform random cropping of the image SAMPLES times
        for _ in range(0,SAMPLES):
            x = random.randint(0,img.shape[1]-BLOCK_SIZE-1)
            y = random.randint(0,img.shape[0]-BLOCK_SIZE-1)

            J = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] - 128
            J = np.tril(J)

            if VERBOSE > 1:
                print("Input image block:")
                print(J)
                print("=====================")


            # Transform image to frequency domain
            if ORTHOGONAL_TRANSFORM:
                triangleTransformedPart = transform_triangle_forward(J,dot,idx)
            else:
                triangleTransformedPart = transform_triangle_forward(J,dtt,idx)

            # Add the transform to the energy list
            transform = reshape_array_to_block(triangleTransformedPart)
            energy.append(transform)

    # Average the energy
    energy = np.mean(np.array(energy), axis=0)
    energy = np.abs(energy)

    print("Transform 'energy':")
    print(np.round(energy).astype(np.int64))

    plt.imshow(energy)
    plt.show()
