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

SAMPLES = 1





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

# Initializes the required data for the TDCT
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



def reshape_array_to_block(triangle_array, size):
    block = np.zeros((size,size))
    i = 0
    for y in range(0, size):
        for x in range(0, y+1):
            block[y,x] = triangle_array[i]
            i = i+1
    return block

def reshape_block_to_array(triangle_block, size):
    arr = []
    for y in range(0, size):
        for x in range(0, y+1):
            arr.append(triangle_block[y,x])

    return np.array(arr)


def get_lower_triangle(block):
    T = np.tril(block)
    return T

def get_upper_triangle(block):
    T = block.T
    T = T[1:,:BLOCK_SIZE-1]
    T = np.tril(T)
    return T


# Join two triangles (BLOCK_SIZE, BLOCK_SIZE) and (BLOCK_SIZE-1, BLOCK_SIZE-1)
def join_lower_upper_triangles(lowerT, upperT):

    transform_lowerT = reshape_array_to_block(lowerT, BLOCK_SIZE)
    transform_upperT = reshape_array_to_block(upperT, BLOCK_SIZE-1)
    tmp = np.zeros((BLOCK_SIZE,BLOCK_SIZE))
    tmp[:-1,1:] = transform_upperT.T
    #transform = cv2.flip(transform_lowerT,-1)
    transform = transform_lowerT + tmp
    #transform = cv2.flip(transform,-1)

    return transform

#=====================================================
# Transforms of the triangles
#=====================================================
def transform_triangle_forward(triangle, dtt, idx):
    if VERBOSE > 0:
        print("Transforming block to frequency domain...")

    transform = np.matmul(dtt,triangle[idx].astype(np.double))
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

    #transform = reshape_block_to_array(transform, transform.shape[0])
    reconstructed = linalg.lstsq(dtt, transform)[0]
    reconstructed = np.reshape(reconstructed,(reconstructed.shape[0],1))

    if VERBOSE > 1:
        block = reshape_array_to_block(reconstructed)
        print("Inversed block:")
        print(block)
        print("=====================")

    return reconstructed



# Transform block into the triangle block frequency domain
def transform_block_TDCT_forward(block, data):
    J = get_lower_triangle(block)
    Jp = get_upper_triangle(block)

    lowerT = transform_triangle_forward(J,data['dot'],data['idx'])
    upperT = transform_triangle_forward(Jp,data['dotp'],data['idxp'])

    return lowerT, upperT

def transform_block_TDCT_inverse(lowerT, upperT, data):

    lowerT = transform_triangle_inverse(lowerT,data['dot'],data['idx'])
    upperT = transform_triangle_inverse(upperT,data['dotp'],data['idxp'])

    return lowerT, upperT

#=====================================================
# Squared DCT transform
#=====================================================

def DCT_coeff(block, u, v):
    bh, bw = block.shape

    # Normalization factors
    a_u = 1
    a_v = 1
    if u == 0:
        a_u = 1/math.sqrt(2)
    if v == 0:
        a_v = 1/math.sqrt(2)

    # Compute the frequence (u,v) coeff sum
    sum = 0
    for x in range(bw):
        for y in range(bh):
            c1 = math.cos( (2*x+1)*u*math.pi / 16 )
            c2 = math.cos( (2*y+1)*v*math.pi / 16 )
            sum = sum + block[y,x] * c1 * c2
    sum = 0.25 * a_u*a_v * sum

    return sum

def DCT_transform(img):
    h, w = img.shape
    coeffs = np.zeros(img.shape)
    for u in range(0, w):
        for v in range(0, h):
            coeffs[v, u] = DCT_coeff(img, u, v)
    return coeffs

def DCT_inv_coeff(coeff, x, y):
    h, w = coeff.shape

    sum = 0
    for u in range(0, w):
        for v in range(0, h):
            # Normalization factors
            a_u = 1
            a_v = 1
            if u == 0:
                a_u = 1/math.sqrt(2)
            if v == 0:
                a_v = 1/math.sqrt(2)

            c1 = math.cos( (2*x+1)*u*math.pi / 16 )
            c2 = math.cos( (2*y+1)*v*math.pi / 16 )
            sum = sum + a_u*a_v * coeff[v, u] * c1 * c2
    return sum / 4

def DCT_inv_transform(coeff):
    h, w = coeff.shape
    img = np.zeros(coeff.shape)
    for x in range(0, w):
        for y in range(0, h):
            img[y,x] = DCT_inv_coeff(coeff, x, y)
    return img



if __name__ == '__main__':

    filename = sys.argv[1]
    if not isfile(filename):
        print("Filename not valid")

    data = init_data()

    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    img = cv2.flip(img, 1)
    h, w = img.shape
    img_TDCT = np.zeros_like(img)
    img_SDCT = np.zeros_like(img)
    inverse_TDCT = np.zeros_like(img)
    inverse_SDCT = np.zeros_like(img)
    # Perform random cropping of the image SAMPLES times
    for x in range(0,w,BLOCK_SIZE):
        for y in range(0,h,BLOCK_SIZE):
            if x+BLOCK_SIZE<=w and y+BLOCK_SIZE<=h:


                J = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE].astype(np.float64)
                J = J - 128

                # SDCT over a block
                dct_transform = DCT_transform(J)
                img_SDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = dct_transform
                inverse_SDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = DCT_inv_transform(dct_transform) + 128

                # TDCT overe a block (two triangles)
                lowerT_transform, upperT_transform = transform_block_TDCT_forward(J, data)
                transform = join_lower_upper_triangles(lowerT_transform, upperT_transform)
                img_TDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = transform

                lowerT_inverse, upperT_inverse = transform_block_TDCT_inverse(lowerT_transform, upperT_transform, data)
                inverse = join_lower_upper_triangles(lowerT_inverse, upperT_inverse)

                inverse_TDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = inverse + 128

    print(np.min(inverse_SDCT))
    """
    cv2.imshow("out", inverse_TDCT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    f = plt.figure()
    f.add_subplot(2,3,1)
    plt.imshow(img)
    plt.title("Original img")
    f.add_subplot(2,3,2)
    plt.imshow(img_TDCT)
    plt.title("TDCT")
    f.add_subplot(2,3,3)
    plt.imshow(inverse_TDCT)
    plt.title("Inverse of TDCT")

    f.add_subplot(2,3,4)
    plt.imshow(img)
    plt.title("Original img")
    f.add_subplot(2,3,5)
    plt.imshow(img_SDCT)
    plt.title("SDCT")
    f.add_subplot(2,3,6)
    plt.imshow(inverse_SDCT)
    plt.title("Inverse of SDCT")
    plt.show()





    #print("Incorrect inverse #: {}".format(e))
