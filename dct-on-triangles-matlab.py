import numpy as np
import cv2
import math
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs


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
    print(triangleDCT.shape)
    return triangleDCT


if __name__ == '__main__':
    sizeN = 16

    dtt = createTriangleDCT(sizeN);
    print(dtt[0])

    b = generate_basis_img(dtt, sizeN)
    b = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("basis", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    J = np.ones((sizeN, sizeN))
    J = np.tril(J)
    idx = np.tril(np.ones((sizeN,sizeN))) == 1

    triangleTransformedPart = np.matmul(dtt,J[idx].astype(np.double))
    triangleTransformedPart = np.round(triangleTransformedPart, decimals=4)

    """
    # throw away three quarter of the frequencies
    triangleTransformedPart[int((sizeN/2)*(sizeN/2)):-1] = 0
    triangleTransformedPart[-1] = 0


    # transform to image space domain again
    print(dtt.shape, triangleTransformedPart.shape)
    tmp = np.uint8(linalg.lstsq(dtt, triangleTransformedPart))
    print(J)
    """

    # Convert to image to visually check the image energy in the frequency domain
    i=0
    triangleDCT = np.empty((sizeN,sizeN))
    for y in range(0, sizeN):
        for x in range(0, y+1):
            triangleDCT[y,x] = triangleTransformedPart[i]
            i= i+1

    np.set_printoptions(suppress=True)
    print("Triangular DCT coefficients")
    print(triangleDCT)
    img = triangleDCT - np.min(triangleDCT)
    img[img<0] = 0
    img = img / np.max(img)
    img = (img*255).astype(np.uint8)
    print("Triangular UINT8 image values")
    print(img)
    img = cv2.resize(img, None, fx=32, fy=32, interpolation = cv2.INTER_NEAREST)
    cv2.imshow("out", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("triangleDCT.png", img)
