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

import TDCT
import SDCT
import utils
import quantizer
import encoder


SHOW_BASIS = False
VERBOSE = 0

BLOCK_SIZE = 8

ORTHOGONAL_TRANSFORM = True

SAMPLES = 1

zig_TDCT = encoder.zig_TDCT

zig_SDCT = encoder.zig_SDCT



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






if __name__ == '__main__':

    filename = sys.argv[1]
    if not isfile(filename):
        print("Filename not valid")

    data = TDCT.init_data()

    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, None, fx=0.25, fy=0.25)
    img = cv2.flip(img,1)
    h, w = img.shape
    print(w,h)
    img_TDCT = np.zeros_like(img).astype(np.float32)
    img_SDCT = np.zeros_like(img).astype(np.float32)
    inverse_TDCT = np.zeros_like(img).astype(np.float32)
    inverse_SDCT = np.zeros_like(img).astype(np.float32)
    energy_TDCT = []
    energy_SDCT = []
    energy_zig_TDCT = []
    energy_zig_SDCT = []
    energy_zig_TDCT_rec = []
    energy_residuals_TDCT = []
    energy_residuals_SDCT = []
    energy_rle_TDCT = []
    energy_rle_SDCT = []

    q_factor = 30

    for x in tqdm(range(0,w,BLOCK_SIZE)):
        for y in range(0,h,BLOCK_SIZE):
            if x+BLOCK_SIZE<=w and y+BLOCK_SIZE<=h:
                J = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE].astype(np.float64)
                J = J - 128

                # SDCT over a block
                dct_transform = SDCT.DCT_transform(J)
                dct_transform = quantizer.quantize_sdct(dct_transform, q_factor)#transform / Q_sdct
                dct_transform = np.round(dct_transform)
                tmp = encoder.order_transform_coefficients(dct_transform, zig_SDCT)
                energy_rle_SDCT.append(len(encoder.run_length_encoding(encoder.residuals(tmp))))
                energy_residuals_SDCT.append(encoder.residuals(tmp))
                energy_zig_SDCT.append(tmp)
                energy_SDCT.append(dct_transform)
                img_SDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = dct_transform

                # TDCT over a block (two triangles)
                lowerT_transform, upperT_transform = TDCT.transform_block_TDCT_forward(J, data)
                transform = utils.join_lower_upper_triangles(lowerT_transform, upperT_transform)


                transform = quantizer.quantize_tdct(transform, q_factor) #transform / Q_tdct
                transform = np.round(transform)
                #transform = reduct_triangle_transform_coefficients(transform)

                tmp = encoder.order_transform_coefficients(transform, zig_TDCT)
                energy_rle_TDCT.append(len(encoder.run_length_encoding(encoder.residuals(tmp))))
                energy_residuals_TDCT.append(encoder.residuals(tmp))
                energy_zig_TDCT.append(tmp)
                energy_TDCT.append(transform)
                img_TDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = transform

                # Inverse SDCT
                dct_transform = quantizer.quantize_sdct_inv(dct_transform, q_factor) #dct_transform * Q_sdct
                inverse_SDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = SDCT.DCT_inv_transform(dct_transform) + 128

                # Reconstruct TDCT coefficients
                transform = encoder.unorder_transform_coefficients(tmp, zig_TDCT)
                #transform = unreduct_triangle_transform_coefficients(transform)
                transform = quantizer.quantize_tdct_inv(transform, q_factor)# transform * Q_tdct

                # Inverse TDCT
                lowerT_transform, upperT_transform = utils.get_upper_lower_triangles(transform)
                lowerT_transform = utils.reshape_block_to_array(lowerT_transform, BLOCK_SIZE)
                upperT_transform = utils.reshape_block_to_array(upperT_transform, BLOCK_SIZE-1)
                lowerT_inverse, upperT_inverse = TDCT.transform_block_TDCT_inverse(lowerT_transform, upperT_transform, data)
                inverse = utils.join_lower_upper_triangles(lowerT_inverse, upperT_inverse)
                inverse_TDCT[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = inverse + 128


    print("Energy TDCT")
    print(np.round(np.std(np.array(energy_TDCT), axis=0)))
    print("================")
    print("Energy SDCT")
    print(np.round(np.std(np.array(energy_SDCT), axis=0)))
    print("================")
    print("Energy zig TDCT (std)")
    print(np.round(np.std(np.array(energy_zig_TDCT), axis=0)))
    print("================")
    print("Energy zig SDCT (std)")
    print(np.round(np.std(np.array(energy_zig_SDCT), axis=0)))
    print("================")
    print("Energy zig TDCT (mean)")
    print(np.round(np.mean(np.array(energy_zig_TDCT), axis=0)))
    print("================")
    print("Energy zig SDCT (mean)")
    print(np.round(np.mean(np.array(energy_zig_SDCT), axis=0)))
    print("================")
    print("Energy residuals TDCT (mean)")
    print(np.round(np.mean(np.array(energy_residuals_TDCT), axis=0)))
    print("================")
    print("Energy residuals TDCT (std)")
    print(np.round(np.std(np.array(energy_residuals_TDCT), axis=0)))
    print("================")
    print("Energy residuals SDCT (mean)")
    print(np.round(np.mean(np.array(energy_residuals_SDCT), axis=0)))
    print("================")
    print("Energy residuals SDCT (std)")
    print(np.round(np.std(np.array(energy_residuals_SDCT), axis=0)))
    print("================")
    print("Energy rle TDCT")
    print(np.mean(np.array(energy_rle_TDCT)))
    print("================")
    print("Energy rle SDCT")
    print(np.mean(np.array(energy_rle_SDCT)))

    print(img.shape, inverse_TDCT.shape)
    psnr_TDCT = cv2.PSNR(img.astype(np.uint8), inverse_TDCT.astype(np.uint8))
    psnr_SDCT = cv2.PSNR(img.astype(np.uint8), inverse_SDCT.astype(np.uint8))

    print("PSNR TDCT: {}".format(psnr_TDCT))
    print("PSNR SDCT: {}".format(psnr_SDCT))

    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(np.mean(np.array(energy_TDCT), axis=0))
    plt.title("TDCT")
    f.add_subplot(1,2,2)
    plt.imshow(np.mean(np.array(energy_SDCT), axis=0))
    plt.title("SDCT")
    plt.show()
    plt.clf()


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


    #plt.savefig("TDCT-SDCT_comparison.png")


    #print("Incorrect inverse #: {}".format(e))
