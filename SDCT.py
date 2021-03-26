import numpy as np
import math
import config


#=====================================================
# Squared/Standar DCT transform
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
            c1 = math.cos( (2*x+1)*u*math.pi / (2*config.BLOCK_SIZE) )
            c2 = math.cos( (2*y+1)*v*math.pi / (2*config.BLOCK_SIZE) )
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

            c1 = math.cos( (2*x+1)*u*math.pi / (2*config.BLOCK_SIZE) )
            c2 = math.cos( (2*y+1)*v*math.pi / (2*config.BLOCK_SIZE) )
            sum = sum + a_u*a_v * coeff[v, u] * c1 * c2
    return sum / 4

def DCT_inv_transform(coeff):
    h, w = coeff.shape
    img = np.zeros(coeff.shape)
    for x in range(0, w):
        for y in range(0, h):
            img[y,x] = DCT_inv_coeff(coeff, x, y)
    return img
