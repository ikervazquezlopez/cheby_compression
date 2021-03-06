import numpy as np
import cv2
import math

block_size = (8, 8)
PI = math.pi

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])



def compute_Q_matrix(quality_factor = 50):
    # Compute S
    S = 0
    if quality_factor < 50:
        S = 5000 / quality_factor
    else:
        S = 200 - 2*quality_factor

    Qm = np.floor((S*Q + 50) / 100) # compute new Q matrix
    Qm[Qm == 0] = 1 # prevents dividint by 0

    return Qm


def generate_image_blocks(img):
    h, w = img.shape
    grid_size_x = int( w / block_size[1] )
    grid_size_y = int( h / block_size[0] )
    block_img = np.zeros((grid_size_y, grid_size_x, block_size[0], block_size[1]), dtype=img.dtype)
    for x in range(0, grid_size_x):
        for y in range(0, grid_size_y):
            y_start = y*block_size[0]
            y_end = y_start + block_size[0]
            x_start = x*block_size[1]
            x_end = x_start + block_size[1]
            block_img[y, x] = img[y_start:y_end, x_start:x_end]
    return block_img

def generate_image_from_blocks(blocks):
    gh, gw, bh, bw = blocks.shape
    img = np.zeros((gh*bh, gw*bw))
    for u in range(0, bw):
        for v in range(0, bh):
            for x in range(0, gw):
                for y in range(0, gh):
                    img[y*bh+v, x*bw+u] = blocks[y, x, v, u]
    return img

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
            c1 = math.cos( (2*x+1)*u*PI / 16 )
            c2 = math.cos( (2*y+1)*v*PI / 16 )
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

            c1 = math.cos( (2*x+1)*u*PI / 16 )
            c2 = math.cos( (2*y+1)*v*PI / 16 )
            sum = sum + a_u*a_v * coeff[v, u] * c1 * c2
    return sum / 4

def sort_DCT_coefficients(blocks):
    gh, gw, bv, bu = blocks.shape
    arranged = np.zeros_like(blocks)
    for u in range(0, bu):
        for v in range(0, bv):
            for x in range(0, gw):
                for y in range(0, gh):
                    arranged[y, x, v, u] = blocks[v, u, y, x]
    return arranged


if __name__ == '__main__':
    img = cv2.imread("arranged_coefficients.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    img = img - 128


    # Split image into block_size (8x8) blocks
    blocks = generate_image_blocks(img)
    grid_size_x = blocks.shape[1]
    grid_size_y = blocks.shape[0]


    DCT_blocks = np.zeros_like(blocks)
    for x in range(0,grid_size_x):
        for y in range(0,grid_size_y):
            DCT_blocks[y, x] = DCT_transform(blocks[y,x]) # Transform step
            DCT_blocks[y, x] = DCT_blocks[y, x] / Q  #Quantization step

    #arranged_DCT_coeffs = sort_DCT_coefficients(DCT_blocks)
    arranged_DCT_coeffs = np.moveaxis(DCT_blocks, 2, 0)
    arranged_DCT_coeffs = np.moveaxis(arranged_DCT_coeffs, 3, 1)
    #arranged_DCT_coeffs = DCT_blocks
    DCT_block_image = generate_image_from_blocks(DCT_blocks)
    arranged_DCT_image = generate_image_from_blocks(arranged_DCT_coeffs)

    print(np.sum(DCT_block_image), DCT_block_image.shape, DCT_block_image.dtype)
    print(np.sum(arranged_DCT_image), arranged_DCT_image.shape, arranged_DCT_image.dtype)

    cv2.imwrite("image_from_blocks.png", arranged_DCT_image)
