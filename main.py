import numpy as np
import cv2
import math

PI = math.pi

test_block = np.array([ [52, 55, 61, 66, 70, 61, 64, 73],
                        [63, 59, 55, 90, 109, 85, 69, 72],
                        [62, 59, 68, 113, 144, 104, 66, 73],
                        [63, 58, 71, 122, 154, 106, 70, 69],
                        [67, 61, 68, 104, 126, 88, 68, 70],
                        [79, 65, 60, 70, 77, 68, 58, 75],
                        [85, 71, 64, 59, 55, 61, 65, 83],
                        [87, 79, 69, 68, 65, 76, 78, 94]])

test_block_bin = np.array([ [255, 255, 255, 255, 255, 255, 255, 255],
                        [255, 255, 255, 255, 255, 255, 255, 255],
                        [255, 255, 255, 255, 255, 255, 255, 255],
                        [255, 255, 255, 255, 255, 255, 255, 255],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

block_size = (8, 8)

CP = np.zeros(block_size)

def generate_CP():
    for n in range(0, block_size[1]):
        for k in range(0, block_size[1]):
            T_1D(n, k)

def T_1D(n, k):
    x = math.cos(k/(2*block_size[1]))
    print([n,k], x)
    if n == 0:
        CP[k, 0] = 1
    elif n == 1:
        CP[k,1] = x
    else:
        CP[k,n] = 2*x*CP[k,n-1] - CP[k,n-2]

def T(k,l, theta1, theta2):
    if l == 0:
        return math.cos(2*PI*k*theta2) * math.cos(2*PI*k*(theta1-theta2))
    elif k == 0:
        return math.cos(PI*k*theta1) * math.cos(PI*k*(theta1-2*theta2))
    else:
        return 0.25 * (math.cos(2*PI*(k*theta1 + l*theta2)) +
                    math.cos(2*PI*((k+l)*theta1 - l*theta2)) +
                    math.cos(2*PI*(k*theta1 - (2*k+l)*theta2)) +
                    math.cos(2*PI*((k+l)*theta1 - (2*k+l)*theta2)) )

def cheby_coeff_square(img, x1, x2):
    h, w = img.shape
    i = w
    sum = 0

    for l in range(0, h):
        y = math.cos(l/(2*block_size[1]))
        uy = 1/(math.sqrt(1-y*y)+0.000001)
        for k in range(0,w):
            x = math.cos(k/(2*block_size[1]))
            ux = 1/(math.sqrt(1-x*x)+0.000001)
            sum = sum + ux*uy*img[l,k]*CP[l,x1]*CP[k,x2]
        #i = i-1
    return 4*sum/(PI*PI) # WARNING with this, I am not sure about the division by 4

def cheby_inv_coeff_square(img, x, y):
    h, w = img.shape
    i = w
    sum = 0
    for l in range(0, h):
        for k in range(0,w):
            sum = sum + coeff[l,k]*CP[l,x]*CP[k,y]
        #i = i-1
    return sum # WARNING with this, I am not sure about the division by 4

def cheby_transform_square(img):
    h, w = img.shape
    coeffs = np.zeros(img.shape)
    i = w
    for u in range(0, w):
        for v in range(0, h):
            coeffs[v, u] = cheby_coeff_square(img, u, v)#DCT_coeff(img, 0.5*u/w, 0.5*v/h)
    return coeffs

def cheby_inv_transform_square(coeffs):
    h, w = coeffs.shape
    img = np.zeros(coeffs.shape)
    i = w
    for x in range(0, w):
        for y in range(0, h):
            img[x, y] = cheby_coeff_square(coeffs, x, y)#DCT_coeff(img, 0.5*u/w, 0.5*v/h)
    return img



def cheby_coeff(img, theta1, theta2):
    h, w = img.shape
    i = w
    sum = 0
    for l in range(0, h):
        for k in range(0,w):
            if k >= i:
                continue
            sum = sum + coef[l,k]*T(k,l,theta1, theta2)
        #i = i-1
    return sum/4 # WARNING with this, I am not sure about the division by 4

def cheby_coeff_inv(coeff, theta1, theta2):
    h, w = img.shape
    i = w
    sum = 0
    for l in range(0, h):
        for k in range(0,w):
            if k >= i:
                continue
            sum = sum + coeff[l,k]*T(k,l,theta1, theta2)
        #i = i-1
    return sum/4 # WARNING with this, I am not sure about the division by 4

def cheby_transform(img):
    h, w = img.shape
    coeffs = np.zeros(img.shape)
    i = w
    for u in range(0, w):
        for v in range(0, h):
            if u == 0:
                coeffs[v-1, w-1] = cheby_coeff_square(img, u, v)#DCT_coeff(img, 0.5*u/w, 0.5*v/h)
            if v == 0:
                coeffs[h-1, u-1] = cheby_coeff_square(img, u, v)#DCT_coeff(img, 0.5*u/w, 0.5*v/h)
            else:
                coeffs[v-1, u-1] = cheby_coeff_square(img, u, v)#DCT_coeff(img, 0.5*u/w, 0.5*v/h)
    return coeffs

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

def DCT_inv_transform(coeff):
    h, w = coeff.shape
    img = np.zeros(coeff.shape)
    for x in range(0, w):
        for y in range(0, h):
            img[y,x] = DCT_inv_coeff(coeff, x, y)
    return img


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



if __name__ == '__main__':

    generate_CP()

    #img = cv2.imread("test_pano.png", cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, None, fx=0.0625, fy=0.0625)
    print("Generating blocks...")
    #blocks = generate_image_blocks(img)
    #bh, bw, _, _ = blocks.shape
    dct = np.zeros(block_size)
    dct2 = np.zeros(block_size)
    #print(bw, bh)
    print("Processing blocks...")
    dct = cheby_transform_square(test_block_bin/255)#np.full(block_size, 2))
    #dct = cheby_inv_transform_square(dct)
    """
    for bx in range(0,bw):
        print(bx, bw)
        for by in range(0,bh):
            freq = cheby_transform(blocks[by,bx])
            dct = dct + np.abs(freq)
            dct2 = dct2 + np.abs(DCT_transform(freq))
    #dct = dct / (bx*by)
    #dct2 = dct2 / (bx*by)
    """
    cv2.imwrite("out/dct_avg.png", 255*np.abs(dct)/np.max(np.abs(dct)))
    #cv2.imwrite("out/dct_avg.png", 255*np.abs(dct)/np.max(np.abs(dct)))
    #cv2.imwrite("out/dct2_avg.png", 255*np.abs(dct2)/np.max(np.abs(dct2)))


    """
    w = 8
    h = 8
    x = 140
    y = 200
    img = cv2.imread("test.jpeg", cv2.IMREAD_GRAYSCALE)

    test_block_shift = test_block - 128
    G = DCT_transform(test_block_shift)
    IG = DCT_inv_transform(G) + 128
    C = cheby_transform(test_block_shift)
    Gp = DCT_transform(G)
    GP_inv = DCT_inv_transform(DCT_inv_transform(Gp))

    print("test_block DCT")
    print(G)
    print("test_block cheby")
    print(C)
    blocks = generate_image_blocks(img)

    block = img[y:y+h, x:x+w]
    coeff = np.zeros_like(block, dtype=np.float64)
    print(block.shape, coeff.shape)
    for u in range(0, coeff.shape[1]):
        for v in range(0, coeff.shape[0]):
            coeff[v,u] = cheby_coeff(block, h, w )

    #coeff = cv2.resize(coeff, (128,128), cv2.INTER_NEAREST)
    #block = cv2.resize(block, (128,128), cv2.INTER_NEAREST)



    #cv2.imwrite("block.png", block)
    #cv2.imwrite("coeff.png", coeff)
    #print(coeff)
    print(np.max(np.abs(G)))
    print(255*np.abs(G)/np.max(np.abs(G)))
    cv2.imwrite("out/source.png", test_block)
    cv2.imwrite("out/cheby_transform.png", 255*np.abs(C)/np.max(np.abs(C)))
    cv2.imwrite("out/DCT_transform_of_cheby.png", 255*np.abs(Gp)/np.max(np.abs(Gp)))
    cv2.imwrite("out/DCT_transform.png", 255*np.abs(G)/np.max(np.abs(G)))
    cv2.imwrite("out/DCT_reconstructed.png", GP_inv + 128)

    print(np.max(np.abs(IG-test_block)))

    #cv2.imwrite("out/block00.png", blocks[20,20])
    #cv2.imwrite("out/block01.png", blocks[20,21])
    #cv2.imwrite("out/block02.png", blocks[20,22])
    #cv2.imwrite("out/block10.png", blocks[21,20])
    #cv2.imwrite("out/block20.png", blocks[22,20])
    """
