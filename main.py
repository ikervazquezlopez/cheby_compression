import numpy as np
import cv2
import math

PI = math.pi

block_size = (8, 8)

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

def cheby_coeff(img, h, w):
    i = w
    sum = 0
    for l in range(0, h):
        for k in range(0,w):
            if k >= i:
                continue
            theta1 = 0.5 * k/w
            theta2 = 0.5 * l/h
            sum = sum + img[l,k]*math.cos(PI*k/w)*math.cos(PI*l/h)#T(k,l,theta1, theta2)
        i = i-1
    return sum/4 # WARNING with this, I am not sure about the division by 4

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

    w = 8
    h = 8
    x = 140
    y = 200
    img = cv2.imread("test.jpeg", cv2.IMREAD_GRAYSCALE)

    blocks = generate_image_blocks(img)

    block = img[y:y+h, x:x+w]
    coeff = np.zeros_like(block, dtype=np.float64)
    print(block.shape, coeff.shape)
    for u in range(0, coeff.shape[1]):
        for v in range(0, coeff.shape[0]):
            coeff[v,u] = cheby_coeff(block, h, w )

    #coeff = cv2.resize(coeff, (128,128), cv2.INTER_NEAREST)
    #block = cv2.resize(block, (128,128), cv2.INTER_NEAREST)



    cv2.imwrite("block.png", block)
    cv2.imwrite("coeff.png", coeff)
    print(coeff)


    cv2.imwrite("out/block00.png", blocks[20,20])
    cv2.imwrite("out/block01.png", blocks[20,21])
    cv2.imwrite("out/block02.png", blocks[20,22])
    cv2.imwrite("out/block10.png", blocks[21,20])
    cv2.imwrite("out/block20.png", blocks[22,20])
