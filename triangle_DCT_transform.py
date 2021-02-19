import numpy as np
import cv2

import math

B = np.array([ [52, 55, 61, 66, 70, 61, 64, 73],
                [63, 59, 55, 90, 109, 85, 69, 72],
                [62, 59, 68, 113, 144, 104, 66, 73],
                [63, 58, 71, 122, 154, 106, 70, 69],
                [67, 61, 68, 104, 126, 88, 68, 70],
                [79, 65, 60, 70, 77, 68, 58, 75],
                [85, 71, 64, 59, 55, 61, 65, 83],
                [87, 79, 69, 68, 65, 76, 78, 94]])

def T_0_0():
    return 1.0

def T_n_0(n, t1, t2):
    return math.cos(2*n*math.pi*t2)*math.cos(2*n*math.pi*(t1-t2))

def T_0_n(n, t1, t2):
    return math.cos(n*math.pi*t1)*math.cos(n*math.pi*(t1-2*t2))

def T_k_l(k, l, t1,t2):
    c0 = math.cos(2*math.pi*(k*t1 + l*t2))
    c1 = math.cos(2*math.pi*((k+l)*t1 - l*t2))
    c2 = math.cos(2*math.pi*(k*t1 - (2*k+l)*t2))
    c3 = math.cos(2*math.pi*((k+l)*t1 - (2*k+l)*t2))

    return (c0 + c1 + c2 + c3) / 4

def generate_base_0_0(triangle_w, triangle_h):
    triangle = np.zeros((triangle_h,triangle_w))
    for y in range(0,triangle_h):
        for x in range(0, y+1):
            triangle[x,triangle_h-y-1] = 1.0
    return triangle

def generate_base_0_n(n, triangle_w, triangle_h):
    triangle = np.zeros((triangle_h,triangle_w))
    for y in range(0,triangle_h):
        t2 = y/triangle_h * 0.5
        for x in range(0, y+1):
            t1 = x/triangle_w * 0.5
            triangle[x,triangle_h-y-1] = T_0_n(n,t1,t2)
    return triangle

def generate_base_n_0(n, triangle_w, triangle_h):
    triangle = np.zeros((triangle_h,triangle_w))
    for y in range(0,triangle_h):
        t2 = y/triangle_h * 0.5
        for x in range(0, y+1):
            t1 = x/triangle_w * 0.5
            triangle[x,triangle_h-y-1] = T_n_0(n,t1,t2)
    return triangle


def generate_base(k, l, triangle_w, triangle_h):
    triangle = np.zeros((triangle_h,triangle_w))
    for y in range(0,triangle_h):
        t2 = y/triangle_h * 0.5
        for x in range(0, y+1):
            t1 = x/triangle_w * 0.5
            triangle[x,triangle_h-y-1] = T_k_l(k,l,t1,t2)
    return triangle

def triangle_DCT_coeff(triangle, t1, t2):
    triangle_w, triangle_h = triangle.shape

    # Compute the frequence (u,v) coeff sum
    sum = 0
    for k in range(0, triangle_w):
        for l in range(0,triangle_h-k):
            if k==0 and l==0:
                base = T_0_0()
            elif k==0:
                base = T_0_n(l, t1, t2)
            elif l==0:
                base = T_n_0(k, t1, t2)
            else:
                base = T_k_l(k, l, t1, t2)
            sum = sum + triangle[k,l] * base
    return sum

def triangle_DCT_inv_coeff(triangle, k, l):
    triangle_w, triangle_h = triangle.shape

    # Compute the frequence (u,v) coeff sum
    sum = 0
    for y in range(0, triangle_w):
        t2 = y / triangle_h * 0.5
        for x in range(0,triangle_h-y):
            t1 = x/triangle_w * 0.5
            if k==0 and l==0:
                base = T_0_0()
            elif k==0:
                base = T_0_n(l, t1, t2)
            elif l==0:
                base = T_n_0(k, t1, t2)
            else:
                base = T_k_l(k, l, t1, t2)
            sum = sum + triangle[k,l] * base
    return sum



def transform_triangle(triangle): # remember that the triangle is a squared block and I have to treat only the reuqired pixels!
    triangle_w, triangle_h = triangle.shape

    transform = np.zeros((triangle_h,triangle_w))
    for y in range(0, triangle_h):
        t2 = y / triangle_h * 0.5
        for x in range(0,y+1):
            t1 = x/triangle_w * 0.5
            transform[triangle_h-y-1,x] = triangle_DCT_coeff(triangle, t1, t2)
    return transform


def transform_inv_triangle(triangle_coeff): #remember that the triangle is a squared block and I have to treat only the reuqired pixels!
    w, h = triangle_coeff.shape

    triangle = np.zeros((h,w))
    for l in range(0, h):
        for k in range(0,y+1):
            triangle[x,y] = triangle_DCT_inv_coeff(triangle_coeff, k, l)
    return triangle


def generate_basis(triangle_w, triangle_h):

    for k in range(0, triangle_w):
        for l in range(0,triangle_h-k):

            if k==0 and l==0:
                base = generate_base_0_0(triangle_w, triangle_h)
            elif k==0:
                base = generate_base_0_n(l, triangle_w, triangle_h)
            elif l==0:
                base = generate_base_n_0(k, triangle_w, triangle_h)
            else:
                base = generate_base(k, l, triangle_w, triangle_h)

            base = base + 1
            base = (base/2*255).astype(np.uint8)
            cv2.imwrite("8x8_basis/T_{}_{}.png".format(k, l), base)
            base = cv2.resize(base, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite("8x8_basis_resized/T_{}_{}.png".format(k, l), base)



if __name__ == '__main__':
    triangle = np.ones((8,8)) * 255
    #print(B-128)
    transform = transform_triangle(triangle)
    print(transform.astype(np.int32))
    transform = cv2.resize(transform, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('coeffs', transform)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
