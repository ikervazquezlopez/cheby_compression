import numpy as np
import cv2

import math



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


def join_basis(triangle_w,triangle_h):
    basis = np.zeros((triangle_h*triangle_h, triangle_w*triangle_w))
    for k in range(0, triangle_w):
        for l in range(0,triangle_h-k):
            if k==0 and l==0:
                base = generate_base_0_0(triangle_w, triangle_h)
            if k==0:
                base = generate_base_0_n(l, triangle_w, triangle_h)
            elif l==0:
                base = generate_base_n_0(k, triangle_w, triangle_h)
            else:
                base = generate_base(k, l, triangle_w, triangle_h)
            basis[k*triangle_w:k*triangle_w+triangle_w, l*triangle_h:l*triangle_h+triangle_h] = base + 1

    basis = (basis/2*255).astype(np.uint8)
    basis = cv2.resize(basis, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("joint_basis/{}x{}.png".format(triangle_w,triangle_h), basis)


if __name__ == '__main__':
    generate_basis(8, 8)
    join_basis(16,16)
