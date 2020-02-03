import numpy as np
import cv2
import math

PI = math.pi

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



if __name__ == '__main__':

    w = 4
    h = 4
    basis = np.zeros((h,w))

    i = w
    for l in range(0, h):
        for k in range(0,w):
            if k >= i:
                continue
            theta1 = 0.5 * k/w
            theta2 = 0.5 * l/h
            basis[l,k] = T(k,l,theta1, theta2)
        i = i-1

    print(basis)
