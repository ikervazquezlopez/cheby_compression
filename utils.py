import numpy as np
import math
import config

BLOCK_SIZE = config.BLOCK_SIZE





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

def get_upper_lower_triangles(block):
    lower = get_lower_triangle(block)
    upper = get_upper_triangle(block)
    return (lower, upper)


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


def reduct_triangle_transform_coefficients(t_block):

    redux = np.copy(t_block)
    for x in range(BLOCK_SIZE-1, 2, -1):
        redux[0,x] = t_block[0,x] - t_block[0,x-1]
    for y in range(BLOCK_SIZE-1, 1, -1):
        redux[y,0] = t_block[y,0] - t_block[y-1, 0]

    for y in range(2, BLOCK_SIZE):
        redux[y,y] = t_block[y,y] - t_block[y-1, y-1]
    for  y in range(2, BLOCK_SIZE-1):
        redux[y,y+1] = t_block[y,y+1] - t_block[y-1, y]

    return redux

def unreduct_triangle_transform_coefficients(t_block):

    redux = np.copy(t_block)
    for x in range(3, BLOCK_SIZE):
        redux[0,x] = redux[0,x-1] + redux[0,x]
    for y in range(2, BLOCK_SIZE):
        redux[y,0] = redux[y-1, 0] + redux[y,0]

    for y in range(2, BLOCK_SIZE):
        redux[y,y] = redux[y-1, y-1] + redux[y,y]
    for  y in range(2, BLOCK_SIZE-1):
        redux[y,y+1] = redux[y-1, y] + redux[y,y+1]

    return redux
