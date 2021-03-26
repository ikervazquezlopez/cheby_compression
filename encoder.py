from dahuffman import HuffmanCodec
import cv2
import numpy as np
import config


BLOCK_SIZE = config.BLOCK_SIZE


zig_TDCT = [[0,0],[0,1],[1,1],[1,2],[1,0],[0,2],[2,2],[2,3],
        [2,0],[0,3],[2,1],[1,3],[3,3],[3,4],[3,0],[0,4],
        [3,1],[1,4],[3,2],[2,4],[4,4],[4,5],[5,5],[5,6],
        [6,6],[6,7],[4,0],[0,5],[4,1],[1,5],[4,2],[2,5],
        [4,3],[3,5],[5,0],[0,6],[5,1],[1,6],[5,2],[2,6],
        [5,3],[3,6],[5,4],[4,6],[6,0],[0,7],[6,1],[1,7],
        [6,2],[2,7],[6,3],[3,7],[6,4],[4,7],[6,5],[5,7],
        [7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7]]

zig_SDCT = [[0,0],[0,1],[1,0],[2,0],[1,1],[0,2],[0,3],[1,2],
        [2,1],[3,0],[4,0],[3,1],[2,2],[1,3],[0,4],[0,5],
        [1,4],[2,3],[3,2],[4,1],[5,0],[6,0],[5,1],[4,2],
        [3,3],[2,4],[1,5],[0,6],[0,7],[1,6],[2,5],[3,4],
        [4,3],[5,2],[6,1],[7,0],[7,1],[6,2],[5,3],[4,4],
        [3,5],[2,6],[1,7],[2,7],[3,6],[4,5],[5,4],[6,3],
        [7,2],[7,3],[6,4],[5,5],[4,6],[3,7],[4,7],[5,6],
        [6,5],[7,4],[7,5],[6,6],[5,7],[6,7],[7,6],[7,7]]



# Rearranges the transform coefficients base on 'order' list (Customized JPEG ZigZag)
def order_transform_coefficients(transform, order):
    return [transform[y,x] for y,x in order]

# Inverse of order_transform_coefficients if 'order' is the same.
def unorder_transform_coefficients(array, order):
    tmp = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i,c in zip(list(range(len(array))), order):
        tmp[c[0],c[1]] = array[i]
    return tmp


def residuals(array):
    a = np.array(array)
    b = np.insert(a,0,0)
    return a - b[:-1]



def run_length_encoding(array):
    encoded = []
    i=0
    for k in range(array.shape[0]):
        if array[k] == 0:
            i = i+1
        else:
            encoded.append((i,array[k]))
            i = 0
    return encoded
