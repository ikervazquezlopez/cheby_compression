import numpy as np
import math
import config


Q_sdct = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

Q_tdct_low = np.array([[16,   0,  0,  0,  0, 0,  0,  0],
                        [12, 11,  0,  0,  0, 0,  0,  0],
                        [14, 12, 10,  0,  0, 0,  0,  0],
                        [14, 13, 14, 16,  0, 0,  0,  0],
                        [18, 17, 16, 19, 24, 0,  0,  0],
                        [24, 22, 22, 24, 26, 40, 0,  0],
                        [49, 35, 37, 29, 40, 58, 51, 0],
                        [72, 64, 55, 56, 51, 57, 60, 61]])


Q_tdct_up = np.array([[0,  16, 12, 14, 14, 18, 24, 49],
                        [0, 0, 11, 12, 13, 17, 22, 35],
                        [0, 0,  0, 10, 14, 16, 22, 35],
                        [0, 0,  0,  0, 16, 19, 24, 29],
                        [0, 0,  0,  0,  0, 24, 26, 40],
                        [0, 0,  0,  0,  0,  0, 40, 58],
                        [0, 0,  0,  0,  0,  0,  0, 51],
                        [0, 0,  0,  0,  0,  0,  0,  0]])

Q_tdct = Q_tdct_low + Q_tdct_up



def Q_SDCT(q_factor = 50):
    if q_factor > 100:
        return np.ones_like(Q_sdct)

    # Compute S
    S = 0
    if q_factor < 50:
        S = 5000 / q_factor
    else:
        S = 200 - 2*q_factor

    Qm = np.floor((S*Q_sdct + 50) / 100) # compute new Q matrix
    Qm[Qm == 0] = 1 # prevents dividint by 0

    return Qm

def Q_TDCT(q_factor = 50):
    if q_factor > 100:
        return np.ones_like(Q_tdct)

    # Compute S
    S = 0
    if q_factor < 50:
        S = 5000 / q_factor
    else:
        S = 200 - 2*q_factor

    Qm = np.floor((S*Q_tdct + 50) / 100) # compute new Q matrix
    Qm[Qm == 0] = 1 # prevents dividint by 0

    return Qm



def quantize_sdct(block, q_factor = 50):
    return np.round(block / Q_SDCT(q_factor))

def quantize_tdct(block, q_factor = 50):
    return np.round(block / Q_TDCT(q_factor))

def quantize_sdct_inv(block, q_factor = 50):
    return np.round(block * Q_SDCT(q_factor))

def quantize_tdct_inv(block, q_factor = 50):
    return np.round(block * Q_TDCT(q_factor))
