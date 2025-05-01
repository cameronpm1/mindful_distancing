import numpy as np

def angle_between_vectors(vec1,vec2):
    #angle between two vectos w/ direction (clockwise negative)
    det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    return np.atan2(det,dot)