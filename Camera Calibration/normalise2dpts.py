import numpy as np
import math

def normalise2dpts(pts):

    # Find the indices of the points that are not at infinity
    # finiteind = np.argwhere(abs(pts[2,:]) > np.spacing(np.float64(1)))
    finiteind = np.argwhere(abs(pts[2,:]) > np.finfo(np.float).eps)
    pts[0, finiteind] = np.divide(pts[0,finiteind], pts[2,finiteind])
    pts[1, finiteind] = np.divide(pts[1,finiteind], pts[2,finiteind])
    pts[2, finiteind] = 1

    newp = np.zeros((2,len(finiteind)))

    c = np.mean(pts[0:2,finiteind], axis=1)
    newp[0,finiteind] = pts[0,finiteind] - c[0]
    newp[1,finiteind] = pts[1,finiteind] - c[1]
    dist = np.sqrt(np.square(newp[0,finiteind]) + np.square(newp[1,finiteind]))
    meandist = np.mean(dist)
    scale = math.sqrt(2)/meandist
    
    T = np.array([[scale, 0, -scale*c[0]],
         [0, scale, -scale*c[1]],
         [0, 0, 1]], dtype = np.float32)
    newpts = T @ pts
    
    return newpts, T