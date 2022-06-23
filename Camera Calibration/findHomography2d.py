import numpy as np
import math
from normalise2dpts import normalise2dpts

def findHomography2d(x1, x2):

    x1, T1 = normalise2dpts(x1)
    x2, T2 = normalise2dpts(x2)

    Npts = x1.shape[1]
    A = np.zeros((3*Npts, 9))
    O = np.zeros((1, 3))
    
    for n in range(Npts):
        X = x1[:,n]
        x = x2[0,n] 
        y = x2[1,n] 
        w = x2[2,n]
        
        A[3*n,:]   = [ *O[0],  *(-w*X),  *(y*X) ]
        A[3*n+1,:] = [ *(w*X), *O[0],    *(-x*X)]
        A[3*n+2,:] = [ *(-y*X),  *(x*X),  *O[0] ]

    U, S, V = np.linalg.svd(A)
    M = V[-1].reshape(3,3)  #transpose
    
    # M = np.linalg.lstsq(T2, (M @ T1), rcond=-1)[0]
    M = np.linalg.solve(T2, M) @ T1 

    return M