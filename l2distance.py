import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

"""
||x-z|| = sqrt(||x||^2 + ||z||^2 - 2*x.z)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    xz = 2 * np.dot(np.transpose(X),Z)
    x_square = np.sum(X*X, axis = 0)
    z_square = np.sum(Z*Z, axis = 0)
    xx = np.reshape(x_square, (-1,1))
    zz = np.reshape(z_square, (1,-1))
    distance = np.tile(xx,m) + np.transpose(np.tile(zz.T,n)) - xz
    distance[distance < 0] = 0
    D = np.sqrt(distance)
    
    return D
